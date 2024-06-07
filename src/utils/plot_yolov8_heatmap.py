import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam import (
    GradCAM, HiResCAM, GradCAMElementWise, AblationCAM, XGradCAM, 
    GradCAMPlusPlus, ScoreCAM, LayerCAM, EigenCAM, EigenGradCAM, RandomCAM)
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from ultralytics.nn.tasks import attempt_load_weights


def letterbox(im, new_shape=(640, 640), auto=True, scaleFill=False, scaleup=True, center=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    return im, ratio, (dw, dh)


class Yolov8ActivationsAndGradients(ActivationsAndGradients):
    def post_process(self, result):
        logits_ = result[:, 4:]  # [1, num_classes + 4, num_detections] -> [1, 1, 78200]
        boxes_ = result[:, :4]  # [1, num_classes + 4, num_detections] -> [1, 4, 78200]
        _, indices = torch.sort(logits_.max(1)[0], descending=True)  # 按照最大置信度 降序排序 indices: [1, 78200]
        return (
            torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]],  # logits_ 按照 检测框的类别置信度降序排序
            torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]],  # boxes_ 按照 检测框的类别置信度降序排序 xywh
            xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()  # boxes_ 按照 检测框的类别置信度降序排序 xxyy
        )

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)  # 前向传播 x: [1, 3, 736, 1280]  model_output: ([1, 5, 78200], [4个head的输出]) 78200=4个head的anchor总数
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])  # post_result: [78200, 1]  pre_post_boxes: [78200, 4]
        return [[post_result, pre_post_boxes]] 


class Yolov8Target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.output_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data  # Yolov8ActivationsAndGradients的返回值
        result = []
        for i in range(int(post_result.size(0) * self.ratio)):  # 只考虑高置信度的anchor
            if float(post_result[i].max()) < self.conf:  # 置信度
                break
            if self.output_type == 'class' or self.output_type == 'all':
                result.append(post_result[i].max())  # 追加置信度
            elif self.output_type == 'box' or self.output_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])  # 追加bbox各坐标
        return sum(result)  # 累加(得到新的tensor) 相当于 分类任务中的 最终的class


class Yolov8Heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, iou_thres, max_det, ratio, save_cam, show_box, renormalize):

        # 设备和模型权重/信息
        self.device = torch.device(device)
        self.ckpt = torch.load(weight)

        # 数据集参数
        self.imgsz = self.ckpt['train_args']['imgsz']
        self.class_names = self.ckpt['model'].names

        # 模型参数
        self.model = attempt_load_weights(weight, device)
        self.model.info()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()

        # NMS参数
        self.conf_threshold = conf_threshold
        self.iou_thres = iou_thres
        self.max_det = max_det

        # 可视化参数
        self.save_cam = save_cam
        self.show_box = show_box
        self.renormalize = renormalize
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3)).astype('int')  # bbox不同类别对应的颜色

        # CAM对象构造
        self.method = method
        self.target = Yolov8Target(backward_type, conf_threshold, ratio)
        self.target_layers = [self.model.model[l] for l in layer]

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=self.iou_thres, max_det=self.max_det)[0]  # 压缩batch_size维度
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)  # 空cam图
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())  # 截取bbox范围内的热力图 归一化 重构
        renormalized_cam = scale_cam_image(renormalized_cam)  # todo 考虑是否移除 for multi object整体归一化 
        cam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)
        return cam_image_renormalized

    def process(self, img_path, save_path):

        # 图像读取
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orin, w_orin = img.shape[:2]

        # 原始图片备份
        img_orin= img.copy()
        img_orin = np.float32(img_orin) / 255.0

        # 图像预处理
        img = letterbox(img, new_shape=(self.imgsz, self.imgsz))[0]  # 取图像(第一个参数)
        h_letter, w_letter = img.shape[:2]
        img = np.float32(img) / 255.0

        # Tensor构造
        x = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 计算CAM
        grayscale_cam = self.grad_cam(x, [self.target])  # (1, 736, 1280)
        
        # 取唯一一个样本
        grayscale_cam = grayscale_cam[0, :]  # (736, 1280)
        grayscale_cam = cv2.resize(grayscale_cam, (w_orin, h_orin))  # (2160, 3840) 恢复热力图尺度至原图尺度

        # 原图上绘制CAM热力图
        cam_image = show_cam_on_image(img_orin, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)  # (2160, 3840, 3)

        # 模型推理
        pred = self.model(x)[0]

        # 后处理 
        pred = self.post_process(pred)  # nms: (num_boxes, 6 + num_masks) (x1, y1, x2, y2, confidence, class, mask1, mask2, ...)

        # 恢复bbox尺度至原图尺度(原地操作)
        bbox = pred[:, :4] 
        bbox[:, [0, 2]] /= w_letter
        bbox[:, [1, 3]] /= h_letter
        bbox[:, [0, 2]] *= w_orin
        bbox[:, [1, 3]] *= h_orin
        
        # 归一化各bbox内部的热力图
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(bbox.cpu().detach().numpy().astype('int'), img_orin, grayscale_cam)
        
        # 绘制检测结果
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                bbox = data[:4]
                score = float(data[4])
                cls_id = int(data[5])
                cam_image = self.draw_detections(
                    box=bbox, 
                    color=self.colors[cls_id], 
                    name=f'{self.class_names[cls_id]} {score:.2f}',
                    img=cam_image)
        
        # 热力图持久化
        if self.save_cam:
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, cam_image)

        # 返回图像
        return np.uint8(img_orin*255), cam_image

    def __call__(self, img_path, save_path):
        
        all_heatmaps = {}

        # 目录和文件处理
        if os.path.isdir(img_path):
            all_images = os.listdir(img_path)  # 全部文件名列表
        else:
            img_filename = os.path.basename(img_path)  # 文件名
            img_path = os.path.dirname(img_path)  # 文件路径
            all_images = [img_filename]  # 全部文件名列表

        for img_filename in tqdm(all_images, ncols=100):

            # 文件名处理
            img_filename_without_extension, img_extension = os.path.splitext(img_filename)
            first = True

            # 初始化字典
            all_heatmaps[img_filename] = []

            for method in self.method:

                # 构造CAM对象
                self.grad_cam = eval(method)(self.model, self.target_layers)
                self.grad_cam.activations_and_grads = Yolov8ActivationsAndGradients(self.model, self.target_layers, None)
                orin_image, cam_image = self.process(f'{img_path}/{img_filename}', f'{save_path}/{img_filename_without_extension}_{method}.{img_extension}')
                
                # 存储原图
                if first:
                    all_heatmaps[img_filename].append(orin_image)
                    first = False

                # 追加CAM图
                all_heatmaps[img_filename].append(cam_image)

        return all_heatmaps


if __name__ == '__main__':

    all_methods = [
        'GradCAM', 'HiResCAM', 'GradCAMElementWise', 'XGradCAM', 
        'GradCAMPlusPlus', 'LayerCAM', 'EigenCAM', 'EigenGradCAM', 'RandomCAM'
    ]

    all_methods = ['GradCAM']
   
    params = {
        'weight': 'runs/ultralytics/finetune/yolov8n-p2_lsk_bifpn_DIoU_aug10_1280/triple_split/train/weights/best.pt',
        'device': 'cuda:0',
        'method': all_methods,  # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': [10, 13, 16, 19, 22, 25, 28],
        'backward_type': 'all',  # class, box, all
        'conf_threshold': 0.5,  # 0.2+
        'iou_thres': 0.6,
        'max_det': 100,
        'ratio': 0.01,  # 0.02-0.1
        'save_cam': True,
        'show_box': True,
        'renormalize': True
    }

    heatmap = Yolov8Heatmap(**params)
    all_heapmaps = heatmap('results/images/heatmaps/test_imgs', 'results/images/heatmaps')

    # todo: https://github.com/rigvedrs/YOLO-V8-CAM/tree/main
