import os
import json
import pandas as pd
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion


coco_metrics_name = [
    "AP (IoU=0.50:0.95, area=all, maxDets=100)",
    "AP (IoU=0.50, area=all, maxDets=100)",
    "AP (IoU=0.75, area=all, maxDets=100)",
    "AP (IoU=0.50:0.95, area=small, maxDets=100)",
    "AP (IoU=0.50:0.95, area=medium, maxDets=100)",
    "AP (IoU=0.50:0.95, area=large, maxDets=100)",
    "AR (IoU=0.50:0.95, area=all, maxDets=1)",
    "AR (IoU=0.50:0.95, area=all, maxDets=10)",
    "AR (IoU=0.50:0.95, area=all, maxDets=100)",
    "AR (IoU=0.50:0.95, area=small, maxDets=100)",
    "AR (IoU=0.50:0.95, area=medium, maxDets=100)",
    "AR (IoU=0.50:0.95, area=large, maxDets=100)"
]


def get_coco_imgname_2_imgid(annotation_file: str) -> dict:

    # 获取一个字典 name_2_id {图片名(无拓展名): 图片id}
    coco = COCO(annotation_file=annotation_file)
    img_ids = coco.getImgIds()

    imgname_2_imgid = {}
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        filename = os.path.splitext(file_name)[0]  # 无拓展名的文件名
        imgname_2_imgid[filename] = img_id

    return imgname_2_imgid


def get_coco_imgid_2_imgname(annotation_file: str) -> dict:

    # 获取一个字典 id_2_name {图片id: 图片名(无拓展名)}
    coco = COCO(annotation_file=annotation_file)
    img_ids = coco.getImgIds()

    imgid_2_imgname = {}
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        filename = os.path.splitext(file_name)[0]  # 无拓展名的文件名
        imgid_2_imgname[img_id] = filename

    return imgid_2_imgname


def get_coco_imgid_2_imgsize(annotation_file: str) -> dict:

    # 获取一个字典 imgid_2_imgsize {图片id: 图片wh}
    coco = COCO(annotation_file=annotation_file)
    img_ids = coco.getImgIds()

    imgid_2_imgsize = {}
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        w = img_info['width']
        h = img_info['height']
        img_id = img_info['id']
        imgid_2_imgsize[img_id] = (w, h)
    return imgid_2_imgsize


def cal_coco_metrics(annotation_file: str, prediction_file: str):

    coco_true = COCO(annotation_file=annotation_file)
    coco_pred = coco_true.loadRes(resFile=prediction_file)

    coco_evaluator = COCOeval(
        cocoGt=coco_true,
        cocoDt=coco_pred,
        iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def ensembling_boxes(method, boxes_list, scores_list, labels_list, weights=None, iou_thr=0.5, sigma=0.1, skip_box_thr=0.0001):

    # 目标检测模型bbox集成
    if method == 'nms':
        boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    elif method == 'soft':
        boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    elif method == 'nmw':
        boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    elif method == 'wbf':
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    else:
        print('Invalid method')

    return boxes, scores, labels


def ensemble_coco_metrics(path_preds_json_list, path_instance_json, verbose=True, path_merge_json='prediction_merge.json', ensemble_method='nms'):
    """
    :param path_preds_json_list: 子模型预测json文件列表
    :param path_instance_json: 标签json文件
    :param path_merge_json: 融合预测json文件
    :param ensemble_method: bbox融合算法
    :return: all_metrics: 子模型和集成模型精度指标
    """

    imgid_2_imgsize = get_coco_imgid_2_imgsize(path_instance_json)

    # 获取有预测结果的全部图片id列表
    img_id_with_preds = []

    # 遍历每个模型的预测json文件
    for preds_json in path_preds_json_list:
        with open(preds_json, 'r') as file:
            data = json.load(file)
            img_ids = [pred['image_id'] for pred in data]
            img_id_with_preds.extend(img_ids)
            print(f"模型预测文件{preds_json}包含检测结果{len(data)}个")

    # 去除重复
    img_id_with_preds = list(set(img_id_with_preds))

    print(f"以下id的图片有预测结果: {img_id_with_preds}")
    print(f"共计: {len(img_id_with_preds)}")

    # 融合预测结果json文件内容
    merge_json = []

    # 遍历每一个有检测结果的图像
    for img_id in img_id_with_preds:

        # 获得每个模型对图像img_id给出的检测结果
        # preds_list [
        #   [[], [], []], # 第一个模型的3个预测
        #   [[], []],     # 第二个模型的2个预测
        #   [[]],         # 第三个模型的1个预测
        # ]
        preds_list = []  # [preds_model_1, preds_model_2, ...]
        for file in path_preds_json_list:
            with open(file, 'r') as file:
                data = json.load(file)
                preds_of_current_model = [pred for pred in data if pred['image_id'] == img_id]  # 当前模型对该图片的预测结果
                if preds_of_current_model != []:  # 如果该模型对该图像有预测，则追加所有预测结果到preds_list
                    preds_list.append(preds_of_current_model)  # 第i个模型的多个预测

        # 图像img_id仅一个模型给出预测 无法融合 直接保留原始预测结果 extend到新json
        if len(preds_list) == 1:
            if verbose:
                print(f"图片{img_id}: {len(preds_list)}个模型给出检测结果, 不做融合直接保留")
            merge_json.extend(preds_list[0])

        # 至少两个模型对图像img_id给出预测 融合后 extend到新json
        elif len(preds_list) > 1:
            if verbose:
                print(f"图片{img_id}: {len(preds_list)}个模型给出检测结果", [f"模型{i}给出{len(pred)}" for i, pred in enumerate(preds_list)], end=' ') 

            # 存储各个模型对图像img_id的预测 [box, score, label] 每个列表的长度同preds_list
            boxes_of_each_model = []
            scores_of_each_model = []
            labels_of_each_model = []

            # 遍历各模型对当前图片的预测结果
            for i, preds_of_current_model in enumerate(preds_list):

                # 存储重构的各检测结果
                boxes_of_current_model = []
                scores_of_current_model = []
                labels_of_current_model = []

                # 遍历当前模型的各个预测
                for pred in preds_of_current_model:
                    
                    # 读取注释数据
                    image_id = pred['image_id']
                    bbox = pred['bbox']
                    score = pred['score']
                    label = pred['category_id']

                    # 1. 读取image_id对应文件获取图像尺寸
                    w, h = imgid_2_imgsize[image_id]

                    # 2. 转换 bbox 为 [x_min, y_min, x_max, y_max] 格式
                    x_min, y_min, width, height = bbox
                    x_max = x_min + width
                    y_max = y_min + height

                    # 3. 归一化
                    converted_bbox = [min(x_min/w, 1), min(y_min/h, 1), min(x_max/w, 1), min(y_max/h, 1)]
                    
                    # 追加重构后的该模型给出的检测
                    boxes_of_current_model.append(converted_bbox)
                    scores_of_current_model.append(score)
                    labels_of_current_model.append(label)

                # 追加重构后的各模型给出的检测
                boxes_of_each_model.append(boxes_of_current_model)
                scores_of_each_model.append(scores_of_current_model)
                labels_of_each_model.append(labels_of_current_model)
            
            # 超参设置
            weights = [1 for _ in range(len(path_preds_json_list))]        
            iou_thr = 0.5
            skip_box_thr = 0.0001
            sigma = 0.1

            # 融合各模型的检测结果
            try:
                boxes, scores, labels = ensembling_boxes(method=ensemble_method, boxes_list=boxes_of_each_model, scores_list=scores_of_each_model, labels_list=labels_of_each_model, weights=weights, iou_thr=iou_thr, sigma=sigma, skip_box_thr=skip_box_thr)
                if verbose:
                    print(f"融合后检测结果数量: {boxes.shape[0]}")
            except Exception as e:
                print(f"融合失败: {e}")
                continue

            # 重构为dict已追加至merger_json
            for i in range(boxes.shape[0]):
                x_min, y_min, x_max, y_max = boxes[i]
                w, h = imgid_2_imgsize[img_id]
                new_box = [x_min*w, y_min*h, (x_max-x_min)*w, (y_max-y_min)*h]
                pred = {
                    "image_id": int(img_id),
                    "category_id": int(labels[i]),
                    "bbox": new_box,
                    "score": float(scores[i])
                }
                merge_json.append(pred)
                
    # 持久化merge_json为文件
    print(f"融合后累计预测数量: {len(merge_json)}")
    with open(path_merge_json, 'w') as f:
        json.dump(merge_json, f, indent=4)

    # 各模型精度评价
    all_metrics = {}

    # 子模型
    for i, pred_json in enumerate(path_preds_json_list):
        cocoEval = cal_coco_metrics(path_instance_json, pred_json)
        all_metrics[f"model_{i}"] = cocoEval.stats

    # 融合模型
    cocoEval = cal_coco_metrics(path_instance_json, path_merge_json)
    all_metrics['model_ensemble'] = cocoEval.stats


    # 重构数据
    all_metrics = pd.DataFrame(all_metrics, index=coco_metrics_name)

    return all_metrics
