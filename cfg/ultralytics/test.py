import os
import sys
import argparse
import json
import glob
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
from ultralytics import YOLO
sys.path.append('/home/yeli/workspace/lard/lard-detection/')
from src.utils.metric import get_coco_imgname_2_imgid  # noqa
from src.utils.metric import cal_coco_metrics  # noqa
from src.utils.metric import coco_metrics_name  # noqa


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='', help="测试实验模式")
    parser.add_argument("--merge_mode", type=str, default='', help="模块集成模式")

    parser.add_argument("--model", type=str, default='yolov8n', help="模型名称")
    parser.add_argument("--stru", type=str, default='', choices=['p2'], help="模型结构")
    parser.add_argument("--cfg", type=str, default='', help="模型配置")

    parser.add_argument("--size", type=int, default=640, help="训练图像尺寸")
    parser.add_argument("--test_size", type=int, default=640, help="测试图像尺寸")
    parser.add_argument("--album", type=float, default=0.0, help="albumentation增强概率")
    parser.add_argument("--aug_json", type=str, default='', help="albumentation增强配置文件")
    parser.add_argument("--iou_type", type=str, default='CIoU', choices=['GIoU', 'DIoU', 'CIoU', 'SIoU', 'EIoU', 'WIoU', 'MDPIoU1', 'MDPIoU2', 'ShapeIoU', 'NWD'], help="IOU类型")

    parser.add_argument("--batch_size", type=int, default=64, help="测试批次大小")
    parser.add_argument("--nms_conf", type=float, default=0.001, help="NMS-CONF阈值, 过滤掉置信度过低的bbox, 越大越严格")
    parser.add_argument("--nms_iou", type=float, default=0.6, help="NMS-IOU阈值, 过滤掉重合度过高的bbox, 越小越严格")
    parser.add_argument("--max_det", type=int, default=8, help="最大检测数量")
    parser.add_argument("--soft_nms", action='store_true', help="使用soft nms")

    parser.add_argument("--pretrain", action="store_true", help="是否在coco进行预训练")

    print(f'测试实验参数: {parser.parse_args()}')
    return parser.parse_args()


def get_complex(path_weight, imgsz=640):
    model = YOLO(path_weight, task='detect')
    n_l, n_p, n_g, flops = model.model.info(imgsz=imgsz)
    return n_l, n_p, n_g, flops


def test(args):

    # 环境变量
    ROOT_PROJECT = os.environ.get('LARD_PROJECT_ROOT_PATH')

    # 实验设置
    model = args.model
    stru = f'-{args.stru}' if args.stru else ''
    cfg = f'_{args.cfg}' if args.cfg else ''    
    iou_type = f'_{args.iou_type}' if args.iou_type != 'CIoU' else ''
    album = f'_aug_{args.aug_json}_{int(args.album*100)}' if args.album != 0.0 else ''
    size = f'_{args.size}'
    mode = args.mode
    merge_mode = args.merge_mode

    test_speed = True  # ! 是否计算测试时间
    test_complex = True  # ! 是否计算参数量
    BEST_OR_LAST = 'best'  # ! todo 同步修改本变量

    # 实验路径构建
    exp_name = f'{model}{stru}{cfg}{iou_type}{album}{size}'  # yolov8n-p2_cbam_siou_aug2_1280
    project = 'runs/ultralytics'
    print(f'实验名称: {exp_name}')

    # 预训练时使用coco数据集
    if args.pretrain:
        path_data = f'{ROOT_PROJECT}/cfg/ultralytics/datasets/coco.yaml'
        project = project + '/pretrain'

    # 测试模型权重
    trained_model_path = f'{project}/{mode}/{exp_name}/train/weights/{BEST_OR_LAST}.pt'

    # 模型测试超参数
    batch_size = args.batch_size
    nms_conf = args.nms_conf
    nms_iou = args.nms_iou
    max_det = args.max_det
    soft_nms = args.soft_nms

    # 数据集列表
    all_datasets = ['test_synth', 'test_real_nominal', 'test_real_edge', 'test_real', 'test']

    # 1.测试
    for data_type in all_datasets:

        print(f'正在测试: {data_type}')
        path_data = f'cfg/ultralytics/datasets/lard_val_{data_type}.yaml'

        # 实例化YOLO模型
        model = YOLO(trained_model_path, task='detect')

        # 执行测试
        model.val(
            data=path_data,
            imgsz=args.test_size,
            batch=batch_size,
            save_json=True,  # 保存预测结果JSON
            conf=nms_conf,  # 检测的目标置信度阈值
            iou=nms_iou,  # NMS使用的IOU阈值
            max_det=max_det,  # 最大检测数量
            device=0,  # 设备id
            split='test',  # val时使用的数据集划分
            project=project,
            name=f'{mode}/{exp_name}/test/{data_type}',
            exist_ok=True,  # 允许覆盖
            # 自定义参数
            soft_nms=soft_nms,  # 使用softnms
        )

        # 释放缓存
        torch.cuda.empty_cache()

    # 2.评价
    all_metrics = {}
    for data_type in all_datasets:

        print(f'正在评价: {data_type}')
        path_annotation = f'datasets/lard/annotations/instances_{data_type}.json'
        path_prediction = f'{project}/{mode}/{exp_name}/test/{data_type}/predictions.json'
        path_prediction_modified = f'{project}/{mode}/{exp_name}/test/{data_type}/predictions_modified.json'

        # 读取原始JSON文件
        with open(path_prediction, 'r') as f:
            pred = json.load(f)

        # JSON文件重构
        imgname_2_imgid = get_coco_imgname_2_imgid(path_annotation)
        for item in pred:
            item['image_id'] = imgname_2_imgid[item['image_id']]

        # 保存修改后的JSON文件
        with open(path_prediction_modified, 'w') as f:
            json.dump(pred, f, indent=4)

        # 指标计算
        cocoEval = cal_coco_metrics(path_annotation, path_prediction_modified)
        all_metrics[data_type] = cocoEval.stats

    # 构建指标dataframe
    all_metrics = pd.DataFrame(all_metrics, index=coco_metrics_name)

    # 3.模型复杂度
    if test_complex:
        n_l, n_p, n_g, flops = get_complex(trained_model_path, imgsz=args.test_size)
        n_p = n_p * 1e-6
        all_complex = {}
        for data_type in ['test_synth', 'test_real_nominal', 'test_real_edge', 'test_real', 'test']:
            all_complex[data_type] = [n_p, flops]  # 仅为便于数据拼接

        # 构建速度dataframe
        all_complex = pd.DataFrame(
            data=all_complex,
            index=['Param', 'FLOPs']
        )

        # 合并复杂度指标
        all_metrics = pd.concat([all_metrics, all_complex], axis=0)

    # 4.测速
    if test_speed:
        all_speed = {}
        for data_type in all_datasets:

            # 测试集全部图像列表
            path = f'datasets/lard/detection/{data_type}/images/'
            image_paths = glob.glob(f'{path}*')
            infer_time = []

            # 实例化YOLO模型
            model = YOLO(trained_model_path, task='detect')

            # 分别推理
            for image_path in tqdm(image_paths, ncols=100, miniters=100, desc='推理'):
                result = model.predict(
                    source=image_path,
                    imgsz=args.test_size,
                    conf=nms_conf,
                    iou=nms_iou,
                    max_det=max_det,
                    device=0,
                    verbose=False
                )
                # 累加当前推理时间：前处理, 前向传播, 后处理
                infer_time.append(sum(list(result[0].speed.values())))

            # 释放缓存
            torch.cuda.empty_cache()

            # 推理时间和帧率
            average_time = sum(infer_time)/len(infer_time)
            fps = 1000/average_time
            all_speed[data_type] = [average_time, fps]

            # 输出信息
            print(f'数据集: {data_type}')
            print(f'平均推理时间: {average_time:.2f} ms')
            print(f'平均FPS: {fps:.2f}')

        # 构建速度dataframe
        all_speed = pd.DataFrame(
            data=all_speed,
            index=['Time', 'FPS']
        )

        # 合并测试指标
        all_metrics = pd.concat([all_metrics, all_speed], axis=0)

    # 保存评价指标
    path_results = f'results/tables/metrics/metrics_{mode}_{exp_name}_{args.test_size}.csv'
    if mode == 'merge':
        path_results = f'results/tables/metrics/metrics_{mode}_{merge_mode}_{exp_name}_{args.test_size}.csv'
    all_metrics.index.name = 'metrics'
    all_metrics.to_csv(path_results, float_format='%.3f')

    print(f'测试指标已保存至: {path_results}')
    print(all_metrics.round(3))


def main():

    # 实验开始
    start_time = datetime.now()
    print(f'实验开始时间: {start_time}')

    # 开始实验
    args = parse_arguments()
    test(args)

    # 实验结束
    end_time = datetime.now()
    print(f'实验开始时间: {start_time}')
    print(f'实验结束时间: {end_time}')

    # 计算耗时
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours += duration.days * 24
    print(f'实验耗时: {hours}小时 {minutes}分钟 {seconds}秒')


if __name__ == '__main__':
    main()
