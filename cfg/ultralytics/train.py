import os
import sys
import argparse
from datetime import datetime
from ultralytics import YOLO
sys.path.append('/home/yeli/workspace/lard/lard-detection/')
from src.tools.pushplus import send_info  # noqa


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default='', help="training model, 训练实验模式")
    parser.add_argument("--finetune_mode", type=str, default='', help="finetune mode, Finetune模式")

    parser.add_argument("--model", type=str, default='yolov8n', help="model name, 模型名称")
    parser.add_argument("--stru", type=str, default='', choices=['', 'p2'], help="model structure, 模型结构")
    parser.add_argument("--cfg", type=str, default='', help="model configs, 模型配置")

    parser.add_argument("--dataset", type=str, default='', help="dataset file, 是否指定数据集文件")
    parser.add_argument("--weights", type=str, default='', help="pretrained weights, 是否指定预训练权重")
    parser.add_argument("--num_epochs", type=int, default=300, help="epochs, 训练次数")
    parser.add_argument("--size", type=int, default=640, help="img size, 图像尺寸")
    parser.add_argument("--album", type=float, default=0.0, help="albumentation prob, albumentation增强概率")
    parser.add_argument("--aug_json", type=str, default='', help="albumentation config file, albumentation增强配置文件")
    parser.add_argument("--iou_type", type=str, default='CIoU', choices=['GIoU', 'DIoU', 'CIoU', 'SIoU', 'EIoU', 'WIoU', 'MDPIoU1', 'MDPIoU2', 'ShapeIoU', 'NWD'], help="IOU type, IOU类型")

    parser.add_argument("--pretrain", action="store_true", help="if using coco for pretrain, 是否在coco进行预训练")

    print(f'Training params, 训练实验参数: {parser.parse_args()}')
    return parser.parse_args()


def train(args):
    
    # env 环境变量
    ROOT_PROJECT = os.environ.get('LARD_PROJECT_ROOT_PATH')
    
    # exp settings 实验设置
    model = args.model
    stru = f'-{args.stru}' if args.stru else ''
    cfg = f'_{args.cfg}' if args.cfg else ''    
    iou_type = f'_{args.iou_type}' if args.iou_type != 'CIoU' else ''
    album = f'_aug_{args.aug_json}_{int(args.album*100)}' if args.album != 0.0 else ''
    size = f'_{args.size}'
    mode = args.mode
 
    # finetune mode/ finetuen dataset, finetune模式/finetune使用的数据集
    finetune_mode = args.finetune_mode

    # set exp path 实验路径构建
    exp_name = f'{model}{stru}{cfg}{iou_type}{album}{size}'  # yolov8n-p2_cbam_siou_aug20_1280
    project = 'runs/ultralytics/'
    name = f'{mode}/{exp_name}/train' if not finetune_mode else f'{mode}/{exp_name}/{finetune_mode}/train/'
    print(f'Exp name 实验名称: {exp_name}')

    # Path to config files 训练配置文件路径
    path_yaml = f'{ROOT_PROJECT}/cfg/ultralytics/models/{model}{stru}/{model}{stru}{cfg}.yaml'
    path_trans = f'{ROOT_PROJECT}/cfg/ultralytics/datasets/aug/lard_transform_{args.aug_json}.json'
    if not finetune_mode:
        path_data = f'{ROOT_PROJECT}/cfg/ultralytics/datasets/lard_val_test_synth.yaml' 
    elif finetune_mode in ['single', 'double', 'triple', 'triple_split']:
        path_data = f'{ROOT_PROJECT}/cfg/ultralytics/datasets/finetune/{finetune_mode}.yaml'
    else:
        path_data = f'{ROOT_PROJECT}/cfg/ultralytics/datasets/{finetune_mode}.yaml'
    path_weights = f'{ROOT_PROJECT}/weights/{model}{stru}.pt' if not args.weights else f'{ROOT_PROJECT}/{args.weights}'

    # pretrain use coco 预训练时使用coco数据集
    # todo we haven't use this code for a while, plz check.
    if args.pretrain:
        path_data = f'{ROOT_PROJECT}/cfg/ultralytics/datasets/coco.yaml'
        project = project + '/pretrain'

    # training params 模型训练超参数
    num_gpu = 10
    num_workers_per_gpu = 8
    num_epochs = args.num_epochs
    batch_size_per_gpu = 16
    patience = num_epochs
    batch_size = batch_size_per_gpu * num_gpu

    # get yolo model 实例化YOLO模型
    if mode != 'finetune':
        model = YOLO(path_yaml, task='detect')
    elif path_weights:
        model = YOLO(path_yaml, task='detect').load(weights=path_weights)
    else:
        print('Please provide weights path in finetune mode!')
        return

    # start training 训练模型
    try:
        model.train(
            data=path_data,
            epochs=num_epochs,
            patience=patience,
            batch=batch_size,
            imgsz=args.size,
            cache=True,
            device=list(range(num_gpu)),
            workers=num_workers_per_gpu,
            project=project,
            name=name,
            exist_ok=False,
            optimizer='SGD',
            seed=0,
            deterministic=True,
            close_mosaic=20,
            lr0=batch_size*0.01/(16*8),
            warmup_epochs=3,
            # self defined params 自定义参数
            album=args.album,
            path_transform=path_trans,
            iou_type=args.iou_type,
        )
        send_info(title=f'Train complete: {exp_name}', content=f'Train complete: {exp_name}')
    except Exception as e:
        print(f'Train Error: {e}')
        send_info(title=f'Train Error{exp_name}', content=e)
    

def main():

    # start
    start_time = datetime.now()
    print(f'Start time: {start_time}')

    # start training
    args = parse_arguments()
    train(args)

    # end
    end_time = datetime.now()
    print(f'End time: {start_time}')
    print(f'Time used: {end_time}')

    # get duration
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours += duration.days * 24
    print(f'Exp time used: {hours} h {minutes} m {seconds} s')


if __name__ == '__main__':
    main()
