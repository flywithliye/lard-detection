import os
from ultralytics import YOLO

ROOT_PROJECT = os.environ.get('LARD_PROJECT_ROOT_PATH')

# 变量定义
model_name = 'yolov8n'
model_stru = ''  
model_cfg = '_aug'  # _aug

assert model_name in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
assert model_stru in ['', '-p2', '-p6']

# 路径构建
exp_name = f'{model_name}{model_stru}{model_cfg}'
path_yaml = f'{model_name}{model_stru}.yaml'
path_weights = f'{ROOT_PROJECT}/weights/{model_name}.pt'
path_trans = f'{ROOT_PROJECT}/datasets/cfg/lard_transform.json'
path_data = f'{ROOT_PROJECT}/datasets/cfg/lard_val_test_synth.yaml'

print(f"实验名称: {exp_name}")

# 超参数定义
num_gpu = 8
device = [i for i in range(num_gpu)]

num_workers = 48
num_epochs = 1000
batch_size = num_gpu * 16

# 实例化YOLO模型
model = YOLO(path_yaml, task='detect').load(weights=path_weights)

# 训练模型
results = model.train(
    data=path_data,
    epochs=num_epochs,
    patience=50,  # 无显著改善停止训练
    batch=batch_size,
    imgsz=640,
    cache=True,
    device=device,
    workers=num_workers,
    project='runs/ultralytics',
    name=f'{exp_name}/train',
    exist_ok=True,
    optimizer='auto',
    close_mosaic=10,
    warmup_epochs=3,
    custom_aug=True,  # 是否启用自定义数据增强
    path_transform=path_trans,  # 自定义数据增强配置文件路径
)
