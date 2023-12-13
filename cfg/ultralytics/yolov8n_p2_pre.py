import os
import sys
from ultralytics import YOLO
sys.path.append('/home/yeli/workspace/lard/lard-detection/')
from src.tools.pushplus import send_info  # noqa

ROOT_PROJECT = os.environ.get('LARD_PROJECT_ROOT_PATH')

# 变量定义
model_name = 'yolov8n'
model_stru = '-p2'
model_cfg = '_pre'

# 路径构建
exp_name = f'{model_name}{model_stru}{model_cfg}'
path_yaml = f'{ROOT_PROJECT}/cfg/ultralytics/models/{model_name}{model_stru}.yaml'
path_weights = f'{ROOT_PROJECT}/runs/ultralytics/pretrain/yolov8n-p2/train4/weights/best.pt'
path_trans = f'{ROOT_PROJECT}/datasets/cfg/lard_transform.json'
path_data = f'{ROOT_PROJECT}/cfg/ultralytics/datasets/lard_val_test_synth.yaml'

print(f"实验名称: {exp_name}")

# 超参数定义
num_gpu = 10
num_workers_per_gpu = 8
num_epochs = 500
batch_size_per_gpu = 16
patience = num_epochs
batch_size = batch_size_per_gpu * num_gpu

# 实例化YOLO模型
model = YOLO(path_yaml, task='detect').load(weights=path_weights)

# 训练模型
results = model.train(
    data=path_data,
    epochs=num_epochs,
    patience=patience,
    batch=batch_size,
    imgsz=640,
    cache=True,
    device=list(range(num_gpu)),
    workers=num_workers_per_gpu,
    project='runs/ultralytics',
    name=f'{exp_name}/train',
    exist_ok=False,
    optimizer='SGD',
    seed=0,
    deterministic=True,
    close_mosaic=20,
    lr0=batch_size*0.01/(16*8), # 16=2*8
    warmup_epochs=3,
    album=0.0,
    path_transform=path_trans,
)

send_info(exp_name)

# bash ./scripts/ultralytics_train.sh yolov8n_p2_pre
