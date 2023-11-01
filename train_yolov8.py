from ultralytics import YOLO

# 变量定义
model_name = 'yolov8n'
model_stru = '-p2'
model_cfg = '_train_val'
assert model_name in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
assert model_cfg in ['-p2', '-p6']

# 路径构建
exp_name = f'{model_name}{model_stru}{model_cfg}_TEST'
path_yaml = f'{model_name}{model_stru}.yaml'
path_weights = f'weights/{model_name}.pt'

print(f"实验名称: {exp_name}")

# 实例化YOLO模型
model = YOLO(path_yaml, task='detect').load(weights=path_weights)

# 训练模型
results = model.train(
    data='datasets/cfg/lard_val_test_synth.yaml',
    epochs=20,
    batch=32,
    imgsz=640,
    cache=False,
    device=[0],
    workers=12,
    name=f"{exp_name}/train",
    use_custom_aug=False, # 是否启用自定义数据增强
)
