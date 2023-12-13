from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
results = model.train(
    data='coco128.yaml',
    epochs=1,
    soft_nms=True,
    project="test",
    plots=False)