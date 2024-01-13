from ultralytics import YOLO

model = YOLO('cfg/ultralytics/models/yolov8n-p2/yolov8n-p2_cbam_afpn.yaml')
results = model.train(
    data='coco128.yaml',
    epochs=1,
    project="test",
    plots=False)