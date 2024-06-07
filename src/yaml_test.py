from ultralytics import YOLO
import shutil

path_yaml = 'cfg/ultralytics/models/yolov8n-p2/yolov8n-p2_dysample.yaml'
model = YOLO(path_yaml)
results = model.train(
    data='cfg/ultralytics/datasets/coco128.yaml',
    epochs=1,
    project="test",
    plots=False)

shutil.rmtree("test")
