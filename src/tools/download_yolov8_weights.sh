#!/bin/bash

# 预训练权重下载

# 使用方式：./src/tools/download_yolov8_weights.sh

# orin
wget -P ./weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget -P ./weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget -P ./weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget -P ./weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
wget -P ./weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# v8.1
# https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
# https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
# https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt
# https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt
# https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt