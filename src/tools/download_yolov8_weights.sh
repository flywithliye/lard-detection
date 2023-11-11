#!/bin/bash

# 预训练权重下载

# 使用方式：./download_yolov8_weights.sh

wget -P ./../../weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget -P ./../../weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget -P ./../../weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget -P ./../../weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
wget -P ./../../weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt