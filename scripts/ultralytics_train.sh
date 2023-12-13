#!/bin/bash

# 使用方式：./ultralytics_train.sh <model_name>
echo_rb() { echo -e "\e[1;31m$1\e[0m"; }

# 确保脚本参数数量正确
if [ "$#" -ne 1 ]; then
    echo_rb "使用方式: $0 <model_name>"
    exit 1
fi

model_name=$1

# 使用传入的参数启动训练进程
setsid python cfg/ultralytics/${model_name}.py > logs/ultralytics_train_${model_name}.log 2>&1 &

# ./scripts/ultralytics_train.sh yolov8n
# setsid yolo train resume model=runs/ultralytics/pretrain/yolov8n-p2/train4/weights/last.pt > logs/ultralytics_train_yolov8n_p2_pretrain_resume.log 2>&