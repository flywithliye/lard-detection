#!/bin/bash

# 使用方式：./ultra_train.sh <model_name>

# 确保脚本参数数量正确
if [ "$#" -ne 1 ]; then
    echo "使用方式: $0 <model_name>"
    exit 1
fi

model_name=$1

# 使用传入的参数启动训练进程
setsid python cfg/ultra/${model_name}.py > logs/ultralytics_train_${model_name}.log 2>&1 &

# ./scripts/ultralytics_train.sh yolov8n 10