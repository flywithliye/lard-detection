#!/bin/bash

# Usage: ./script/mmyolo_train.sh <model_name> <num_gpu> <img_size>

echo_rb() { echo -e "\e[1;31m$1\e[0m"; }

model_list=("yolov5n" "yolov6n" "yolov7t" "yolov8n")

# param check
if [ "$#" -ne 3 ]; then
    echo_rb "使用方式: $0 <model_name> <num_gpu> <img_size>，其中 <model_name> 必须是以下之一: ${model_list[*]}"
    exit 1
fi

# define params
model_name=$1
num_gpu=$2
img_size=$3

# verify model_name
if [[ ! " ${model_list[*]} " =~ " ${model_name} " ]]; then
    echo_rb "Error: model_name must be one o: ${model_list[*]}"
    exit 1
fi

# verify num_gpu (1 to 10)
if [[ $num_gpu -lt 1 || $num_gpu -gt 10 ]]; then
    echo_rb "Error: num_gpu must be between 1 and 10"
    exit 1
fi

# verify img_size (640 or 1280)
if [[ "$img_size" != "640" && "$img_size" != "1280" ]]; then
    echo "Error: img_size must be 640 or 1280"
    exit 1
fi

# set training cmd according to num_gpu
if [[ $num_gpu > 1 ]]; then
    # multi gpus
    echo_rb "Model: ${model_name}, multi GPUs training: ${num_gpu}"
    setsid bash ./3rdparty/mmyolo/tools/dist_train.sh cfg/mmyolo/${model_name}_${img_size}.py ${num_gpu} > logs/train/mmyolo_train_${model_name}_${img_size}.log 2>&1 &
else
    # single gpu
    echo_rb "Model: ${model_name}, singe GPU training"
    setsid python 3rdparty/mmyolo/tools/train.py cfg/mmyolo/${model_name}_${img_size}.py > logs/train/mmyolo_train_${model_name}_${img_size}.log 2>&1 &
fi

# ./scripts/mmyolo_train.sh yolov5n 10 640
# ./scripts/mmyolo_train.sh yolov6n 10 640
# ./scripts/mmyolo_train.sh yolov7t 10 640
# ./scripts/mmyolo_train.sh yolov8n 10 640

# ./scripts/mmyolo_train.sh yolov5n 10 1280
# ./scripts/mmyolo_train.sh yolov6n 10 1280
# ./scripts/mmyolo_train.sh yolov7t 10 1280
# ./scripts/mmyolo_train.sh yolov8n 10 1280