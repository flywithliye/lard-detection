#!/bin/bash

echo_rb() { echo -e "\e[1;31m$1\e[0m"; }
model_list=("faster_rcnn" "ssd" "yolov3" "retinanet" "centernet")

# param check
if [ "$#" -ne 3 ]; then
    echo_rb "Usage: $0 <model_name> <num_gpu> <img_size>, <model_name> must be one of: ${model_list[*]}"
    exit 1
fi

# define params
model_name=$1
num_gpu=$2
img_size=$3

# verify model_name
if [[ ! " ${model_list[*]} " =~ " ${model_name} " ]]; then
    echo_rb "Error: model_name must be one of: ${model_list[*]}"
    exit 1
fi

# verify num_gpu (1 to 10)
if [[ $num_gpu -lt 1 || $num_gpu -gt 10 ]]; then
    echo_rb "Error: num_gpu must be between 1 and 10"
    exit 1
fi

# set training cmd according to num_gpu
if [[ $num_gpu > 1 ]]; then
    # multi gpus
    echo_rb "Model: ${model_name}, multi GPUs training: ${num_gpu}"
    setsid bash ./3rdparty/mmdetection/tools/dist_train.sh cfg/mmdet/${model_name}_${img_size}.py ${num_gpu} > logs/train/mmdet_train_${model_name}_${img_size}.log 2>&1 &
else
    # single gpu
    echo_rb "Model: ${model_name}, singe GPU training"
    setsid python 3rdparty/mmdetection/tools/train.py cfg/mmdet/${model_name}_${img_size}.py > logs/train/mmdet_train_${model_name}_${img_size}.log 2>&1 &
fi

# ./scripts/mmdet_train.sh faster_rcnn 10 1333
# ./scripts/mmdet_train.sh ssd 10 512
# ./scripts/mmdet_train.sh yolov3 10 608
# ./scripts/mmdet_train.sh retinanet 10 1333
# ./scripts/mmdet_train.sh centernet 10 1280

# ./scripts/mmdet_train.sh faster_rcnn 10 1280
# ./scripts/mmdet_train.sh ssd 10 1280
# ./scripts/mmdet_train.sh yolov3 10 1280
# ./scripts/mmdet_train.sh retinanet 10 1280
# ./scripts/mmdet_train.sh centernet 10 1280