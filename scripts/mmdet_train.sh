#!/bin/bash

echo_rb() { echo -e "\e[1;31m$1\e[0m"; }
model_list=("faster_rcnn" "ssd" "yolov3" "retinanet" "centernet")

# 确保脚本参数数量正确
if [ "$#" -ne 3 ]; then
    echo_rb "使用方式: $0 <model_name> <num_gpu> <img_size>，其中 <model_name> 必须是以下之一: ${model_list[*]}"
    exit 1
fi

# 参数定义
model_name=$1
num_gpu=$2
img_size=$3

# 检查model_name是否有效
if [[ ! " ${model_list[*]} " =~ " ${model_name} " ]]; then
    echo_rb "错误: model_name 必须是以下之一: ${model_list[*]}"
    exit 1
fi

# 检查num_gpu是否在1到10之间
if [[ $num_gpu -lt 1 || $num_gpu -gt 10 ]]; then
    echo_rb "错误: num_gpu 的值必须在 1 到 10 之间"
    exit 1
fi

# 根据num_gpu参数调整训练命令
if [[ $num_gpu > 1 ]]; then
    # 在多GPU模式下启动训练
    echo_rb "模型: ${model_name} 多GPU训练: ${num_gpu}"
    setsid bash ./3rdparty/mmdetection/tools/dist_train.sh cfg/mmdet/${model_name}_${img_size}.py ${num_gpu} > logs/train/mmdet_train_${model_name}_${img_size}.log 2>&1 &
else
    # 在单GPU模式下启动训练
    echo_rb "模型: ${model_name} 单GPU训练"
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