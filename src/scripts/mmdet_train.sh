#!/bin/bash

# 使用方式：./mmdet_train.sh <model_name> <multi_gpu>

model_list=("faster_rcnn" "ssd" "yolov3")
multi_gpu=false  # 默认值为false

# 确保脚本参数数量正确
if [ "$#" -ne 1 ]; then
    echo "使用方式: $0 <model_name>，其中 <model_name> 必须是以下之一: ${model_list[*]}"
    exit 1
fi

# 参数定义
model_name=$1
multi_gpu=$2
num_gpu=8

# 检查model_name是否有效
if [[ ! " ${model_list[*]} " =~ " ${model_name} " ]]; then
    echo "错误: model_name 必须是以下之一: ${model_list[*]}"
    exit 1
fi

# 根据multi_gpu参数调整训练命令
if [[ $multi_gpu = true ]]; then
    # 在多GPU模式下启动训练
    setsid bash ./3rdparty/mmdetection/tools/dist_train.sh cfg/${model_name}_lard.py ${num_gpu} > logs/train_${model_name}.log 2>&1 &
else
    # 在单GPU模式下启动训练
    setsid python 3rdparty/mmdetection/tools/train.py cfg/${model_name}_lard.py > logs/train_${model_name}.log 2>&1 &
fi
