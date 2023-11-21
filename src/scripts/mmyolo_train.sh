#!/bin/bash

# 使用方式：./mmyolo_train.sh <model_name> <multi_gpu>

model_list=("yolov8n")
multi_gpu=1  # 默认值为1

# 确保脚本参数数量正确
if [ "$#" -ne 2 ]; then
    echo "使用方式: $0 <model_name>  <multi_gpu>，其中 <model_name> 必须是以下之一: ${model_list[*]}"
    exit 1
fi

# 参数定义
model_name=$1
num_gpu=$2

# 检查model_name是否有效
if [[ ! " ${model_list[*]} " =~ " ${model_name} " ]]; then
    echo "错误: model_name 必须是以下之一: ${model_list[*]}"
    exit 1
fi

# 根据multi_gpu参数调整训练命令
if [[ $num_gpu > 1 ]]; then
    # 在多GPU模式下启动训练
    echo "模型: ${model_name} 多GPU训练: ${num_gpu}"
    setsid bash ./3rdparty/mmyolo/tools/dist_train.sh cfg/mmyolo/${model_name}.py ${num_gpu} > logs/train_mmyolo_${model_name}.log 2>&1 &
else
    # 在单GPU模式下启动训练
    echo "模型: ${model_name} 单GPU训练"
    setsid python 3rdparty/mmyolo/tools/train.py cfg/mmyolo/${model_name}.py > logs/train_mmyolo_${model_name}.log 2>&1 &
fi

# ./src/scripts/mmyolo_train.sh yolov8n 10