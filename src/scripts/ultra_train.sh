#!/bin/bash

# 使用方式：./ultra_train.sh <model_name>

# 确保脚本参数数量正确
if [ "$#" -ne 1 ]; then
    echo "使用方式: $0 <model_name>，其中 <model_name> 必须是以下之一: ${model_list[*]}"
    exit 1
fi

model_name=$1

# 使用传入的参数启动训练进程
setsid python cfg/${model_name}_lard.py > logs/train_${model_name}.log 2>&1 &