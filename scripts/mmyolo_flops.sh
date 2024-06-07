#!/bin/bash

# 使用方式：./mmyolo_flops.sh <cfg_path> <shape>

cfg_path=$1
shape=$2

# 确保脚本参数数量正确
if [ "$#" -ne 2 ]; then
    echo_rb "使用方式: $0 <cfg_path> <shape>"
    exit 1
fi

# 计算 params & flops
python 3rdparty/mmyolo/tools/analysis_tools/get_flops.py $cfg_path --shape $shape
