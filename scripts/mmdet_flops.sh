#!/bin/bash

# 使用方式：./mmdet_flops.sh <cfg_path> 

cfg_path=$1

# 确保脚本参数数量正确
if [ "$#" -ne 1 ]; then
    echo_rb "使用方式: $0 <cfg_path>"
    exit 1
fi

# 计算params & flops
python 3rdparty/mmdetection/tools/analysis_tools/get_flops.py $cfg_path 
