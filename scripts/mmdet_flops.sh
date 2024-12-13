#!/bin/bash

# Usage: ./mmdet_flops.sh <cfg_path> 

cfg_path=$1

# param check
if [ "$#" -ne 1 ]; then
    echo_rb "Usage: $0 <cfg_path>"
    exit 1
fi

# get params & flops
python 3rdparty/mmdetection/tools/analysis_tools/get_flops.py $cfg_path 
