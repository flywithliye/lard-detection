#!/bin/bash

# Usage./mmyolo_flops.sh <cfg_path> <shape>

cfg_path=$1
shape=$2

# param check
if [ "$#" -ne 2 ]; then
    echo_rb "Usage: $0 <cfg_path> <shape>"
    exit 1
fi

# get params & flops
python 3rdparty/mmyolo/tools/analysis_tools/get_flops.py $cfg_path --shape $shape
