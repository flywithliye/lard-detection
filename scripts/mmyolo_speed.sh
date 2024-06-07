#!/bin/bash

echo_rb() { echo -e "\e[1;31m$1\e[0m"; }

# 确保脚本参数数量正确
if [ "$#" -ne 3 ]; then
    echo_rb "使用方式: $0 <cfg_path> <best_model_path> <gpu_id>"
    exit 1
fi

cfg_path=$1
best_model_path=$2
gpu_id=$3

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=$gpu_id

# 测速-synth
python 3rdparty/mmyolo/tools/analysis_tools/benchmark.py \
    $cfg_path \
    $best_model_path \
    --repeat-num 1 \
    --log-interval 500 \

# 测速-nominal
python 3rdparty/mmyolo/tools/analysis_tools/benchmark.py \
    $cfg_path \
    $best_model_path \
    --repeat-num 1 \
    --log-interval 500 \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test_real_nominal.json \
    test_dataloader.dataset.data_prefix.img=detection/test_real_nominal/images

# 测速-edge
python 3rdparty/mmyolo/tools/analysis_tools/benchmark.py \
    $cfg_path \
    $best_model_path \
    --repeat-num 1 \
    --log-interval 100 \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test_real_edge.json \
    test_dataloader.dataset.data_prefix.img=detection/test_real_edge/images
    
# 测速-real
python 3rdparty/mmyolo/tools/analysis_tools/benchmark.py \
    $cfg_path \
    $best_model_path \
    --repeat-num 1 \
    --log-interval 500 \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test_real.json \
    test_dataloader.dataset.data_prefix.img=detection/test_real/images

# 测速-test
python 3rdparty/mmyolo/tools/analysis_tools/benchmark.py \
    $cfg_path \
    $best_model_path \
    --repeat-num 1 \
    --log-interval 500 \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test.json \
    test_dataloader.dataset.data_prefix.img=detection/test/images
