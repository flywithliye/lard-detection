#!/bin/bash

# 使用方式：./mmdet_speed.sh <cfg_path> <best_model_path>

cfg_path=$1
best_model_path=$2

# 确保脚本参数数量正确
if [ "$#" -ne 2 ]; then
    echo "使用方式: $0 <cfg_path> <best_model_path>"
    exit 1
fi

# 测速-synth
python 3rdparty/mmdetection/tools/analysis_tools/benchmark.py \
    $cfg_path \
    --checkpoint $best_model_path \
    --task inference \
    --repeat-num 1 \
    --log-interval 500 \
    --num-warmup 100 \
    --dataset-type test

# 测速-nominal
python 3rdparty/mmdetection/tools/analysis_tools/benchmark.py \
    $cfg_path \
    --checkpoint $best_model_path \
    --task inference \
    --repeat-num 1 \
    --log-interval 500 \
    --num-warmup 100 \
    --dataset-type test \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test_real_nominal.json \
    test_dataloader.dataset.data_prefix.img=YoloFormat/test_real_nominal/images

# 测速-edge
python 3rdparty/mmdetection/tools/analysis_tools/benchmark.py \
    $cfg_path \
    --checkpoint $best_model_path \
    --task inference \
    --repeat-num 1 \
    --log-interval 100 \
    --num-warmup 100 \
    --dataset-type test \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test_real_edge.json \
    test_dataloader.dataset.data_prefix.img=YoloFormat/test_real_edge/images
