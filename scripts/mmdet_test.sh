#!/bin/bash

# 使用方式：./mmdet_test.sh <cfg_path> <best_model_path> <exp_name>

# 确保脚本参数数量正确
if [ "$#" -ne 3 ]; then
    echo "使用方式: $0 <cfg_path> <best_model_path> <exp_name>"
    exit 1
fi

cfg_path=$1
best_model_path=$2
exp_name=$3

# 测试-synth
python 3rdparty/mmdetection/tools/test.py \
    $cfg_path \
    $best_model_path \
    --work-dir runs/mmdetection/$exp_name/test \
    --out runs/mmdetection/$exp_name/test/coco_detection/prediction_test_synth.pkl \
    --cfg-options \
    test_evaluator.outfile_prefix=runs/mmdetection/$exp_name/test/coco_detection/prediction_test_synth

# 测试-nominal
python 3rdparty/mmdetection/tools/test.py \
    $cfg_path \
    $best_model_path \
    --work-dir runs/mmdetection/$exp_name/test \
    --out runs/mmdetection/$exp_name/test/coco_detection/prediction_test_real_nominal.pkl \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test_real_nominal.json \
    test_dataloader.dataset.data_prefix.img=YoloFormat/test_real_nominal/images \
    test_evaluator.ann_file=datasets/lard/annotations/instances_test_real_nominal.json \
    test_evaluator.outfile_prefix=runs/mmdetection/$exp_name/test/coco_detection/prediction_test_real_nominal

# 测试-edge
python 3rdparty/mmdetection/tools/test.py \
    $cfg_path \
    $best_model_path \
    --work-dir runs/mmdetection/$exp_name/test \
    --out runs/mmdetection/$exp_name/test/coco_detection/prediction_test_real_edge.pkl \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test_real_edge.json \
    test_dataloader.dataset.data_prefix.img=YoloFormat/test_real_edge/images \
    test_evaluator.ann_file=datasets/lard/annotations/instances_test_real_edge.json \
    test_evaluator.outfile_prefix=runs/mmdetection/$exp_name/test/coco_detection/prediction_test_real_edge
