#!/bin/bash

echo_rb() { echo -e "\e[1;31m$1\e[0m"; }

# 确保脚本参数数量正确
if [ "$#" -ne 4 ]; then
    echo_rb "使用方式: $0 <cfg_path> <best_model_path> <exp_name> <gpu_id>"
    exit 1
fi

cfg_path=$1
best_model_path=$2
exp_name=$3
gpu_id=$4

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=$gpu_id

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
    test_dataloader.dataset.data_prefix.img=detection/test_real_nominal/images \
    test_evaluator.ann_file=datasets/lard/annotations/instances_test_real_nominal.json \
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
    test_dataloader.dataset.data_prefix.img=detection/test_real_edge/images \
    test_evaluator.ann_file=datasets/lard/annotations/instances_test_real_edge.json \
    test_evaluator.outfile_prefix=runs/mmdetection/$exp_name/test/coco_detection/prediction_test_real_edge

# 测试-real
python 3rdparty/mmdetection/tools/test.py \
    $cfg_path \
    $best_model_path \
    --work-dir runs/mmdetection/$exp_name/test \
    --out runs/mmdetection/$exp_name/test/coco_detection/prediction_test_real.pkl \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test_real.json \
    test_dataloader.dataset.data_prefix.img=detection/test_real/images \
    test_evaluator.ann_file=datasets/lard/annotations/instances_test_real.json \
    test_evaluator.outfile_prefix=runs/mmdetection/$exp_name/test/coco_detection/prediction_test_real

# 测试-test
python 3rdparty/mmdetection/tools/test.py \
    $cfg_path \
    $best_model_path \
    --work-dir runs/mmdetection/$exp_name/test \
    --out runs/mmdetection/$exp_name/test/coco_detection/prediction_test.pkl \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/instances_test.json \
    test_dataloader.dataset.data_prefix.img=detection/test/images \
    test_evaluator.ann_file=datasets/lard/annotations/instances_test.json \
    test_evaluator.outfile_prefix=runs/mmdetection/$exp_name/test/coco_detection/prediction_test
