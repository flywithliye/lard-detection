#!/bin/bash

echo_rb() { echo -e "\e[1;31m$1\e[0m"; }

# param check
if [ "$#" -ne 4 ]; then
    echo_rb "Usage: $0 <cfg_path> <best_model_path> <exp_name> <gpu_id>"
    exit 1
fi

cfg_path=$1
best_model_path=$2
exp_name=$3
gpu_id=$4

# specify GPU
export CUDA_VISIBLE_DEVICES=$gpu_id

# test - synth
python 3rdparty/mmdetection/tools/test.py \
    $cfg_path \
    $best_model_path \
    --work-dir runs/mmdetection/$exp_name/test \
    --out runs/mmdetection/$exp_name/test/coco_detection/prediction_test_synth.pkl \
    --cfg-options \
    test_evaluator.outfile_prefix=runs/mmdetection/$exp_name/test/coco_detection/prediction_test_synth

# test - nominal
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

# test - edge
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

# test - real
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

# test - test
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
