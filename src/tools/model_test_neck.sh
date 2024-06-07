#!/bin/bash

source ./src/tools/func.sh
data=cfg/ultralytics/datasets/coco128.yaml

# Neck模块测试-标准模型
neck_types=('bifpn' 'afpn' 'cbam_bifpn' 'cbam_afpn')
for neck_type in "${neck_types[@]}"; do
    echo_rb "正在测试 yolov8n_$neck_type 模型"
    yolo detect train \
        data=$data \
        model=cfg/ultralytics/models/yolov8n/yolov8n_$neck_type.yaml \
        epochs=1 \
        project=test \
        plots=False
done

# Neck模块测试-p2模型
neck_types=('bifpn' 'afpn' 'cbam_bifpn' 'cbam_afpn')
for neck_type in "${neck_types[@]}"; do
    echo_rb "正在测试 yolov8n-p2_$neck_type 模型"
    yolo detect train \
        data=$data \
        model=cfg/ultralytics/models/yolov8n-p2/yolov8n-p2_$neck_type.yaml \
        epochs=1 \
        project=test \
        plots=False
done

# 删除临时测试文件
rm -r test
echo_rb "临时测试文件删除完成"