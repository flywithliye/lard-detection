#!/bin/bash

source ./src/tools/func.sh
data=cfg/ultralytics/datasets/coco128.yaml

# test the yolo model YAML files with added up-sampling modules - standard yolo 
# 上采样模块测试-标准模型
up_types=('carafe')
for up_type in "${up_types[@]}"; do
    echo "Testing model: $up_type"
    yolo detect train \
        data=$data \
        model=cfg/ultralytics/models/yolov8n/yolov8n_$up_type.yaml \
        epochs=1 \
        project=test \
        plots=False
done

# test the yolo model YAML files with added up-sampling modules - yolo with p2-head
# 上采样模块测试-P2模型
up_types=('carafe')
for up_type in "${up_types[@]}"; do
    echo "Testing model: $up_type"
    yolo detect train \
        data=$data \
        model=cfg/ultralytics/models/yolov8n-p2/yolov8n-p2_$up_type.yaml \
        epochs=1 \
        project=test \
        plots=False
done

# delete temporary files 
# 删除临时测试文件
rm -r test
echo_rb "Temporary test files have been deleted 临时测试文件删除完成"