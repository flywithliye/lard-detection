#!/bin/bash

source ./src/tools/func.sh
data=cfg/ultralytics/datasets/coco128.yaml

# test the yolo model YAML files with added attention modules - standard yolo 
# 注意力模块测试-标准模型
att_types=('se' 'cbam' 'eca' 'ese' 'gam' 'sa' 'cpca' 'ema' 'ta' 'lsk' 'lska' 'vit')
for att_type in "${att_types[@]}"; do
    echo_rb "Testing model: $att_type"
    yolo detect train \
        data=$data \
        model=cfg/ultralytics/models/yolov8n/yolov8n_$att_type.yaml \
        epochs=1 \
        project=test \
        plots=False
done

# test the yolo model YAML files with added attention modules - yolo with p2-head
#  注意力模块测试-p2模型
att_types=('se' 'cbam' 'eca' 'ese' 'gam' 'sa' 'cpca' 'ema' 'ta' 'lsk' 'lska' 'vit')
for att_type in "${att_types[@]}"; do
    echo_rb "Testing model: p2_$att_type"
    yolo detect train \
        data=$data \
        model=cfg/ultralytics/models/yolov8n-p2/yolov8n-p2_$att_type.yaml \
        pretrained=yolov8n.pt \
        epochs=1 \
        project=test \
        plots=False \
done

# delete temporary files 
# 删除临时测试文件
rm -r test
echo_rb "Temporary test files have been deleted 临时测试文件已删除完成"