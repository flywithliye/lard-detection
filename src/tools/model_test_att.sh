#!/bin/bash

source ./src/tools/func.sh
data=cfg/ultralytics/datasets/coco128.yaml

# 注意力模块测试-标准模型
att_types=('se' 'cbam' 'eca' 'ese' 'gam' 'sa' 'cpca' 'ema' 'ta' 'lsk' 'lska' 'vit')
for att_type in "${att_types[@]}"; do
    echo_rb "正在测试 $att_type 模型"
    yolo detect train \
        data=$data \
        model=cfg/ultralytics/models/yolov8n/yolov8n_$att_type.yaml \
        epochs=1 \
        project=test \
        plots=False
done

# 注意力模块测试-p2模型
att_types=('se' 'cbam' 'eca' 'ese' 'gam' 'sa' 'cpca' 'ema' 'ta' 'lsk' 'lska' 'vit')
for att_type in "${att_types[@]}"; do
    echo_rb "正在测试 p2_$att_type 模型"
    yolo detect train \
        data=$data \
        model=cfg/ultralytics/models/yolov8n-p2/yolov8n-p2_$att_type.yaml \
        pretrained=yolov8n.pt \
        epochs=1 \
        project=test \
        plots=False \
done

# 删除临时测试文件
rm -r test
echo_rb "临时测试文件已删除完成"