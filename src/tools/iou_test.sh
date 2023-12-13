#!/bin/bash

source ~/func.sh

# IOU损失函数测试
iou_types=('GIoU' 'DIoU' 'CIoU' 'SIoU' 'EIoU' 'WIoU' 'MDPIoU1' 'MDPIoU2')
inner_iou_values=('False' 'True')
for inner_iou in "${inner_iou_values[@]}"; do
    for iou_type in "${iou_types[@]}"; do
        if [ "$inner_iou" == "True" ]; then
            inner_iou_echo="Inner"
        else
            inner_iou_echo=""
        fi
        echo_rb "使用 $inner_iou_echo$iou_type 进行训练"
        yolo detect train \
        data=coco128.yaml \
        model=yolov8n.yaml \
        epochs=1 \
        project=test \
        iou_type=$iou_type \
        inner_iou=$inner_iou \
        plots=False
    done
done

# 删除临时测试文件
rm -r test
echo_rb "临时测试文件删除完成"