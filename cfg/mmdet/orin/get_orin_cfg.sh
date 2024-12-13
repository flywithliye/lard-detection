#!/bin/bash

# change path
LARD_PROJECT_ROOT_PATH=$(echo $LARD_PROJECT_ROOT_PATH)
cd $LARD_PROJECT_ROOT_PATH

# get config files into `orin`
python ./3rdparty/mmdetection/tools/misc/print_config.py 3rdparty/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --save-path cfg/mmdet/orin/orin_faster_rcnn.py
python ./3rdparty/mmdetection/tools/misc/print_config.py 3rdparty/mmdetection/configs/ssd/ssd512_coco.py --save-path cfg/mmdet/orin/orin_ssd.py
python ./3rdparty/mmdetection/tools/misc/print_config.py 3rdparty/mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py --save-path cfg/mmdet/orin/orin_yolov3.py
python ./3rdparty/mmdetection/tools/misc/print_config.py 3rdparty/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py --save-path cfg/mmdet/orin/orin_retinanet.py
python ./3rdparty/mmdetection/tools/misc/print_config.py 3rdparty/mmdetection/configs/centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py --save-path cfg/mmdet/orin/orin_centernet.py
