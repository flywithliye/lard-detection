#!/bin/bash

# 切换路径
LARD_PROJECT_ROOT_PATH=$(echo $LARD_PROJECT_ROOT_PATH)
cd $LARD_PROJECT_ROOT_PATH

# 构建配置文件
# python 3rdparty/mmyolo/tools/misc/print_config.py 3rdparty/mmyolo/configs/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py --save-path cfg/mmyolo/orin/yolov5n.py
# python 3rdparty/mmyolo/tools/misc/print_config.py 3rdparty/mmyolo/configs/yolov6/yolov6_n_syncbn_fast_8xb32-300e_coco.py --save-path cfg/mmyolo/orin/yolov6n.py
python 3rdparty/mmyolo/tools/misc/print_config.py 3rdparty/mmyolo/configs/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py --save-path cfg/mmyolo/orin/yolov7t.py
# python 3rdparty/mmyolo/tools/misc/print_config.py 3rdparty/mmyolo/configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py --save-path cfg/mmyolo/orin/yolov8n.py
