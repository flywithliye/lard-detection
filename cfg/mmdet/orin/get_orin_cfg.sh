cd /home/yeli/workspace/lard/lard-detection

python ./3rdparty/mmdetection/tools/misc/print_config.py 3rdparty/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --save-path cfg/mmdet/orin/orin_faster_rcnn.py
python ./3rdparty/mmdetection/tools/misc/print_config.py 3rdparty/mmdetection/configs/ssd/ssd512_coco.py --save-path cfg/mmdet/orin/orin_ssd.py
python ./3rdparty/mmdetection/tools/misc/print_config.py 3rdparty/mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py --save-path cfg/mmdet/orin/orin_yolov3.py
