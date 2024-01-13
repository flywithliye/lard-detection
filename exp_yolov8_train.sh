#!/bin/bash

# 使用方式：./exp_yolov8_train.sh <train_mode>

# 导入函数
source ~/func.sh
source src/tools/pushplus.sh

# 确保脚本参数数量正确
if [ "$#" -ne 1 ]; then
    echo_rb "使用方式: $0 <train_mode>"
    exit 1
fi

# 模式参数
train_mode=$1
valid_modes=("base" "att" "tf" "neck" "iou" "aug" "merge")
if [[ ! " ${valid_modes[*]} " =~ " ${train_mode} " ]]; then
    echo_rb "错误: train_mode 必须是以下之一: ${valid_modes[*]}"
    exit 1
fi

read -p "您选择了模式 '$train_mode' 确定继续吗？(y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "操作已取消"
    exit 0
fi

# 实验定义
run_base() {
    train_mode="base"
    echo_rb '所有训练进程已在后台启动'

    # 640
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode > logs/train/ultra_train_yolov8n_base.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_stru=p2 > logs/train/ultra_train_yolov8n_p2_base.log 2>&1

    # 1280
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --img_size=1280 > logs/train/ultra_train_yolov8n_base_1280.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_stru=p2 --img_size=1280 > logs/train/ultra_train_yolov8n_p2_base_1280.log 2>&1

    send_info "实验: $train_mode" '全部训练实验结束'
}

run_att() {
    train_mode="att"
    echo_rb '所有训练进程已在后台启动'

    # 640
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=se > logs/train/ultra_train_yolov8n_att_se.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cbam > logs/train/ultra_train_yolov8n_att_cbam.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=eca > logs/train/ultra_train_yolov8n_att_eca.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ese > logs/train/ultra_train_yolov8n_att_ese.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=gam > logs/train/ultra_train_yolov8n_att_gam.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=sa > logs/train/ultra_train_yolov8n_att_sa.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cpca > logs/train/ultra_train_yolov8n_att_cpca.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ema > logs/train/ultra_train_yolov8n_att_ema.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ta > logs/train/ultra_train_yolov8n_att_ta.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lsk > logs/train/ultra_train_yolov8n_att_lsk.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lska > logs/train/ultra_train_yolov8n_att_lska.log 2>&1

    # 1280
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=se --img_size=1280 > logs/train/ultra_train_yolov8n_att_se_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cbam --img_size=1280 > logs/train/ultra_train_yolov8n_att_cbam_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=eca --img_size=1280 > logs/train/ultra_train_yolov8n_att_eca_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ese --img_size=1280 > logs/train/ultra_train_yolov8n_att_ese_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=gam --img_size=1280 > logs/train/ultra_train_yolov8n_att_gam_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=sa --img_size=1280 > logs/train/ultra_train_yolov8n_att_sa_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cpca --img_size=1280 > logs/train/ultra_train_yolov8n_att_cpca_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ema --img_size=1280 > logs/train/ultra_train_yolov8n_att_ema_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ta --img_size=1280 > logs/train/ultra_train_yolov8n_att_ta_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lsk --img_size=1280 > logs/train/ultra_train_yolov8n_att_lsk_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lska --img_size=1280 > logs/train/ultra_train_yolov8n_att_lska_1280.log 2>&1
    
    send_info "实验: $train_mode" '全部训练实验结束'
}

run_tf() {
    train_mode="tf"
    echo_rb '所有训练进程已在后台启动'

    # 640
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=vit > logs/train/ultra_train_yolov8n_tf_vit.log 2>&1

    # 1280
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=vit --img_size=1280 > logs/train/ultra_train_yolov8n_tf_vit_1280.log 2>&1

    send_info "实验: $train_mode" '全部训练实验结束'
}

run_neck() {
    train_mode="neck"
    echo_rb '所有训练进程已在后台启动'

    # 640
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=bifpn > logs/train/ultra_train_yolov8n_fpn_bifpn.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=afpn > logs/train/ultra_train_yolov8n_fpn_afpn.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_stru=p2 --model_cfg=bifpn > logs/train/ultra_train_yolov8n_p2_fpn_bifpn.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_stru=p2 --model_cfg=afpn > logs/train/ultra_train_yolov8n_p2_fpn_afpn.log 2>&1

    # 1280
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=bifpn --img_size=1280 > logs/train/ultra_train_yolov8n_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=afpn --img_size=1280 > logs/train/ultra_train_yolov8n_fpn_afpn_1280.log 2>&1

    send_info "实验: $train_mode" '全部训练实验结束'
}

run_iou() {
    train_mode="iou"
    echo_rb '所有训练进程已在后台启动'

    # 640
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=GIoU > logs/train/ultra_train_yolov8n_iou_giou.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=DIoU > logs/train/ultra_train_yolov8n_iou_diou.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=SIoU > logs/train/ultra_train_yolov8n_iou_siou.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=EIoU > logs/train/ultra_train_yolov8n_iou_eiou.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=WIoU > logs/train/ultra_train_yolov8n_iou_wiou.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=MDPIoU1 > logs/train/ultra_train_yolov8n_iou_mdpiou1.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=MDPIoU2 > logs/train/ultra_train_yolov8n_iou_mdpiou2.log 2>&1

    # 1280
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=GIoU --img_size=1280 > logs/train/ultra_train_yolov8n_iou_giou_1280.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=DIoU --img_size=1280 > logs/train/ultra_train_yolov8n_iou_diou_1280.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=SIoU --img_size=1280 > logs/train/ultra_train_yolov8n_iou_siou_1280.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=EIoU --img_size=1280 > logs/train/ultra_train_yolov8n_iou_eiou_1280.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=WIoU --img_size=1280 > logs/train/ultra_train_yolov8n_iou_wiou_1280.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=MDPIoU1 --img_size=1280 > logs/train/ultra_train_yolov8n_iou_mdpiou1_1280.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --iou_type=MDPIoU2 --img_size=1280 > logs/train/ultra_train_yolov8n_iou_mdpiou2_1280.log 2>&1

    send_info "实验: $train_mode" '全部训练实验结束'
}

run_aug() {
    train_mode="aug"
    echo_rb '所有训练进程已在后台启动'

    # 640
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.05 > logs/train/ultra_train_yolov8n_aug05.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.1 > logs/train/ultra_train_yolov8n_aug10.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.2 > logs/train/ultra_train_yolov8n_aug20.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.4 > logs/train/ultra_train_yolov8n_aug40.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.6 > logs/train/ultra_train_yolov8n_aug60.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.8 > logs/train/ultra_train_yolov8n_aug80.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=1.0 > logs/train/ultra_train_yolov8n_aug100.log 2>&1

    # 1280
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.05 --img_size=1280 > logs/train/ultra_train_yolov8n_aug05_1280.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.1 --img_size=1280 > logs/train/ultra_train_yolov8n_aug10_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.2 --img_size=1280 > logs/train/ultra_train_yolov8n_aug20_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.4 --img_size=1280 > logs/train/ultra_train_yolov8n_aug40_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.6 --img_size=1280 > logs/train/ultra_train_yolov8n_aug60_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=0.8 --img_size=1280 > logs/train/ultra_train_yolov8n_aug80_1280.log 2>&1
    # python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --album=1.0 --img_size=1280 > logs/train/ultra_train_yolov8n_aug100_1280.log 2>&1

    send_info "实验: $train_mode" '全部训练实验结束'
}

run_merge() {
    train_mode="merge"
    echo_rb '所有训练进程已在后台启动'

    ## 640
    # Part I
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cpca --iou_type=WIoU --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge1_1.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ta --iou_type=WIoU --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge1_2.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=sa --iou_type=WIoU --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge1_3.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lska --iou_type=WIoU --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge1_4.log 2>&1
    
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cpca --iou_type=SIoU --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge2_1.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ta --iou_type=SIoU --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge2_2.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=sa --iou_type=SIoU --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge2_3.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lska --iou_type=SIoU --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge2_4.log 2>&1
    
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cpca --iou_type=MDPIoU2 --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge3_1.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ta --iou_type=MDPIoU2 --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge3_2.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=sa --iou_type=MDPIoU2 --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge3_3.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lska --iou_type=MDPIoU2 --album=0.1 > logs/train/merge/ultra_train_yolov8n_merge3_4.log 2>&1
    
    # Part II
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cpca --iou_type=WIoU --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge4_1.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ta --iou_type=WIoU --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge4_2.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=sa --iou_type=WIoU --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge4_3.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lska --iou_type=WIoU --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge4_4.log 2>&1
    
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cpca --iou_type=SIoU --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge5_1.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ta --iou_type=SIoU --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge5_2.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=sa --iou_type=SIoU --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge5_3.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lska --iou_type=SIoU --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge5_4.log 2>&1

    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=cpca --iou_type=MDPIoU2 --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge6_1.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=ta --iou_type=MDPIoU2 --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge6_2.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=sa --iou_type=MDPIoU2 --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge6_3.log 2>&1
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_cfg=lska --iou_type=MDPIoU2 --album=0.05 > logs/train/merge/ultra_train_yolov8n_merge6_4.log 2>&1

    ## 1280
    python -u cfg/ultralytics/yolov8_train.py --train_mode=$train_mode --model_stru=p2 --model_cfg=lska --iou_type=MDPIoU2 --album=0.05 --img_size=1280 > logs/train/merge/ultra_train_yolov8n_merge7_1.log 2>&1

    send_info "实验: $train_mode" '全部训练实验结束'
}

# 依据参数开展不同实验

# 0. 全部实验
if [ $train_mode == "all" ]; then
(
    run_base
    run_att
    run_tf
    run_neck
    run_iou
    run_aug
    run_merge
) &

# 1. 基础模型
elif [ $train_mode == "base" ]; then
(
    run_base
) &

# 2. 基础模型+ATT
elif [ $train_mode == "att" ]; then
(
    run_att
) &

# 3. 基础模型+Transformer
elif [ $train_mode == "tf" ]; then
(
    run_tf
) &

# 4. 基础模型+Neck
elif [ $train_mode == "neck" ]; then
(
    run_neck
) &

# 5. 基础模型+IOU
elif [ $train_mode == "iou" ]; then
(
    run_iou
) &

# 6. 基础模型+AUG
elif [ $train_mode == "aug" ]; then
(
    run_aug
) &

# 7. 融合模型
elif [ $train_mode == "merge" ]; then
(
    run_merge
) &

# -1. 异常处理
else
    echo_rb "参数错误: $test_mode"
fi
