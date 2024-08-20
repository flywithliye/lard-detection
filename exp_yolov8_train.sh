#!/bin/bash

# 使用方式：./exp_yolov8_train.sh <mode>

# 导入函数
source src/tools/func.sh
source src/tools/pushplus.sh

# 确保脚本参数数量正确
if [ "$#" -ne 1 ]; then
    echo_rb "使用方式: $0 <mode>"
    exit 1
fi

# 模式参数
mode=$1
valid_modes=("base" "att" "tf" "fpn" "iou" "aug" "up" "merge" "finetune")
if [[ ! " ${valid_modes[*]} " =~ " ${mode} " ]]; then
    echo_rb "错误: mode 必须是以下之一: ${valid_modes[*]}"
    exit 1
fi

read -p "您选择了模式 '$mode' 确定继续吗？(y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "操作已取消"
    exit 0
fi

# 实验定义
run_base() {
    mode="base"
    echo_rb '所有训练进程已在后台启动'

    # n_640
    # python -u cfg/ultralytics/train.py --mode=$mode > logs/train/ultra_train_yolov8n_base_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 > logs/train/ultra_train_yolov8n_base_p2_640.log 2>&1

    # s_640
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s > logs/train/ultra_train_yolov8s_base_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --stru=p2 > logs/train/ultra_train_yolov8s_base_p2_640.log 2>&1

    # v10_n_640
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov10n > logs/train/ultra_train_yolov10n_base_640.log 2>&1

    # n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --size=1280 > logs/train/ultra_train_yolov8n_base_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --size=1280 > logs/train/ultra_train_yolov8n_p2_base_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p6 --size=1280 > logs/train/ultra_train_yolov8s_p2_base_1280.log 2>&1

    # s_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --size=1280 > logs/train/ultra_train_yolov8s_base_1280.log 2>&1

    send_info "实验: $mode" '全部训练实验结束'
}

run_att() {
    echo_rb '所有训练进程已在后台启动'

    # n_640
    # mode="att"
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=se > logs/train/ultra_train_yolov8n_att_se_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=cbam > logs/train/ultra_train_yolov8n_att_cbam_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=eca > logs/train/ultra_train_yolov8n_att_eca_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ese > logs/train/ultra_train_yolov8n_att_ese_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=gam > logs/train/ultra_train_yolov8n_att_gam_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=sa > logs/train/ultra_train_yolov8n_att_sa_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=cpca > logs/train/ultra_train_yolov8n_att_cpca_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ema > logs/train/ultra_train_yolov8n_att_ema_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ta > logs/train/ultra_train_yolov8n_att_ta_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lsk > logs/train/ultra_train_yolov8n_att_lsk_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska > logs/train/ultra_train_yolov8n_att_lska_640.log 2>&1

    # mode="atts"
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ses > logs/train/ultra_train_yolov8n_atts_ses_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=cbams > logs/train/ultra_train_yolov8n_atts_cbams_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ecas > logs/train/ultra_train_yolov8n_atts_ecas_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=eses > logs/train/ultra_train_yolov8n_atts_eses_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=gams > logs/train/ultra_train_yolov8n_atts_gams_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=sas > logs/train/ultra_train_yolov8n_atts_sas_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=cpcas > logs/train/ultra_train_yolov8n_atts_cpcas_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=emas > logs/train/ultra_train_yolov8n_atts_emas_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas > logs/train/ultra_train_yolov8n_atts_tas_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lsks > logs/train/ultra_train_yolov8n_atts_lsks_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lskas > logs/train/ultra_train_yolov8n_atts_lskas_640.log 2>&1
    
    # s_640
    # mode="att"
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=se > logs/train/ultra_train_yolov8s_att_se_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=cbam > logs/train/ultra_train_yolov8s_att_cbam_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=eca > logs/train/ultra_train_yolov8s_att_eca_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=ese > logs/train/ultra_train_yolov8s_att_ese_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=gam > logs/train/ultra_train_yolov8s_att_gam_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=sa > logs/train/ultra_train_yolov8s_att_sa_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=cpca > logs/train/ultra_train_yolov8s_att_cpca_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=ema > logs/train/ultra_train_yolov8s_att_ema_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=ta > logs/train/ultra_train_yolov8s_att_ta_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=lsk > logs/train/ultra_train_yolov8s_att_lsk_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=lska > logs/train/ultra_train_yolov8s_att_lska_640.log 2>&1

    # p2_n_640
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=se > logs/train/ultra_train_yolov8n_p2_att_se_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=cbam > logs/train/ultra_train_yolov8n_p2_att_cbam_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=eca > logs/train/ultra_train_yolov8n_p2_att_eca_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=ese > logs/train/ultra_train_yolov8n_p2_att_ese_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=gam > logs/train/ultra_train_yolov8n_p2_att_gam_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=sa > logs/train/ultra_train_yolov8n_p2_att_sa_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=cpca > logs/train/ultra_train_yolov8n_p2_att_cpca_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=ema > logs/train/ultra_train_yolov8n_p2_att_ema_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=ta > logs/train/ultra_train_yolov8n_p2_att_ta_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk > logs/train/ultra_train_yolov8n_p2_att_lsk_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lska > logs/train/ultra_train_yolov8n_p2_att_lska_640.log 2>&1

    # n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=se --size=1280 > logs/train/ultra_train_yolov8n_att_se_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=cbam --size=1280 > logs/train/ultra_train_yolov8n_att_cbam_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=eca --size=1280 > logs/train/ultra_train_yolov8n_att_eca_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ese --size=1280 > logs/train/ultra_train_yolov8n_att_ese_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=gam --size=1280 > logs/train/ultra_train_yolov8n_att_gam_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=sa --size=1280 > logs/train/ultra_train_yolov8n_att_sa_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=cpca --size=1280 > logs/train/ultra_train_yolov8n_att_cpca_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ema --size=1280 > logs/train/ultra_train_yolov8n_att_ema_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ta --size=1280 > logs/train/ultra_train_yolov8n_att_ta_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lsk --size=1280 > logs/train/ultra_train_yolov8n_att_lsk_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska --size=1280 > logs/train/ultra_train_yolov8n_att_lska_1280.log 2>&1
    
    # p2_n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=se --size=1280 > logs/train/ultra_train_yolov8n_p2_att_se_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=cbam --size=1280 > logs/train/ultra_train_yolov8n_p2_att_cbam_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=eca --size=1280 > logs/train/ultra_train_yolov8n_p2_att_eca_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=ese --size=1280 > logs/train/ultra_train_yolov8n_p2_att_ese_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=gam --size=1280 > logs/train/ultra_train_yolov8n_p2_att_gam_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=sa --size=1280 > logs/train/ultra_train_yolov8n_p2_att_sa_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=cpca --size=1280 > logs/train/ultra_train_yolov8n_p2_att_cpca_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=ema --size=1280 > logs/train/ultra_train_yolov8n_p2_att_ema_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=ta --size=1280 > logs/train/ultra_train_yolov8n_p2_att_ta_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk --size=1280 > logs/train/ultra_train_yolov8n_p2_att_lsk_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lska --size=1280 > logs/train/ultra_train_yolov8n_p2_att_lska_1280.log 2>&1
    
    send_info "实验: $mode" '全部训练实验结束'
}

run_tf() {
    echo_rb '所有训练进程已在后台启动'

    # n_640
    # mode="tf"
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=vit > logs/train/ultra_train_yolov8n_tf_vit.log 2>&1
    
    # mode="tfs"
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=vits > logs/train/ultra_train_yolov8n_tfs_vits.log 2>&1

    # s_640
    # mode="tf"
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=vit > logs/train/ultra_train_yolov8s_tf_vit.log 2>&1
    
    # p2_n_640
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=vit > logs/train/ultra_train_yolov8n_p2_tf_vit.log 2>&1

    # n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=vit --size=1280 > logs/train/ultra_train_yolov8n_tf_vit_1280.log 2>&1

    send_info "实验: $mode" '全部训练实验结束'
}

run_fpn() {
    mode="fpn"
    echo_rb '所有训练进程已在后台启动'

    # n_640
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=bifpn > logs/train/ultra_train_yolov8n_fpn_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=afpn > logs/train/ultra_train_yolov8n_fpn_afpn_640.log 2>&1
    
    # s_640
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=bifpn > logs/train/ultra_train_yolov8s_fpn_bifpn_640.log 2>&1
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=afpn > logs/train/ultra_train_yolov8s_fpn_afpn_640.log 2>&1
    
    # p2_n_640
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=bifpn > logs/train/ultra_train_yolov8n_p2_fpn_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=afpn > logs/train/ultra_train_yolov8n_p2_fpn_afpn_640.log 2>&1

    # n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=bifpn --size=1280 > logs/train/ultra_train_yolov8n_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=afpn --size=1280 > logs/train/ultra_train_yolov8n_fpn_afpn_1280.log 2>&1
    
    # p2_n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=afpn --size=1280 > logs/train/ultra_train_yolov8n_p2_fpn_afpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_fpn_bifpn_1280.log 2>&1
    
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=se_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_se_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=cbam_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_cbam_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=eca_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_eca_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=ese_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_ese_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=gam_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_gam_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=sa_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_sa_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=cpca_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_cpca_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=ema_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_ema_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=ta_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_ta_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_lsk_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lska_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_lska_fpn_bifpn_1280.log 2>&1
    
    # c2f_att
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=c2f_se_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_c2f_se_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=c2f_cbam_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_c2f_cbam_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=c2f_eca_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_c2f_eca_fpn_bifpn_1280.log 2>&1

    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=carafe_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_up_carafe_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=se_carafe_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_se_up_carafe_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=c2f_se_carafe_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_att_c2f_se_up_carafe_fpn_bifpn_1280.log 2>&1
    
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=dysample_bifpn --size=1280 > logs/train/ultra_train_yolov8n_p2_up_dysample_fpn_bifpn_1280.log 2>&1
    
    send_info "实验: $mode" '全部训练实验结束'
}

run_iou() {
    mode="iou"
    echo_rb '所有训练进程已在后台启动'

    # n_640
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=GIoU > logs/train/ultra_train_yolov8n_iou_giou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=DIoU > logs/train/ultra_train_yolov8n_iou_diou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=SIoU > logs/train/ultra_train_yolov8n_iou_siou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=EIoU > logs/train/ultra_train_yolov8n_iou_eiou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=WIoU > logs/train/ultra_train_yolov8n_iou_wiou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=MDPIoU1 > logs/train/ultra_train_yolov8n_iou_mdpiou1_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=MDPIoU2 > logs/train/ultra_train_yolov8n_iou_mdpiou2_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=ShapeIoU > logs/train/ultra_train_yolov8n_iou_shapeiou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=NWD > logs/train/ultra_train_yolov8n_iou_nwd_640.log 2>&1
    
    # s_640
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --iou_type=GIoU > logs/train/ultra_train_yolov8s_iou_giou_640.log 2>&1
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --iou_type=DIoU > logs/train/ultra_train_yolov8s_iou_diou_640.log 2>&1
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --iou_type=SIoU > logs/train/ultra_train_yolov8s_iou_siou_640.log 2>&1
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --iou_type=EIoU > logs/train/ultra_train_yolov8s_iou_eiou_640.log 2>&1
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --iou_type=WIoU > logs/train/ultra_train_yolov8s_iou_wiou_640.log 2>&1
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --iou_type=MDPIoU1 > logs/train/ultra_train_yolov8s_iou_mdpiou1_640.log 2>&1
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --iou_type=MDPIoU2 > logs/train/ultra_train_yolov8s_iou_mdpiou2_640.log 2>&1
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --iou_type=ShapeIoU > logs/train/ultra_train_yolov8s_iou_shapeiou_640.log 2>&1
    python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --iou_type=NWD > logs/train/ultra_train_yolov8s_iou_nwd_640.log 2>&1


    # n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=GIoU --size=1280 > logs/train/ultra_train_yolov8n_iou_giou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=DIoU --size=1280 > logs/train/ultra_train_yolov8n_iou_diou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=SIoU --size=1280 > logs/train/ultra_train_yolov8n_iou_siou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=EIoU --size=1280 > logs/train/ultra_train_yolov8n_iou_eiou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=WIoU --size=1280 > logs/train/ultra_train_yolov8n_iou_wiou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=MDPIoU1 --size=1280 > logs/train/ultra_train_yolov8n_iou_mdpiou1_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=MDPIoU2 --size=1280 > logs/train/ultra_train_yolov8n_iou_mdpiou2_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=ShapeIoU --size=1280 > logs/train/ultra_train_yolov8n_iou_shapeiou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --iou_type=NWD --size=1280 > logs/train/ultra_train_yolov8n_iou_nwd_1280.log 2>&1

    send_info "实验: $mode" '全部训练实验结束'
}

run_up() {
    mode="up"
    echo_rb '所有训练进程已在后台启动'

    # n_640
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=carafe > logs/train/ultra_train_yolov8n_up_carafe_640.log 2>&1
    
    # s_640
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --cfg=carafe > logs/train/ultra_train_yolov8s_up_carafe_640.log 2>&1

    # n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=carafe --size=1280 > logs/train/ultra_train_yolov8n_up_carafe_1280.log 2>&1
    
    # p2_n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=carafe --size=1280 > logs/train/ultra_train_yolov8n_up_p2_carafe_1280.log 2>&1

    send_info "实验: $mode" '全部训练实验结束'
}

run_aug() {
    mode="aug"
    echo_rb '所有训练进程已在后台启动'

    # n_640
    # python -u cfg/ultralytics/train.py --mode=$mode --aug_json=all --album=0.01 > logs/train/ultra_train_yolov8n_aug_all_01_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --aug_json=all --album=0.05 > logs/train/ultra_train_yolov8n_aug_all_05_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --aug_json=all --album=0.10 > logs/train/ultra_train_yolov8n_aug_all_10_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --aug_json=all --album=0.15 > logs/train/ultra_train_yolov8n_aug_all_15_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --aug_json=all --album=0.20 > logs/train/ultra_train_yolov8n_aug_all_20_640.log 2>&1
    
    # s_640
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --aug_json=all --album=0.01 > logs/train/ultra_train_yolov8s_aug_all_01_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --aug_json=all --album=0.05 > logs/train/ultra_train_yolov8s_aug_all_05_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --aug_json=all --album=0.10 > logs/train/ultra_train_yolov8s_aug_all_10_640.log 2>&1
    # todo python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --aug_json=all --album=0.15 > logs/train/ultra_train_yolov8s_aug_all_15_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --model=yolov8s --aug_json=all --album=0.20 > logs/train/ultra_train_yolov8s_aug_all_20_640.log 2>&1

    # n_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --album=0.05 --size=1280 > logs/train/ultra_train_yolov8n_aug05_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --album=0.1 --size=1280 > logs/train/ultra_train_yolov8n_aug10_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --album=0.2 --size=1280 > logs/train/ultra_train_yolov8n_aug20_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --album=0.4 --size=1280 > logs/train/ultra_train_yolov8n_aug40_1280.log 2>&1
    
    # p2_1280
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --aug_json=all --album=0.01 --size=1280 > logs/train/ultra_train_yolov8n_p2_aug_all_01_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --aug_json=all --album=0.05 --size=1280 > logs/train/ultra_train_yolov8n_p2_aug_all_05_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --aug_json=all --album=0.10 --size=1280 > logs/train/ultra_train_yolov8n_p2_aug_all_10_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --aug_json=all --album=0.15 --size=1280 > logs/train/ultra_train_yolov8n_p2_aug_all_15_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --aug_json=all --album=0.20 --size=1280 > logs/train/ultra_train_yolov8n_p2_aug_all_20_1280.log 2>&1

    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --aug_json=color --album=0.10 --size=1280 > logs/train/ultra_train_yolov8n_p2_aug_color_10_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --aug_json=color_blur --album=0.10 --size=1280 > logs/train/ultra_train_yolov8n_p2_aug_color_blur_10_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --aug_json=color_blur_geo --album=0.10 --size=1280 > logs/train/ultra_train_yolov8n_p2_aug_color_blur_geo_10_1280.log 2>&1
    
    send_info "实验: $mode" '全部训练实验结束'
}


run_merge() {
    mode="merge"
    echo_rb '所有训练进程已在后台启动'

    ## 640
    # ATT + FPN
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=cpca_bifpn > logs/train/ultra_train_yolov8n_merge_att_fpn_cpca_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ese_bifpn > logs/train/ultra_train_yolov8n_merge_att_fpn_ese_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ta_bifpn > logs/train/ultra_train_yolov8n_merge_att_fpn_ta_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn > logs/train/ultra_train_yolov8n_merge_att_fpn_lska_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=sa_bifpn > logs/train/ultra_train_yolov8n_merge_att_fpn_sa_bifpn_640.log 2>&1

    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=emas_bifpn > logs/train/ultra_train_yolov8n_merge_atts_fpn_emas_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=cbams_bifpn > logs/train/ultra_train_yolov8n_merge_atts_fpn_cbams_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=gams_bifpn > logs/train/ultra_train_yolov8n_merge_atts_fpn_gams_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn > logs/train/ultra_train_yolov8n_merge_atts_fpn_tas_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=eses_bifpn > logs/train/ultra_train_yolov8n_merge_atts_fpn_eses_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=ses_bifpn > logs/train/ultra_train_yolov8n_merge_atts_fpn_ses_bifpn_640.log 2>&1

    # ATT + IOU
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska --iou_type=EIoU > logs/train/ultra_train_yolov8n_merge_att_iou_lska_eiou_640.log 2>&1
    
    # ATT + AUG
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska --aug_json=all --album=0.10 > logs/train/ultra_train_yolov8n_merge_att_aug_lska_aug_all_10_640.log 2>&1
    
    # FPN + AUG
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=bifpn --aug_json=all --album=0.10 > logs/train/ultra_train_yolov8n_merge_fpn_aug_bifpn_aug_all_10_640.log 2>&1

    # FPN + IOU
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=bifpn --iou_type=EIoU > logs/train/ultra_train_yolov8n_merge_fpn_iou_bifpn_eiou_640.log 2>&1
    
    # P2 + ATT + FPN
    # todo python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lska_bifpn > logs/train/ultra_train_yolov8n_merge_head_att_fpn_p2_lska_bifpn_640.log 2>&1

    # ATT + FPN + IOU
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=MDPIoU1 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_lska_bifpn_mdpiou1_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=GIoU > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_lska_bifpn_giou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=MDPIoU2 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_lska_bifpn_mdpiou2_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=DIoU > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_lska_bifpn_diou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=SIoU > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_lska_bifpn_siou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=EIoU > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_lska_bifpn_eiou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=WIoU > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_lska_bifpn_wiou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=ShapeIoU > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_lska_bifpn_shapeiou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=NWD > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_lska_bifpn_nwd_640.log 2>&1

    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=MDPIoU1 > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_tas_bifpn_mdpiou1_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=GIoU > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_tas_bifpn_giou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=MDPIoU2 > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_tas_bifpn_mdpiou2_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=DIoU > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_tas_bifpn_diou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=SIoU > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_tas_bifpn_siou_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=ShapeIoU > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_tas_bifpn_shapeiou_640.log 2>&1

    # ATT + FPN + AUG
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --aug_json=all --album=0.01 > logs/train/ultra_train_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_01_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --aug_json=all --album=0.05 > logs/train/ultra_train_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_05_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --aug_json=all --album=0.10 > logs/train/ultra_train_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_10_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --aug_json=all --album=0.15 > logs/train/ultra_train_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_15_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --aug_json=all --album=0.20 > logs/train/ultra_train_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_20_640.log 2>&1

    # ATT + FPN + UP + AUG
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn_carafe --aug_json=all --album=0.10 > logs/train/ultra_train_yolov8n_merge_att_fpn_up_aug_lska_bifpn_carafe_aug_all_10_640.log 2>&1
    
    # ATT + FPN + IOU + UP
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn_carafe --iou_type=ShapeIoU > logs/train/ultra_train_yolov8n_merge_atts_fpn_up_iou_tas_bifpn_carafe_shapeiou_640.log 2>&1

    # ATT + FPN + IOU + AUG
    python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.01 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_01_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.05 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_05_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.10 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_10_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.20 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_20_640.log 2>&1

    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=ShapeIoU --aug_json=all --album=0.01 > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_aug_tas_bifpn_shapeiou_aug_all_01_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=ShapeIoU --aug_json=all --album=0.05 > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_aug_tas_bifpn_shapeiou_aug_all_05_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=ShapeIoU --aug_json=all --album=0.10 > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_aug_tas_bifpn_shapeiou_aug_all_10_640.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=tas_bifpn --iou_type=ShapeIoU --aug_json=all --album=0.15 > logs/train/ultra_train_yolov8n_merge_atts_fpn_iou_aug_tas_bifpn_shapeiou_aug_all_15_640.log 2>&1

    ## 1280
    # ATT + FPN + AUG
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --aug_json=all --album=0.10 --size=1280 > logs/train/ultra_train_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_10_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --aug_json=all --album=0.15 --size=1280 > logs/train/ultra_train_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_15_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --aug_json=all --album=0.20 --size=1280 > logs/train/ultra_train_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_20_1280.log 2>&1

    # ATT + FPN + IOU + AUG
    python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.01 --size=1280 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_01_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.05 --size=1280 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_05_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.10 --size=1280 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_10_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.20 --size=1280 > logs/train/ultra_train_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_20_1280.log 2>&1

    ## 1280
    # template python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lska --iou_type=MDPIoU2 --album=0.05 --size=1280 > logs/train/merge/ultra_train_yolov8n_merge7_1.log 2>&1
    
    # 2
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=bifpn --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=afpn --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_afpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --iou_type=DIoU --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_diou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lsk_bifpn --size=1280 > logs/train/ultra_train_yolov8n_merge_lsk_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=bifpn --iou_type=DIoU --size=1280 > logs/train/ultra_train_yolov8n_merge_bifpn_diou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lsk --iou_type=DIoU --size=1280 > logs/train/ultra_train_yolov8n_merge_lsk_diou_1280.log 2>&1

    # 3
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=bifpn --iou_type=DIoU --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_bifpn_diou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk --iou_type=DIoU --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_diou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --cfg=lsk_bifpn --iou_type=DIoU --size=1280 > logs/train/ultra_train_yolov8n_merge_lsk_bifpn_diou_1280.log 2>&1
    
    # 4
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_bifpn_diou_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.01 --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_bifpn_diou_aug01_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.05 --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_bifpn_diou_aug05_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.1 --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_bifpn_diou_aug10_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.15 --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_bifpn_diou_aug15_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.2 --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_bifpn_diou_aug20_1280.log 2>&1

    # 5
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn_carafe --iou_type=DIoU --album=0.05 --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_bifpn_carafe_diou_aug05_1280.log 2>&1
    # python -u cfg/ultralytics/train.py --mode=$mode --stru=p2 --cfg=lsk_bifpn_carafe --iou_type=DIoU --album=0.1 --size=1280 > logs/train/ultra_train_yolov8n_merge_p2_lsk_bifpn_carafe_diou_aug10_1280.log 2>&1

    send_info "实验: $mode" '全部训练实验结束'
}

run_finetune() {
    mode="finetune"
    echo_rb '所有训练进程已在后台启动'

    # bars
    python -u cfg/ultralytics/train.py \
        --mode=$mode \
        --finetune_mode=bars_runway_val_test \
        --weights=runs/ultralytics/finetune/yolov8n_lska_bifpn_EIoU_aug_all_10_640/triple_split/train/weights/best.pt \
        --cfg=lska_bifpn \
        --iou_type=EIoU \
        --aug_json=all \
        --album=0.10 > logs/train/ultra_train_yolov8n_finetune_bars_runway_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # msfs
    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=fs2020_runway_test \
    #     --weights=runs/ultralytics/finetune/yolov8n_lska_bifpn_EIoU_aug_all_10_640/triple_split/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.10 > logs/train/ultra_train_yolov8n_finetune_fs2020_runway_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # !5%
    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_EIoU_aug_all_5_640/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.05 > logs/train/ultra_train_yolov8n_finetune_single_lska_bifpn_eiou_aug_all_05_640.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_EIoU_aug_all_5_640/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.05 > logs/train/ultra_train_yolov8n_finetune_double_lska_bifpn_eiou_aug_all_05_640.log 2>&1
        
    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_EIoU_aug_all_5_640/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.05 > logs/train/ultra_train_yolov8n_finetune_triple_lska_bifpn_eiou_aug_all_05_640.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --weights=runs/ultralytics/finetune/yolov8n_lska_bifpn_EIoU_aug_all_5_640/double/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.05 > logs/train/ultra_train_yolov8n_finetune_triple_split_lska_bifpn_eiou_aug_all_05_640.log 2>&1

    # !10%
    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_EIoU_aug_all_10_640/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.10 > logs/train/ultra_train_yolov8n_finetune_single_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_EIoU_aug_all_10_640/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.10 > logs/train/ultra_train_yolov8n_finetune_double_lska_bifpn_eiou_aug_all_10_640.log 2>&1
        
    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_EIoU_aug_all_10_640/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.10 > logs/train/ultra_train_yolov8n_finetune_triple_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --weights=runs/ultralytics/finetune/yolov8n_lska_bifpn_EIoU_aug_all_10_640/double/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.10 > logs/train/ultra_train_yolov8n_finetune_triple_split_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # !20%
    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_EIoU_aug_all_20_640/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.20 > logs/train/ultra_train_yolov8n_finetune_single_lska_bifpn_eiou_aug_all_20_640.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_EIoU_aug_all_20_640/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.20 > logs/train/ultra_train_yolov8n_finetune_double_lska_bifpn_eiou_aug_all_20_640.log 2>&1
        
    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_EIoU_aug_all_20_640/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.20 > logs/train/ultra_train_yolov8n_finetune_triple_lska_bifpn_eiou_aug_all_20_640.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --weights=runs/ultralytics/finetune/yolov8n_lska_bifpn_EIoU_aug_all_20_640/double/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.20 > logs/train/ultra_train_yolov8n_finetune_triple_split_lska_bifpn_eiou_aug_all_20_640.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_aug_all_10/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --aug_json=all \
    #     --album=0.10 > logs/train/ultra_train_yolov8n_finetune_single_lska_bifpn_aug_all_10_640.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_aug_all_10/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --aug_json=all \
    #     --album=0.10 > logs/train/ultra_train_yolov8n_finetune_double_lska_bifpn_aug_all_10_640.log 2>&1
        
    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --weights=runs/ultralytics/merge/yolov8n_lska_bifpn_aug_all_10/train/weights/best.pt \
    #     --cfg=lska_bifpn \
    #     --aug_json=all \
    #     --album=0.10 > logs/train/ultra_train_yolov8n_finetune_triple_lska_bifpn_aug_all_10_640.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --weights=runs/ultralytics/finetune/yolov8n_lska_bifpn_aug_all_10/double/train/weights/last.pt \
    #     --cfg=lska_bifpn \
    #     --aug_json=all \
    #     --album=0.10 > logs/train/ultra_train_yolov8n_finetune_triple_split_lska_bifpn_aug_all_10_640.log 2>&1



    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --weights=runs/ultralytics/merge/yolov8n-p2_lsk_bifpn_DIoU_aug10_1280/train/weights/best.pt \
    #     --stru=p2 \
    #     --cfg=lsk_bifpn \
    #     --iou_type=DIoU \
    #     --album=0.1 \
    #     --size=1280 > logs/train/ultra_train_yolov8n_finetune_single_p2_lsk_bifpn_diou_aug10_1280.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --weights=runs/ultralytics/merge/yolov8n-p2_lsk_bifpn_DIoU_aug10_1280/train/weights/best.pt \
    #     --stru=p2 \
    #     --cfg=lsk_bifpn \
    #     --iou_type=DIoU \
    #     --album=0.1 \
    #     --size=1280 > logs/train/ultra_train_yolov8n_finetune_double_p2_lsk_bifpn_diou_aug10_1280.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --weights=runs/ultralytics/merge/yolov8n-p2_lsk_bifpn_DIoU_aug10_1280/train/weights/best.pt \
    #     --stru=p2 \
    #     --cfg=lsk_bifpn \
    #     --iou_type=DIoU \
    #     --album=0.1 \
    #     --size=1280 > logs/train/ultra_train_yolov8n_finetune_triple_p2_lsk_bifpn_diou_aug10_1280.log 2>&1

    # python -u cfg/ultralytics/train.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --weights=runs/ultralytics/finetune/yolov8n-p2_lsk_bifpn_DIoU_aug10_1280/double/train/weights/last.pt \
    #     --stru=p2 \
    #     --cfg=lsk_bifpn \
    #     --iou_type=DIoU \
    #     --album=0.1 \
    #     --size=1280 > logs/train/ultra_train_yolov8n_finetune_triple_split_p2_lsk_bifpn_diou_aug10_1280.log 2>&1

    send_info "实验: $mode" '全部训练实验结束'
}

# 依据参数开展不同实验

# 0. 全部实验
if [ $mode == "all" ]; then
(
    run_base
    run_att
    run_tf
    run_fpn
    run_iou
    run_aug
    run_merge
) &

# 1. 基础模型
elif [ $mode == "base" ]; then
(
    run_base
) &

# 2. 基础模型+ATT
elif [ $mode == "att" ]; then
(
    run_att
) &

# 3. 基础模型+Transformer
elif [ $mode == "tf" ]; then
(
    run_tf
) &

# 4. 基础模型+fpn
elif [ $mode == "fpn" ]; then
(
    run_f p n
) &

# 5. 基础模型+IOU
elif [ $mode == "iou" ]; then
(
    run_iou
) &

# 6. 基础模型+AUG
elif [ $mode == "aug" ]; then
(
    run_aug
) &

# 7. 基础模型+UP
elif [ $mode == "up" ]; then
(
    run_up
) &

# 8. 融合模型
elif [ $mode == "merge" ]; then
(
    run_merge
) &

# 9. 微调模型
elif [ $mode == "finetune" ]; then
(
    run_finetune
) &

# -1. 异常处理
else
    echo_rb "参数错误: $mode"
fi
