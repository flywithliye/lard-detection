#!/bin/bash

# Usage:./exp_yolov8_test.sh <mode>

# import util functions 
# 导入函数
source src/tools/func.sh
source src/tools/pushplus.sh

# Make sure the usage is correct
# 确保脚本参数数量正确
if [ "$#" -ne 1 ]; then
    echo_rb "Usage: $0 <mode>"
    exit 1
fi

# Mode params
# 模式参数
mode=$1
valid_modes=("all" "base" "att" "tf" "fpn" "iou" "aug" "up" "merge" "finetune")
if [[ ! " ${valid_modes[*]} " =~ " ${mode} " ]]; then
    echo_rb "Error: mode must be one of: ${valid_modes[*]}"
    exit 1
fi

read -p "You select model '$mode', are you sure to continue? (y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "cencel"
    exit 0
fi

# Define experiments
# 实验定义
run_base() {
    mode="base"
    echo_rb 'All test processes have started in the background'

    # n_640
    python -u cfg/ultralytics/test.py --mode=$mode > "logs/test/ultra_test_yolov8n_base_640.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 > "logs/test/ultra_test_yolov8n_base_p2_640.log" 2>&1

    # s_640
    python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s > "logs/test/ultra_test_yolov8s_base_640.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --stru=p2 > "logs/test/ultra_test_yolov8s_base_p2_640.log" 2>&1

    # todo v10_n_640
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov10n > logs/test/ultra_test_yolov10n_base_640.log 2>&1

    # 1280
    # python -u cfg/ultralytics/test.py --mode=$mode --size=1280 > "logs/test/ultra_test_yolov8n_base_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --size=1280 > "logs/test/ultra_test_yolov8n_p2_base_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --size=1280 > "logs/test/ultra_test_yolov8s_base_1280.log" 2>&1
        
    send_info "Exp: $mode" 'All training experiments have been completed.'
}

run_att() {
    echo_rb 'All test processes have started in the background'

    # n_640
    mode="att"
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=se > "logs/test/ultra_test_yolov8n_att_se_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=cbam > "logs/test/ultra_test_yolov8n_att_cbam_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=eca > "logs/test/ultra_test_yolov8n_att_eca_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=ese > "logs/test/ultra_test_yolov8n_att_ese_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=gam > "logs/test/ultra_test_yolov8n_att_gam_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=sa > "logs/test/ultra_test_yolov8n_att_sa_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=cpca > "logs/test/ultra_test_yolov8n_att_cpca_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=ema > "logs/test/ultra_test_yolov8n_att_ema_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=ta > "logs/test/ultra_test_yolov8n_att_ta_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=lsk > "logs/test/ultra_test_yolov8n_att_lsk_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=lska > "logs/test/ultra_test_yolov8n_att_lska_640.log" 2>&1
    
    # mode="atts"
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=ses > logs/test/ultra_test_yolov8n_atts_ses_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=cbams > logs/test/ultra_test_yolov8n_atts_cbams_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=ecas > logs/test/ultra_test_yolov8n_atts_ecas_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=eses > logs/test/ultra_test_yolov8n_atts_eses_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=gams > logs/test/ultra_test_yolov8n_atts_gams_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=sas > logs/test/ultra_test_yolov8n_atts_sas_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=cpcas > logs/test/ultra_test_yolov8n_atts_cpcas_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=emas > logs/test/ultra_test_yolov8n_atts_emas_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=tas > logs/test/ultra_test_yolov8n_atts_tas_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=lsks > logs/test/ultra_test_yolov8n_atts_lsks_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=lskas > logs/test/ultra_test_yolov8n_atts_lskas_640.log 2>&1

    # s_640
    # mode="att"
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=se > logs/test/ultra_test_yolov8s_att_se_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=cbam > logs/test/ultra_test_yolov8s_att_cbam_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=eca > logs/test/ultra_test_yolov8s_att_eca_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=ese > logs/test/ultra_test_yolov8s_att_ese_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=gam > logs/test/ultra_test_yolov8s_att_gam_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=sa > logs/test/ultra_test_yolov8s_att_sa_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=cpca > logs/test/ultra_test_yolov8s_att_cpca_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=ema > logs/test/ultra_test_yolov8s_att_ema_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=ta > logs/test/ultra_test_yolov8s_att_ta_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=lsk > logs/test/ultra_test_yolov8s_att_lsk_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=lska > logs/test/ultra_test_yolov8s_att_lska_640.log 2>&1

    # n_1280
    # mode="att"
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=se --size=1280 > "logs/test/ultra_test_yolov8n_att_se_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=cbam --size=1280 > "logs/test/ultra_test_yolov8n_att_cbam_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=eca --size=1280 > "logs/test/ultra_test_yolov8n_att_eca_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=ese --size=1280 > "logs/test/ultra_test_yolov8n_att_ese_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=gam --size=1280 > "logs/test/ultra_test_yolov8n_att_gam_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=sa --size=1280 > "logs/test/ultra_test_yolov8n_att_sa_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=cpca --size=1280 > "logs/test/ultra_test_yolov8n_att_cpca_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=ema --size=1280 > "logs/test/ultra_test_yolov8n_att_ema_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=ta --size=1280 > "logs/test/ultra_test_yolov8n_att_ta_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=lsk --size=1280 > "logs/test/ultra_test_yolov8n_att_lsk_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=lska --size=1280 > "logs/test/ultra_test_yolov8n_att_lska_1280.log" 2>&1
    
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=se --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_se_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=cbam --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_cbam_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=eca --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_eca_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=ese --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_ese_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=gam --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_gam_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=sa --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_sa_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=cpca --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_cpca_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=ema --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_ema_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=ta --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_ta_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_lsk_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lska --size=1280 > "logs/test/ultra_test_yolov8n_p2_att_lska_1280.log" 2>&1

    send_info "Exp: $mode" 'All training experiments have been completed.'
}

run_tf() {
    echo_rb "All test processes have started in the background"

    # n_640
    mode="tf"
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=vit > "logs/test/ultra_test_yolov8n_tf_vit_640.log" 2>&1

    # s_640
    # mode="tf"
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=vit > logs/test/ultra_test_yolov8s_tf_vit_640.log 2>&1

    # n_1280
    # mode="tf"
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=vit --size=1280 > "logs/test/ultra_test_yolov8n_tf_vit_1280.log" 2>&1

    # p2_n_1280
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=vit --size=1280 > "logs/test/ultra_test_yolov8n_p2_tf_vit_1280.log" 2>&1

    send_info "Exp: $mode" 'All training experiments have been completed.'
}

run_fpn() {
    mode="fpn"
    echo_rb 'All test processes have started in the background'

    # n_640
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=bifpn > "logs/test/ultra_test_yolov8n_fpn_bifpn_640.log" 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=afpn > "logs/test/ultra_test_yolov8n_fpn_afpn_640.log" 2>&1

    # s_640
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=bifpn > logs/test/ultra_test_yolov8s_fpn_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=afpn > logs/test/ultra_test_yolov8s_fpn_afpn_640.log 2>&1
    
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=bifpn > "logs/test/ultra_test_yolov8n_p2_fpn_bifpn.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=afpn > "logs/test/ultra_test_yolov8n_p2_fpn_afpn.log" 2>&1
    
    # 1280
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=bifpn --size=1280 > "logs/test/ultra_test_yolov8n_fpn_bifpn_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=afpn --size=1280 > "logs/test/ultra_test_yolov8n_fpn_afpn_1280.log" 2>&1
    
    # p2-1280
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=afpn --size=1280 > "logs/test/ultra_test_yolov8n_p2_fpn_afpn_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=bifpn --size=1280 > "logs/test/ultra_test_yolov8n_p2_fpn_bifpn_1280.log" 2>&1
    
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=se_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_se_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=cbam_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_cbam_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=eca_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_eca_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=ese_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_ese_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=gam_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_gam_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=sa_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_sa_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=cpca_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_cpca_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=ema_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_ema_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=ta_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_ta_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_lsk_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lska_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_lska_fpn_bifpn_1280.log 2>&1

    # c2f_att
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=c2f_se_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_c2f_se_fpn_bifpn_1280.log 2>&1

    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=carafe_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_up_carafe_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=se_carafe_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_se_up_carafe_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=c2f_eca_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_c2f_eca_fpn_bifpn_1280.log 2>&1

    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=c2f_se_carafe_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_att_c2f_se_up_carafe_fpn_bifpn_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=dysample_bifpn --size=1280 > logs/test/ultra_test_yolov8n_p2_up_dysample_fpn_bifpn_1280.log 2>&1

    send_info "Exp: $mode" 'All training experiments have been completed.'
}

run_iou() { 
    mode="iou"
    echo_rb 'All test processes have started in the background'

    # n_640
    python -u cfg/ultralytics/test.py --mode=$mode --iou_type=GIoU > logs/test/ultra_test_yolov8n_iou_giou_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --iou_type=DIoU > logs/test/ultra_test_yolov8n_iou_diou_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --iou_type=SIoU > logs/test/ultra_test_yolov8n_iou_siou_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --iou_type=EIoU > logs/test/ultra_test_yolov8n_iou_eiou_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --iou_type=WIoU > logs/test/ultra_test_yolov8n_iou_wiou_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --iou_type=MDPIoU1 > logs/test/ultra_test_yolov8n_iou_mdpiou1_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --iou_type=MDPIoU2 > logs/test/ultra_test_yolov8n_iou_mdpiou2_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --iou_type=ShapeIoU > logs/test/ultra_test_yolov8n_iou_shapeiou_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --iou_type=NWD > logs/test/ultra_test_yolov8n_iou_nwd_640.log 2>&1

    # s_640
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --iou_type=GIoU > logs/test/ultra_test_yolov8s_iou_giou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --iou_type=DIoU > logs/test/ultra_test_yolov8s_iou_diou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --iou_type=SIoU > logs/test/ultra_test_yolov8s_iou_siou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --iou_type=EIoU > logs/test/ultra_test_yolov8s_iou_eiou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --iou_type=WIoU > logs/test/ultra_test_yolov8s_iou_wiou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --iou_type=MDPIoU1 > logs/test/ultra_test_yolov8s_iou_mdpiou1_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --iou_type=MDPIoU2 > logs/test/ultra_test_yolov8s_iou_mdpiou2_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --iou_type=ShapeIoU > logs/test/ultra_test_yolov8s_iou_shapeiou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --iou_type=NWD > logs/test/ultra_test_yolov8s_iou_nwd_640.log 2>&1
    
    # 1280
    # python -u cfg/ultralytics/test.py --mode=$mode --iou_type=GIoU --size=1280 > "logs/test/ultra_test_yolov8n_iou_giou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --iou_type=DIoU --size=1280 > "logs/test/ultra_test_yolov8n_iou_diou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --iou_type=SIoU --size=1280 > "logs/test/ultra_test_yolov8n_iou_siou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --iou_type=EIoU --size=1280 > "logs/test/ultra_test_yolov8n_iou_eiou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --iou_type=WIoU --size=1280 > "logs/test/ultra_test_yolov8n_iou_wiou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --iou_type=MDPIoU1 --size=1280 > "logs/test/ultra_test_yolov8n_iou_mdpiou1_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --iou_type=MDPIoU2 --size=1280 > "logs/test/ultra_test_yolov8n_iou_mdpiou2_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --iou_type=ShapeIoU --size=1280 > "logs/test/ultra_test_yolov8n_iou_shapeiou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --iou_type=NWD --size=1280 > "logs/test/ultra_test_yolov8n_iou_nwd_1280.log" 2>&1

    send_info "Exp: $mode" 'All training experiments have been completed.'
}

run_up() {
    mode="up"
    echo_rb 'All test processes have started in the background'

    # n_640
    python -u cfg/ultralytics/test.py --mode=$mode --cfg=carafe > "logs/test/ultra_test_yolov8n_up_carafe_640.log" 2>&1

    # s_640
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --cfg=carafe > logs/test/ultra_test_yolov8s_up_carafe_640.log 2>&1

    # 1280
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=carafe --size=1280  > "logs/test/ultra_test_yolov8n_up_carafe_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=carafe --size=1280  > "logs/test/ultra_test_yolov8n_p2_up_carafe_1280.log" 2>&1
    
    send_info "Exp: $mode" 'All training experiments have been completed.'
}

run_aug() {
    mode="aug"
    echo_rb 'All test processes have started in the background'

    # n_640
    python -u cfg/ultralytics/test.py --mode=$mode --aug_json=all --album=0.01 > logs/test/ultra_test_yolov8n_aug_all_01_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --aug_json=all --album=0.05 > logs/test/ultra_test_yolov8n_aug_all_05_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --aug_json=all --album=0.10 > logs/test/ultra_test_yolov8n_aug_all_10_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --aug_json=all --album=0.15 > logs/test/ultra_test_yolov8n_aug_all_15_640.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --aug_json=all --album=0.20 > logs/test/ultra_test_yolov8n_aug_all_20_640.log 2>&1
    
    # s_640
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --aug_json=all --album=0.01 > logs/test/ultra_test_yolov8s_aug_all_01_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --aug_json=all --album=0.05 > logs/test/ultra_test_yolov8s_aug_all_05_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --aug_json=all --album=0.10 > logs/test/ultra_test_yolov8s_aug_all_10_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --aug_json=all --album=0.10 > logs/test/ultra_test_yolov8s_aug_all_10_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --model=yolov8s --aug_json=all --album=0.20 > logs/test/ultra_test_yolov8s_aug_all_20_640.log 2>&1

    # 1280
    # python -u cfg/ultralytics/test.py --mode=$mode --album=0.05 --size=1280 > "logs/test/ultra_test_yolov8n_aug05_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --album=0.1 --size=1280 > "logs/test/ultra_test_yolov8n_aug10_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --album=0.2 --size=1280 > "logs/test/ultra_test_yolov8n_aug20_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --album=0.4 --size=1280 > "logs/test/ultra_test_yolov8n_aug40_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --album=0.6 --size=1280 > "logs/test/ultra_test_yolov8n_aug60_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --album=0.8 --size=1280 > "logs/test/ultra_test_yolov8n_aug80_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --album=1.0 --size=1280 > "logs/test/ultra_test_yolov8n_aug100_1280.log" 2>&1
    
    # p2-1280
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --aug_json=all --album=0.01 --size=1280 > logs/test/ultra_test_yolov8n_p2_aug_all_01_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --aug_json=all --album=0.05 --size=1280 > logs/test/ultra_test_yolov8n_p2_aug_all_05_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --aug_json=all --album=0.10 --size=1280 > logs/test/ultra_test_yolov8n_p2_aug_all_10_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --aug_json=all --album=0.15 --size=1280 > logs/test/ultra_test_yolov8n_p2_aug_all_15_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --aug_json=all --album=0.20 --size=1280 > logs/test/ultra_test_yolov8n_p2_aug_all_20_1280.log 2>&1

    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --aug_json=color --album=0.10 --size=1280 > logs/test/ultra_test_yolov8n_p2_aug_color_10_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --aug_json=color_blur --album=0.10 --size=1280 > logs/test/ultra_test_yolov8n_p2_aug_color_blur_10_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --aug_json=color_blur_geo --album=0.10 --size=1280 > logs/test/ultra_test_yolov8n_p2_aug_color_blur_geo_10_1280.log 2>&1
    
    send_info "Exp: $mode" 'All training experiments have been completed.'
}

run_merge() {
    mode="merge"
    echo_rb 'All test processes have started in the background'

    ## 640
    # ATT + FPN
    # merge_mode="att_fpn"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=cpca_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_cpca_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=ese_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_ese_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=ta_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_ta_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_lska_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=sa_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_sa_bifpn_640.log 2>&1
    
    # ATTs + FPN
    # merge_mode="atts_fpn"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=emas_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_emas_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=cbams_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_cbams_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=gams_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_gams_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_tas_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=eses_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_eses_bifpn_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=ses_bifpn > logs/test/ultra_test_yolov8n_merge_att_fpn_ses_bifpn_640.log 2>&1
 
    # ATT + IOU
    # merge_mode="att_iou"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska --iou_type=EIoU > logs/test/ultra_test_yolov8n_merge_att_iou_lska_eiou_640.log 2>&1

    # ATT + AUG
    # merge_mode="att_aug"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska --aug_json=all --album=0.10 > logs/test/ultra_test_yolov8n_merge_att_aug_lska_aug_all_10_640.log 2>&1
    
    # FPN + IOU
    # merge_mode="fpn_iou"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=bifpn --iou_type=EIoU > logs/test/ultra_test_yolov8n_merge_fpn_iou_bifpn_eiou_640.log 2>&1

    # FPN + AUG
    # merge_mode="fpn_aug"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=bifpn --aug_json=all --album=0.10 > logs/test/ultra_test_yolov8n_merge_fpn_aug_bifpn_aug_all_10_640.log 2>&1

    # ATT + FPN + IOU
    # merge_mode="att_fpn_iou"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=MDPIoU1 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_lska_bifpn_mdpiou1_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=GIoU > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_lska_bifpn_giou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=MDPIoU2 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_lska_bifpn_mdpiou2_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=DIoU > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_lska_bifpn_diou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=SIoU > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_lska_bifpn_siou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_lska_bifpn_eiou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=WIoU > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_lska_bifpn_wiou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=ShapeIoU > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_lska_bifpn_shapeiou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=NWD > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_lska_bifpn_nwd_640.log 2>&1

    # merge_mode="atts_fpn_iou"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=MDPIoU1 > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_tas_bifpn_mdpiou1_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=GIoU > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_tas_bifpn_giou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=MDPIoU2 > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_tas_bifpn_mdpiou2_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=DIoU > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_tas_bifpn_diou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=SIoU > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_tas_bifpn_siou_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=ShapeIoU > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_tas_bifpn_shapeiou_640.log 2>&1

    # ATT + FPN + AUG
    # merge_mode="att_fpn_aug"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --aug_json=all --album=0.01 > logs/test/ultra_test_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_01_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --aug_json=all --album=0.05 > logs/test/ultra_test_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_05_640.log 2>&1

    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --aug_json=all --album=0.10 > logs/test/ultra_test_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_10_640_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --aug_json=all --album=0.10 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_10_640_1280.log 2>&1
    
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --aug_json=all --album=0.15 > logs/test/ultra_test_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_15_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --aug_json=all --album=0.20 > logs/test/ultra_test_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_20_640.log 2>&1

    # ATT + FPN + UP + AUG
    # merge_mode="att_fpn_up_aug"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn_carafe --aug_json=all --album=0.10 > logs/test/ultra_test_yolov8n_merge_att_fpn_up_aug_lska_bifpn_carafe_aug_all_10_640_640.log 2>&1

    # ATT + FPN + IOU + UP
    # merge_mode="atts_fpn_up_iou"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn_carafe --iou_type=ShapeIoU > logs/test/ultra_test_yolov8n_merge_atts_fpn_up_iou_tas_bifpn_carafe_shapeiou_640.log 2>&1
    
    # ATT + FPN + IOU + AUG
    merge_mode="att_fpn_iou_aug"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.01 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_01_640_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.01 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_01_640_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.05 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_05_640_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.05 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_05_640_1280.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.05 --half_fp16 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_05_640_640_fp16.log 2>&1
    python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.05 --test_size=1280 --half_fp16 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_05_640_1280_fp16.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.10 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_10_640_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.10 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_10_640_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.20 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_20_640_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.20 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_20_640_1280.log 2>&1

    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=ShapeIoU --aug_json=all --album=0.01 > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_aug_tas_bifpn_shapeiou_aug_all_01_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=ShapeIoU --aug_json=all --album=0.05 > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_aug_tas_bifpn_shapeiou_aug_all_05_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=ShapeIoU --aug_json=all --album=0.10 > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_aug_tas_bifpn_shapeiou_aug_all_10_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=ShapeIoU --aug_json=all --album=0.15 > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_aug_tas_bifpn_shapeiou_aug_all_15_640.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=tas_bifpn --iou_type=ShapeIoU --aug_json=all --album=0.20 > logs/test/ultra_test_yolov8n_merge_atts_fpn_iou_aug_tas_bifpn_shapeiou_aug_all_20_640.log 2>&1

    ## 1280
    # ATT + FPN + AUG
    # merge_mode="att_fpn_aug"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --aug_json=all --album=0.10 --size=1280 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_10_1280_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --aug_json=all --album=0.20 --size=1280 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_aug_lska_bifpn_aug_all_20_1280_1280.log 2>&1

    # ATT + FPN + IOU + AUG
    # merge_mode="att_fpn_iou_aug"
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.01 --size=1280 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_01_1280_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.05 --size=1280 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_05_1280_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.10 --size=1280 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_10_1280_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --merge_mode=$merge_mode --cfg=lska_bifpn --iou_type=EIoU --aug_json=all --album=0.20 --size=1280 --test_size=1280 > logs/test/ultra_test_yolov8n_merge_att_fpn_iou_aug_lska_bifpn_eiou_aug_all_20_1280_1280.log 2>&1
    
    ## 1280
    # template python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lska --iou_type=MDPIoU2 --album=0.05 --size=1280 > "logs/test/merge/ultra_test_yolov8n_merge7_1.log" 2>&1
    
    # # 2
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=bifpn --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_bifpn_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=afpn --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_afpn_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_lsk_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --iou_type=DIoU --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_diou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=lsk_bifpn --size=1280 > "logs/test/ultra_test_yolov8n_merge_lsk_bifpn_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=bifpn --iou_type=DIoU --size=1280 > "logs/test/ultra_test_yolov8n_merge_bifpn_diou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=lsk --iou_type=DIoU --size=1280 > "logs/test/ultra_test_yolov8n_merge_lsk_diou_1280.log" 2>&1
    
    # # 3
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_lsk_bifpn_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=bifpn --iou_type=DIoU --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_bifpn_diou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk --iou_type=DIoU --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_lsk_diou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --cfg=lsk_bifpn --iou_type=DIoU --size=1280 > "logs/test/ultra_test_yolov8n_merge_lsk_bifpn_diou_1280.log" 2>&1

    # # 4
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_lsk_bifpn_diou_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.01 --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_lsk_bifpn_diou_aug01_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.05 --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_lsk_bifpn_diou_aug05_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.10 --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_lsk_bifpn_diou_aug10_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.15 --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_lsk_bifpn_diou_aug15_1280.log" 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn --iou_type=DIoU --album=0.20 --size=1280 > "logs/test/ultra_test_yolov8n_merge_p2_lsk_bifpn_diou_aug20_1280.log" 2>&1

    # # 5
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn_carafe --iou_type=DIoU --album=0.05 --size=1280 > logs/test/ultra_test_yolov8n_merge_p2_lsk_bifpn_carafe_diou_aug05_1280.log 2>&1
    # python -u cfg/ultralytics/test.py --mode=$mode --stru=p2 --cfg=lsk_bifpn_carafe --iou_type=DIoU --album=0.10 --size=1280 > logs/test/ultra_test_yolov8n_merge_p2_lsk_bifpn_carafe_diou_aug10_1280.log 2>&1

    send_info "Exp: $mode" 'All training experiments have been completed.'
}

run_finetune() {
    mode="finetune"
    echo_rb 'All test processes have started in the background'

    # bars
    python -u cfg/ultralytics/test_finetune.py \
        --mode=$mode \
        --finetune_mode=bars_runway_val_test \
        --cfg=lska_bifpn \
        --iou_type=EIoU \
        --aug_json=all \
        --album=0.10 > logs/test/ultra_test_yolov8n_finetune_bars_runway_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # msfs
    python -u cfg/ultralytics/test_finetune.py \
        --mode=$mode \
        --finetune_mode=fs2020_runway_test \
        --cfg=lska_bifpn \
        --iou_type=EIoU \
        --aug_json=all \
        --album=0.10 > logs/test/ultra_test_yolov8n_finetune_fs2020_runway_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # !5%
    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.05 > logs/test/ultra_test_yolov8n_finetune_single_lska_bifpn_eiou_aug_all_05_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.05 > logs/test/ultra_test_yolov8n_finetune_double_lska_bifpn_eiou_aug_all_05_640.log 2>&1
        
    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.05 > logs/test/ultra_test_yolov8n_finetune_triple_lska_bifpn_eiou_aug_all_05_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.05 > logs/test/ultra_test_yolov8n_finetune_triple_split_lska_bifpn_eiou_aug_all_05_640.log 2>&1

    # # !10%
    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.10 > logs/test/ultra_test_yolov8n_finetune_single_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.10 > logs/test/ultra_test_yolov8n_finetune_double_lska_bifpn_eiou_aug_all_10_640.log 2>&1
        
    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.10 > logs/test/ultra_test_yolov8n_finetune_triple_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.10 > logs/test/ultra_test_yolov8n_finetune_triple_split_lska_bifpn_eiou_aug_all_10_640.log 2>&1

    # # !20%
    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.20 > logs/test/ultra_test_yolov8n_finetune_single_lska_bifpn_eiou_aug_all_20_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.20 > logs/test/ultra_test_yolov8n_finetune_double_lska_bifpn_eiou_aug_all_20_640.log 2>&1
        
    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.20 > logs/test/ultra_test_yolov8n_finetune_triple_lska_bifpn_eiou_aug_all_20_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --cfg=lska_bifpn \
    #     --iou_type=EIoU \
    #     --aug_json=all \
    #     --album=0.20 > logs/test/ultra_test_yolov8n_finetune_triple_split_lska_bifpn_eiou_aug_all_20_640.log 2>&1


    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --cfg=lska_bifpn \
    #     --aug_json=all \
    #     --album=0.10 > logs/test/ultra_test_yolov8n_finetune_single_lska_bifpn_aug_all_10_640_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --cfg=lska_bifpn \
    #     --aug_json=all \
    #     --album=0.10 > logs/test/ultra_test_yolov8n_finetune_double_lska_bifpn_aug_all_10_640_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --cfg=lska_bifpn \
    #     --aug_json=all \
    #     --album=0.10 > logs/test/ultra_test_yolov8n_finetune_triple_lska_bifpn_aug_all_10_640_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --cfg=lska_bifpn \
    #     --aug_json=all \
    #     --album=0.10 > logs/test/ultra_test_yolov8n_finetune_triple_split_lska_bifpn_aug_all_10_640_640.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=single \
    #     --stru=p2 \
    #     --cfg=lsk_bifpn \
    #     --iou_type=DIoU \
    #     --album=0.1 \
    #     --size=1280 > logs/test/ultra_test_last_yolov8n_finetune_single_p2_lsk_bifpn_diou_aug10_1280.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=double \
    #     --stru=p2 \
    #     --cfg=lsk_bifpn \
    #     --iou_type=DIoU \
    #     --album=0.1 \
    #     --size=1280 > logs/test/ultra_test_last_yolov8n_finetune_double_nominal_p2_lsk_bifpn_diou_aug10_1280.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple \
    #     --stru=p2 \
    #     --cfg=lsk_bifpn \
    #     --iou_type=DIoU \
    #     --album=0.1 \
    #     --size=1280 > logs/test/ultra_test_last_yolov8n_finetune_triple_p2_lsk_bifpn_diou_aug10_1280.log 2>&1

    # python -u cfg/ultralytics/test_finetune.py \
    #     --mode=$mode \
    #     --finetune_mode=triple_split \
    #     --stru=p2 \
    #     --cfg=lsk_bifpn \
    #     --iou_type=DIoU \
    #     --album=0.1 \
    #     --size=1280 > logs/test/ultra_test_last_yolov8n_finetune_triple_split_p2_lsk_bifpn_diou_aug10_1280.log 2>&1

    send_info "Exp: $mode" 'All training experiments have been completed.'
}

# Conduct different experiments based on parameter `mode`.
# 依据参数开展不同实验

# 1. All exp
if [ $mode == "all" ]; then
(
    run_base
    run_att
    run_tf
    run_fpn
    run_iou
    run_aug
    run_up
    run_merge
    run_finetune
    send_info "实验: 全部test_mode" 'All training experiments have been completed.'
) &

# 1. base model
elif [ $mode == "base" ]; then
(
    run_base
) &

# 2. base model + ATT
elif [ $mode == "att" ]; then
(
    run_att
) &

# 3. base model + Transformer
elif [ $mode == "tf" ]; then
(
    run_tf
) &

# 4. base model + fpn
elif [ $mode == "fpn" ]; then
(
    run_fpn
) &

# 5. base model + IOU
elif [ $mode == "iou" ]; then
(
    run_iou
) &

# 6. base model + AUG
elif [ $mode == "aug" ]; then
(
    run_aug
) &

# 7. base model + UP
elif [ $mode == "up" ]; then
(
    run_up
) &

# 8. merge models
elif [ $mode == "merge" ]; then
(
    run_merge
) &

# 9. finetune models
elif [ $mode == "finetune" ]; then
(
    run_finetune
) &

# -1. exception
else
    echo_rb "wrong mode: $mode"
fi
