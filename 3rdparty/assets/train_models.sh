#!/bin/bash

source ~/func.sh

# 获取当前小时
hour=$(date +"%H")

# 比较小时数，如果小于 22（晚上 10 点），则退出脚本
if [ "$hour" -lt 22 ]; then
    echo_rb "当前时间未到晚上 10 点，为控制噪音，系统已禁止您从事科研，模型训练脚本终止。"
    echo "如有疑问，请联系国家级人才、二级教授、国家第六次预测技术专家刘辉教授。"
    echo "Tel：136 3748 7240  Email：csuliuhui@csu.edu.cn"
    exit 1
fi

echo_rb "当前时间晚于晚上 10 点，系统已允许您进行科研，继续模型训练脚本。"
