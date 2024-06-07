#!/bin/bash

# 确保脚本参数数量正确
if [ "$#" -ne 3 ]; then
    echo "使用方式: $0 <time_train> <time_val> <epoch>"
    exit 1
fi

# 接受两个命令行参数
time_train=$1
time_val=$2
epoch=$3

# 计算总秒数
total_seconds=$(( ($time_train + $time_val) * (300 - $epoch) ))

# 获取当前时间，并添加计算出的秒数
# 使用 date 命令和 -d 选项，在 GNU 环境下运行
# 在 macOS 或其他 BSD 系统中，可能需要使用 -v 选项或安装 gdate
final_time=$(date -d "@$(($(date +%s) + total_seconds))" '+%Y-%m-%d %H:%M:%S')

# 打印最终时间
echo "预计完成时间: $final_time"
