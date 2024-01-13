#!/bin/bash

# URL 编码函数
urlencode() {
    # 使用 perl 对字符串进行 URL 编码
    perl -MURI::Escape -e 'print uri_escape($ARGV[0]);' "$1"
}

send_info() {
    # 固定 token 值
    token="7be5c5ddf7f047fea75c9bb8c589d42f"

    # 获取命令行参数
    title=$1    # 第一个参数是标题
    content=$2  # 第二个参数是内容

    # 检查是否提供了所有必要的参数
    if [ -z "$title" ] || [ -z "$content" ]; then
        echo "用法: $0 \"标题\" \"内容\""
        exit 1
    fi

    # 对标题和内容进行URL编码
    encoded_title=$(urlencode "$title")
    encoded_content=$(urlencode "$content")

    # 发送请求
    url="https://www.pushplus.plus/send?token=$token&title=$encoded_title&content=$encoded_content&template=html"
    curl "$url"
}
