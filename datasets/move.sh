!# ! This file is used to transfer images to NVIDIA Jetson Orin for evaluation.

# 定义本地和远程目录
LOCAL_DIR="/home/yeli/workspace/lard/lard-detection/datasets/lard/detection/test_real/images"
REMOTE_DIR="/home/yeli/test"
REMOTE_USER="yeli"
REMOTE_HOST="192.168.202.28"

# 创建临时文件列表
TEMP_FILE_LIST=$(mktemp)

# 找到所有符号链接指向的实际文件路径并保存到临时文件列表
find "$LOCAL_DIR" -type l -exec readlink -f {} \; > "$TEMP_FILE_LIST"

# 使用 scp 将实际文件复制到远程服务器
while read -r FILE; do
  scp "$FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
done < "$TEMP_FILE_LIST"

# 删除临时文件列表
rm "$TEMP_FILE_LIST"