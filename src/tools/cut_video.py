# 导入所需的库
import cv2
from tqdm import tqdm
import os


def trim_video(video_path, start_time, end_time, output_path):
    """
    裁剪视频文件
    参数:
    - video_path: 视频文件的路径
    - start_time: 裁剪开始时间（秒）
    - end_time: 裁剪结束时间（秒）
    - output_path: 输出视频文件的路径
    """

    # 检查输入视频文件是否存在
    if not os.path.exists(video_path):
        print(f"找不到输入文件：{video_path}")
        return

    # 初始化视频捕捉
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate,
                          (int(cap.get(3)), int(cap.get(4))))

    # 初始化计数器和时间限制
    count = 0
    start_frame = start_time * frame_rate
    end_frame = end_time * frame_rate

    # 初始化进度条
    with tqdm(total=end_frame - start_frame, desc="裁剪进度", ncols=100) as pbar:
        # 读取并写入帧
        while cap.isOpened() and count <= end_frame:
            ret, frame = cap.read()
            if ret:
                if count >= start_frame:
                    out.write(frame)
                    pbar.update(1)  # 更新进度条
                count += 1
            else:
                break

    # 释放视频捕捉和写入对象
    cap.release()
    out.release()

    print(f"{start_time}秒到{end_time}秒的视频剪辑已保存为 '{output_path}'")


# 调用函数
trim_video(
    video_path="orin/F910EDEDB9DE5D133E0A6D5B9E89DB5F.MP4",
    start_time=0,
    end_time=8,
    output_path='landing_1.mp4')
