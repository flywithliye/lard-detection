import cv2
from tqdm import tqdm
import os

ROOT_DATA = os.environ.get('LARD_DATA_ROOT_PATH')
ROOT_PROJECT = os.environ.get('LARD_PROJECT_ROOT_PATH')

print(ROOT_DATA)
print(ROOT_PROJECT)


def trim_video(video_path, start_time, end_time, output_path):
    '''
    function for cutting video files 裁剪视频文件
    参数:
    - video_path: path to video 视频文件的路径
    - start_time: start time (s) 裁剪开始时间（秒）
    - end_time: end time (s) 裁剪结束时间（秒）
    - output_path: output path 输出视频文件的路径
    '''

    # check for input video 
    # 检查输入视频文件是否存在
    if not os.path.exists(video_path):
        print(f'Can not find 找不到输入文件：{video_path}')
        return

    # Get video 
    # 初始化视频捕捉
    cap = cv2.VideoCapture(video_path)

    # Get frame rate
    # 获取视频的帧率
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter
    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate,
                          (int(cap.get(3)), int(cap.get(4))))

    # Init params
    # 初始化计数器和时间限制
    count = 0
    start_frame = start_time * frame_rate
    end_frame = end_time * frame_rate

    # Init tqdm 初始化进度条
    with tqdm(total=end_frame - start_frame, desc='Processing 裁剪进度', ncols=100) as pbar:
        # Read and write 读取并写入帧
        while cap.isOpened() and count <= end_frame:
            ret, frame = cap.read()
            if ret:
                if count >= start_frame:
                    out.write(frame)
                    pbar.update(1)  # Update tqdm 更新进度条
                count += 1
            else:
                break

    # Release
    # 释放视频捕捉和写入对象
    cap.release()
    out.release()

    print(f'The video between {start_time} s to {end_time} s has been saved to {output_path}')


# call func
trim_video(
    video_path=f'{ROOT_PROJECT}/datasets/video/orin/F910EDEDB9DE5D133E0A6D5B9E89DB5F.MP4',
    start_time=0,
    end_time=8,
    output_path=f'{ROOT_PROJECT}/datasets/video/landing_1.mp4')
