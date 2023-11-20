import requests


def send_info(exp_name):
    token = '7be5c5ddf7f047fea75c9bb8c589d42f'
    title = '训练完成'
    content = f'{exp_name}训练完成'
    url = 'http://www.pushplus.plus/send?token=' + \
        token+'&title='+title+'&content='+content
    requests.get(url)


if __name__ == '__main__':
    send_info("Yolov8n-p2")
