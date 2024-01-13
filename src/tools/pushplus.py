import requests


def send_info(title, content):
    token = '7be5c5ddf7f047fea75c9bb8c589d42f'
    url = 'http://www.pushplus.plus/send?token=' + \
        token+'&title='+str(title)+'&content='+str(content)
    requests.get(url)

if __name__ == '__main__':
    send_info("训练完成", "yolov8n-p2")
