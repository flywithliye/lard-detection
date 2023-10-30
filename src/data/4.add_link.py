import os

ROOT_PATH = os.environ.get('LARD_DATA_ROOT_PATH')
ROOT_PROJECT = os.environ.get('LARD_YOLO_ROOT_PATH')

if not os.path.exists(f'{ROOT_PROJECT}/datasets/lard/annotations'):
    os.symlink(f'{ROOT_PATH}/annotations',
               f'{ROOT_PROJECT}/datasets/lard/annotations')

if not os.path.exists(f'{ROOT_PROJECT}/datasets/lard/YoloFormat'):
    os.symlink(f'{ROOT_PATH}/YoloFormat',
               f'{ROOT_PROJECT}/datasets/lard/YoloFormat')
