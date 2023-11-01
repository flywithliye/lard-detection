import os

ROOT_DATA = os.environ.get('LARD_DATA_ROOT_DATA')
ROOT_PROJECT = os.environ.get('LARD_YOLO_ROOT_DATA')

if not os.path.exists(f'{ROOT_PROJECT}/datasets/lard/annotations'):
    os.symlink(f'{ROOT_DATA}/annotations',
               f'{ROOT_PROJECT}/datasets/lard/annotations')

if not os.path.exists(f'{ROOT_PROJECT}/datasets/lard/YoloFormat'):
    os.symlink(f'{ROOT_DATA}/YoloFormat',
               f'{ROOT_PROJECT}/datasets/lard/YoloFormat')
