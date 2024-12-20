import os

ROOT_DATA = os.environ.get('LARD_DATA_ROOT_PATH')
ROOT_PROJECT = os.environ.get('LARD_PROJECT_ROOT_PATH')
print(ROOT_DATA)
print(ROOT_PROJECT)

os.makedirs(f'{ROOT_PROJECT}/datasets/lard', exist_ok=True)

if not os.path.exists(f'{ROOT_PROJECT}/datasets/lard/annotations'):
    os.symlink(f'{ROOT_DATA}/annotations',
               f'{ROOT_PROJECT}/datasets/lard/annotations')

if not os.path.exists(f'{ROOT_PROJECT}/datasets/lard/detection'):
    os.symlink(f'{ROOT_DATA}/YoloFormat/detection',
               f'{ROOT_PROJECT}/datasets/lard/detection')

# ln -s /fileonssd/coco-dataset/coco /home/yeli/workspace/lard/lard-detection/datasets/coco
# ln -s cfg/ultralytics/weights/ weights
