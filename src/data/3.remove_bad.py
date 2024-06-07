import os
import shutil

ROOT_DATA = os.environ.get('LARD_DATA_ROOT_PATH')
ROOT_PROJECT = os.environ.get('LARD_PROJECT_ROOT_PATH')

print(ROOT_DATA)
print(ROOT_PROJECT)

path_train = f"{ROOT_DATA}/YoloFormat/detection/train_all"

# 执行本代码后，eda中的相关分析代码将无法发现以下异常样本
bad_files = ['VABB_32_500_448', 'LWSK_34_500_132']
bad_files_images = [f"{path_train}/images/{file}.jpeg" for file in bad_files]
bad_files_labels = [f"{path_train}/labels/{file}.txt" for file in bad_files]

print(bad_files_images)
print(bad_files_labels)

destination_path = f"{ROOT_DATA}/YoloFormat/detection/bad_files"
os.makedirs(destination_path, exist_ok=True)

for image, label in zip(bad_files_images, bad_files_labels):
    if os.path.exists(image):
        shutil.move(src=image, dst=destination_path)
    if os.path.exists(label):
        shutil.move(src=label, dst=destination_path)