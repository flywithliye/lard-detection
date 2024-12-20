import os
import random


ROOT_DATA = os.environ.get('LARD_DATA_ROOT_PATH')

# 1.data set split
# 1. 数据集划分
# set seeds
# 设置随机数生成器的种子
random.seed(0)

# set paths
# 设置路径
src_image_dir = os.path.abspath(f'{ROOT_DATA}/YoloFormat/detection/train_all/images')
src_label_dir = os.path.abspath(f'{ROOT_DATA}/YoloFormat/detection/train_all/labels')
dst_train_image_dir = os.path.abspath(f'{ROOT_DATA}/YoloFormat/detection/train/images')
dst_train_label_dir = os.path.abspath(f'{ROOT_DATA}/YoloFormat/detection/train/labels')
dst_val_image_dir = os.path.abspath(f'{ROOT_DATA}/YoloFormat/detection/val/images')
dst_val_label_dir = os.path.abspath(f'{ROOT_DATA}/YoloFormat/detection/val/labels')

# create folders
# 创建目录
os.makedirs(dst_train_image_dir, exist_ok=True)
os.makedirs(dst_train_label_dir, exist_ok=True)
os.makedirs(dst_val_image_dir, exist_ok=True)
os.makedirs(dst_val_label_dir, exist_ok=True)

# get all filenames (without extensions)
# 获取所有文件名（不包括扩展名）
filenames = [os.path.splitext(f)[0] for f in os.listdir(
    src_image_dir) if os.path.isfile(os.path.join(src_image_dir, f))]

# shuffle
# 随机洗牌
random.shuffle(filenames)

# split train and validation, 80% for training, 20% for validation
# Note that, this is the validation set, a part of the original training set.
# 划分训练集和验证集，例如，将80%的数据用于训练，20%的数据用于验证
ratio_train = 0.8
train_filenames = filenames[:int(ratio_train * len(filenames))]
val_filenames = filenames[int(ratio_train * len(filenames)):]

print(f'Training set: {len(train_filenames)}, Validation set:{len(val_filenames)}')

# print percentages 打印训练集和验证集的百分比
print(f'Training: {len(train_filenames) / len(filenames):.2%}')
print(f'Validation: {len(val_filenames) / len(filenames):.2%}')


# 2. create dataset
# 2. 数据集构建
# create soft lint to new folder
# 创建软连接到新的目录
for name in train_filenames:
    os.symlink(os.path.join(src_image_dir, f'{name}.jpeg'), os.path.join(
        dst_train_image_dir, f'{name}.jpeg'))
    os.symlink(os.path.join(src_label_dir, f'{name}.txt'), os.path.join(
        dst_train_label_dir, f'{name}.txt'))

for name in val_filenames:
    os.symlink(os.path.join(src_image_dir, f'{name}.jpeg'), os.path.join(
        dst_val_image_dir, f'{name}.jpeg'))
    os.symlink(os.path.join(src_label_dir, f'{name}.txt'), os.path.join(
        dst_val_label_dir, f'{name}.txt'))


# 3. do some statistic
# 3. 新数据集统计
def count_files_in_folder(folder_path):
    return len(os.listdir(folder_path))


# get train/val sample numbers
# 统计各新train/val样本数量
train_image_count = count_files_in_folder(dst_train_image_dir)
train_label_count = count_files_in_folder(dst_train_label_dir)
val_image_count = count_files_in_folder(dst_val_image_dir)
val_label_count = count_files_in_folder(dst_val_label_dir)

print(f"Train image count: {train_image_count}")
print(f"Train label count: {train_label_count}")
print(f"Validation image count: {val_image_count}")
print(f"Validation label count: {val_label_count}")
