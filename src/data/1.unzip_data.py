from concurrent.futures import ThreadPoolExecutor
import zipfile
import os
from tqdm import tqdm
from typing import List
from tidecv import TIDE, Data
ROOT_DATA = os.environ.get('LARD_DATA_ROOT_PATH')


def unzip_single_file(zip_path, dest_folder):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        return f"{zip_path} unzip ok"
    except Exception as e:
        return f"{zip_path} unzip fail: {str(e)}"


def unzip_all_files_in_folder(
        source_folder: str,
        dest_folder: str,
        max_workers: int = 4) -> List[str]:

    # create folder
    # 如果目标文件夹不存在，则创建它
    os.makedirs(dest_folder, exist_ok=True)

    # get all zip file
    # 获取所有 .zip 文件
    zip_files = [f for f in os.listdir(source_folder) if f.endswith('.zip')]
    zip_files = [os.path.join(source_folder, f) for f in zip_files]  # get full path 获取完整路径

    # using ThreadPoolExecutor for parallel unzip
    # 使用 ThreadPoolExecutor 进行并行解压
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(lambda x: unzip_single_file(
            x, dest_folder), zip_files), total=len(zip_files)))

    return results


# unzip training data
# 解压训练数据
unzip_all_files_in_folder(
    source_folder=f"{ROOT_DATA}/LARD_dataset/LARD_train",
    dest_folder=f"{ROOT_DATA}/LARD_dataset_decompressed/LARD_train",
    max_workers=12)

# unzip test data
# 解压测试数据
unzip_all_files_in_folder(
    source_folder=f"{ROOT_DATA}/LARD_dataset/LARD_test",
    dest_folder=f"{ROOT_DATA}/LARD_dataset_decompressed/LARD_test",
    max_workers=2)
