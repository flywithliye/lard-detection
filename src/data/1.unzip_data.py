from concurrent.futures import ThreadPoolExecutor
import zipfile
import os
from tqdm import tqdm
from typing import List

ROOT_DATA = os.environ.get('LARD_DATA_ROOT_DATA')


def unzip_single_file(zip_path, dest_folder):
    '''
    从zip文件中解压缩单个文件。

    参数:
    zip_path (str): zip文件的路径。
    dest_folder (str): 目标文件夹。

    返回:
    str: 操作的结果。
    '''
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        return f"{zip_path} 解压成功"
    except Exception as e:
        return f"{zip_path} 解压失败: {str(e)}"


def unzip_all_files_in_folder(
        source_folder: str,
        dest_folder: str,
        max_workers: int = 4) -> List[str]:
    """
    并行解压指定文件夹中的所有 .zip 文件到目标文件夹。

    参数:
        search_folder (str): 要搜索 .zip 文件的文件夹路径。
        dest_folder (str): 解压文件应存储到的目标文件夹。
        max_workers (int): 并行解压的最大线程数（默认为 4)。

    返回:
        List[str]: 每个 .zip 文件的解压结果（成功或失败）。
    """
    # 如果目标文件夹不存在，则创建它
    os.makedirs(dest_folder, exist_ok=True)

    # 获取所有 .zip 文件
    zip_files = [f for f in os.listdir(source_folder) if f.endswith('.zip')]
    zip_files = [os.path.join(source_folder, f) for f in zip_files]  # 获取完整路径

    # 使用 ThreadPoolExecutor 进行并行解压
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(lambda x: unzip_single_file(
            x, dest_folder), zip_files), total=len(zip_files)))

    return results


# 解压训练数据
unzip_all_files_in_folder(
    source_folder=f"{ROOT_DATA}/LARD_dataset/LARD_train",
    dest_folder=f"{ROOT_DATA}/LARD_dataset_decompressed/LARD_train",
    max_workers=12)

# 解压测试数据
unzip_all_files_in_folder(
    source_folder=f"{ROOT_DATA}/LARD_dataset/LARD_test",
    dest_folder=f"{ROOT_DATA}/LARD_dataset_decompressed/LARD_test",
    max_workers=2)
