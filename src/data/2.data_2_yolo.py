import os
from LARD.src.dataset.lard_dataset import LardDataset

# pandas<v2.1.0 
# FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.

ROOT_DATA = os.environ.get('LARD_DATA_ROOT_PATH')
zip_folder = f"{ROOT_DATA}/LARD_dataset_decompressed/"
yolo_folder = f"{ROOT_DATA}/YoloFormat/detection"

# 训练集
dataset = LardDataset(
    train_path=os.path.join(zip_folder, "LARD_train"))
dataset.export(
    output_dir=os.path.join(yolo_folder, "train_all"),
    bbx_format="xywh",  # Options are 'tlbr', 'tlwh', 'xywh', 'corners'
    normalized=True,  # noqa 'multiple' produces 1 file per label, as expected by yolo architectures.
    label_file="multiple",
    crop=True,  # noqa 'True' recommended to remove the watermark. Pay attention to not crop a picture multiple times
    sep=' ',  # Separator in the label file.
    header=False,  # noqa 'False' is recommender for multiple files, 'True' for single files. It adds a header with column names in the first line of the labels file
    ext="txt"
)

# 测试集-合成
dataset = LardDataset(
    test_path=os.path.join(zip_folder, "LARD_test/LARD_test_synth")
)
dataset.export(
    output_dir=os.path.join(yolo_folder, "test_synth"),
    bbx_format="xywh",
    normalized=True,
    label_file="multiple",
    crop=True,
    sep=' ',
    header=False,
    ext="txt"
)

# 测试集-真实-Nominal
# cd ~/workspace/lard/lard-dataset/LARD_dataset_decompressed/LARD_test/LARD_test_real/
# mv LARD_test_real_nominal LARD_test_real_nominal_cases
# cd LARD_test_real_nominal_cases
# mv Test_Real_Nominal.csv LARD_test_real_nominal_cases.csv
dataset = LardDataset(
    test_path=os.path.join(zip_folder, "LARD_test/LARD_test_real/LARD_test_real_nominal_cases"))  # noqa
dataset.export(
    output_dir=os.path.join(yolo_folder, "test_real_nominal"),
    bbx_format="xywh",
    normalized=True,
    label_file="multiple",
    crop=False,
    sep=' ',
    header=False,
    ext="txt"
)

# 测试集-真实-Edge
# cd ~/workspace/lard/lard-dataset/LARD_dataset_decompressed/LARD_test/LARD_test_real/LARD_test_real_edge_cases
# mv Test_Real_Edge_Cases.csv LARD_test_real_edge_cases.csv
dataset = LardDataset(
    train_path=os.path.join(zip_folder, "LARD_test/LARD_test_real/LARD_test_real_edge_cases"))  # noqa
dataset.export(
    output_dir=os.path.join(yolo_folder, "test_real_edge"),
    bbx_format="xywh",
    normalized=True,
    label_file="multiple",
    crop=False,
    sep=' ',
    header=False,
    ext="txt"
)
