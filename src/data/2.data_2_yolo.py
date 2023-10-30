import os
from LARD.src.dataset.lard_dataset import LardDataset


ROOT_PATH = os.environ.get('LARD_DATA_ROOT_PATH')
zip_folder = f"{ROOT_PATH}/LARD_dataset_decompressed/"
yolo_folder = f"{ROOT_PATH}/YoloFormat/"

# 训练集
dataset = LardDataset(
    train_path=os.path.join(zip_folder, "LARD_train"))
dataset.export(
    output_dir=os.path.join(yolo_folder, "train"),
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
# mv Test_Real_Nominal.csv LARD_test_real_nominal.csv
dataset = LardDataset(
    test_path=os.path.join(zip_folder, "LARD_test/LARD_test_real/LARD_test_real_nominal"))  # noqa
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
