import os
import json
from PIL import Image
from tqdm import tqdm


ROOT_DATA = os.environ.get('LARD_DATA_ROOT_PATH')


# 创建JSON文件夹
PATH_ANNOTATIONS = f"{ROOT_DATA}/annotations"
os.makedirs(PATH_ANNOTATIONS, exist_ok=True)


def yolo_to_coco(path, json_filename, is_real=False, is_mini=False):
    # 定义转换函数

    # 路径
    yolo_labels_dir = path + "/labels"
    yolo_images_dir = path + "/images"
    output_json_path = os.path.join(PATH_ANNOTATIONS, json_filename)

    # 处理不一致的图片格式
    extension = ".png" if is_real else ".jpeg"

    # 获取所有图像和标注文件
    image_files = [f for f in os.listdir(
        yolo_images_dir) if f.endswith(extension)]
    label_files = [f.replace(extension, '.txt') for f in image_files]

    # mini数据
    if is_mini:
        image_files = image_files[:100]
        label_files = label_files[:100]

    # 初始化COCO数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 类别信息
    category_dict = {0: 'runway'}
    for cat_id, cat_name in category_dict.items():
        coco_data["categories"].append({"id": cat_id, "name": cat_name})

    # 遍历各YOLO记录（图像+标签）
    annotation_id = 0
    for img_id, (img_file, label_file) in enumerate(
        tqdm(zip(image_files, label_files),
             desc=f"正在转换:{json_filename}",
             total=len(image_files),
             ncols=100)):

        # 图像信息
        img_path = os.path.join(yolo_images_dir, img_file)
        img = Image.open(img_path)
        height, width = img.height, img.width
        coco_data["images"].append(
            {
                "id": img_id,  # os.path.splitext(img_file)[0],
                "file_name": img_file,
                "width": width,
                "height": height
            }
        )

        # 标签信息
        with open(os.path.join(yolo_labels_dir, label_file), 'r') as f:
            for line in f:
                cat_id, x_center, y_center, w, h = map(
                    float, line.strip().split())
                cat_id = int(cat_id)

                x1 = (x_center - w / 2) * width
                y1 = (y_center - h / 2) * height
                bbox_width = w * width
                bbox_height = h * height

                coco_data["annotations"].append({
                    "id": annotation_id,
                    # os.path.splitext(img_file)[0],  # 图片名代表id
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

    # 保存为JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


# 全量数据集
# 全部训练集
# yolo_to_coco(
#     f"{ROOT_DATA}/YoloFormat/train_all",
#     "instances_train_all.json")

# # 训练&验证
# yolo_to_coco(
#     f"{ROOT_DATA}/YoloFormat/train",
#     "instances_train.json")
# yolo_to_coco(
#     f"{ROOT_DATA}/YoloFormat/val",
#     "instances_val.json")

# # 测试集
# yolo_to_coco(
#     f"{ROOT_DATA}/YoloFormat/test_synth",
#     "instances_test_synth.json")
# yolo_to_coco(
#     f"{ROOT_DATA}/YoloFormat/test_real_nominal",
#     "instances_test_real_nominal.json",
#     is_real=True)
# yolo_to_coco(
#     f"{ROOT_DATA}/YoloFormat/test_real_edge",
#     "instances_test_real_edge.json",
#     is_real=True)


# # mini数据集
# 全部训练集
yolo_to_coco(
    f"{ROOT_DATA}/YoloFormat/train_all",
    "instances_train_all_mini.json",
    is_mini=True)

# 训练&验证
yolo_to_coco(
    f"{ROOT_DATA}/YoloFormat/train",
    "instances_train_mini.json",
    is_mini=True)
yolo_to_coco(
    f"{ROOT_DATA}/YoloFormat/val",
    "instances_val_mini.json",
    is_mini=True)

# 测试集
yolo_to_coco(
    f"{ROOT_DATA}/YoloFormat/test_synth",
    "instances_test_synth_mini.json",
    is_mini=True)
yolo_to_coco(
    f"{ROOT_DATA}/YoloFormat/test_real_nominal",
    "instances_test_real_nominal_mini.json",
    is_real=True,
    is_mini=True)
yolo_to_coco(
    f"{ROOT_DATA}/YoloFormat/test_real_edge",
    "instances_test_real_edge_mini.json",
    is_real=True,
    is_mini=True)
