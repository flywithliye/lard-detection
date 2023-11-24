from PIL import Image
import fiftyone as fo
import json
import pandas as pd
from tqdm import tqdm
from .metric import get_coco_imgid_2_imgname

keep_fileds = ["id", "filepath", "tags", "metadata", "ground_truth_detections"]


def write_detections_2_database(lib_type: str, exp_name: str):

    assert lib_type in ['mmdetection', 'mmyolo', 'ultralytics']

    for data_type in ['test_synth', 'test_real_nominal', 'test_real_edge']:

        extension = ".jpeg" if "synth" in data_type else ".png"

        # 预测JSON
        if lib_type == 'ultralytics':
            path_predictions = f"runs/ultralytics/{exp_name}/{data_type}/predictions.json"
        elif lib_type == 'mmdetection':
            path_predictions = f"runs/mmdetection/{exp_name}/test/coco_detection/prediction_{data_type}.bbox.json"
        else:
            path_predictions = f"runs/mmyolo/{exp_name}/test/coco_detection/prediction_{data_type}.bbox.json"

        # 标签JSON
        path_annotations = f"datasets/lard/annotations/instances_{data_type}.json"
        imgid_2_imgname = get_coco_imgid_2_imgname(path_annotations)

        # 读取数据集
        dataset = fo.load_dataset(f"lard_{data_type}")

        # 类别信息
        classes = dataset.default_classes

        # 加载 JSON 预测文件
        with open(path_predictions, "r") as f:
            predictions_data = json.load(f)

        # 遍历追加预测结果
        for sample in tqdm(dataset, ncols=100, desc=f"正在写入{data_type}检测结果"):

            # 文件名
            filename = sample.filepath.split(
                '/')[-1].split(extension)[0]  # sample 中只保留文件名

            # 图像长宽
            image = Image.open(sample.filepath)
            width, height = image.size

            # 筛选预测
            # ultralytics输出预测rec['image_id']即为文件名
            if lib_type == 'ultralytics':
                predictions_for_sample = [
                    rec for rec in predictions_data if rec['image_id'] == filename]
            # mmdet输出预测rec['image_id']为真实id 需要转换为filename
            else:
                predictions_for_sample = [
                    rec for rec in predictions_data if imgid_2_imgname[rec['image_id']] == filename]

            # 构造detections对象
            detections = []
            for pred in predictions_for_sample:

                category_id = pred['category_id']
                bbox = pred['bbox']
                score = pred['score']

                x1, y1, w, h = bbox
                rel_box = [x1 / width, y1 / height, w / width, h / height]

                detections.append(
                    fo.Detection(
                        label=classes[category_id],
                        bounding_box=rel_box,
                        confidence=score
                    )
                )

            sample[f"predictions_{exp_name}"] = fo.Detections(
                detections=detections)
            sample.save()

        # 检查预测结果字段
        field_schema = dataset.get_field_schema()
        for field_name, field_type in field_schema.items():
            if field_name.find(exp_name) != -1 or field_name.find(exp_name.replace('-', '_')) != -1:
                print(f"{field_name}: {field_type}")


def delete_detections_from_database(exp_name: str):

    for data_type in ['test_synth', 'test_real_nominal', 'test_real_edge']:

        # 读取数据集
        dataset = fo.load_dataset(f"lard_{data_type}")

        # 要删除字段列表
        field_schema = dataset.get_field_schema()
        fileds_to_delete = []

        for field_name, _ in field_schema.items():
            if field_name.find(exp_name) != -1 or field_name.find(exp_name.replace('-', '_')) != -1:
                fileds_to_delete.append(field_name)

        print(f"删除字段: {fileds_to_delete}")

        # 删除字段
        dataset.delete_sample_fields(fileds_to_delete)


def delete_all_predictions(dataset):

    field_schema = dataset.get_field_schema()
    
    # 要删除字段列表
    fileds_to_delete = []
    for field_name, _ in field_schema.items():
        if field_name.find("predictions_") != -1:
            fileds_to_delete.append(field_name)

    print(f"删除字段: {fileds_to_delete}")
    dataset.delete_sample_fields(fileds_to_delete)


def eval_detections_in_database(exp_name: str):

    all_results = {}

    for data_type in ['test_synth', 'test_real_nominal', 'test_real_edge']:

        print(f'正在评估: {data_type}')

        # 读取数据集
        dataset = fo.load_dataset(f"lard_{data_type}")

        # 评估
        results = dataset.evaluate_detections(
            pred_field=f"predictions_{exp_name}",
            gt_field="ground_truth_detections",
            method="coco",
            eval_key=exp_name.replace('-', '_'),
            compute_mAP=True,
        )

        # 打印报告
        results.print_report()
        map = results.mAP()
        metrics = pd.DataFrame(results.metrics(), index=[0])

        # 打印指标
        print(f"mAP: {map:.3f}")
        print(metrics.round(3))

        all_results[data_type] = results

    return all_results
