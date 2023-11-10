import os
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

coco_metrics_name = [
    "AP (IoU=0.50:0.95, area=all, maxDets=100)",
    "AP (IoU=0.50, area=all, maxDets=100)",
    "AP (IoU=0.75, area=all, maxDets=100)",
    "AP (IoU=0.50:0.95, area=small, maxDets=100)",
    "AP (IoU=0.50:0.95, area=medium, maxDets=100)",
    "AP (IoU=0.50:0.95, area=large, maxDets=100)",
    "AR (IoU=0.50:0.95, area=all, maxDets=1)",
    "AR (IoU=0.50:0.95, area=all, maxDets=10)",
    "AR (IoU=0.50:0.95, area=all, maxDets=100)",
    "AR (IoU=0.50:0.95, area=small, maxDets=100)",
    "AR (IoU=0.50:0.95, area=medium, maxDets=100)",
    "AR (IoU=0.50:0.95, area=large, maxDets=100)"
]


def get_coco_imgname_2_imgid(annotation_file: str) -> dict:

    # 获取一个字典 name_2_id {图片名(无拓展名): 图片id}
    coco = COCO(annotation_file=annotation_file)
    img_ids = coco.getImgIds()

    imgname_2_imgid = {}
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        filename = os.path.splitext(img_info['file_name'])[0]  # 无拓展名的文件名
        imgname_2_imgid[filename] = img_id

    return imgname_2_imgid


def get_coco_imgid_2_imgname(annotation_file: str) -> dict:

    # 获取一个字典 id_2_name {图片id: 图片名(无拓展名)}
    coco = COCO(annotation_file=annotation_file)
    img_ids = coco.getImgIds()

    imgid_2_imgname = {}
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        filename = os.path.splitext(img_info['file_name'])[0]  # 无拓展名的文件名
        imgid_2_imgname[img_id] = filename

    return imgid_2_imgname


def cal_coco_metrics(annotation_file: str, prediction_file: str):

    coco_true = COCO(annotation_file=annotation_file)
    coco_pred = coco_true.loadRes(resFile=prediction_file)

    coco_evaluator = COCOeval(
        cocoGt=coco_true,
        cocoDt=coco_pred,
        iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator
