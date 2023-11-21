# LARD-Detection

围绕LARD数据集展开的目标检测研究

# 下载

```bash
git clone --recurse-submodules https://github.com/flywithliye/lard-detection.git
git submodule update --init --recursive
```

# 系统配置

1. 创建环境变量 `vim ~/.bashrc`

   ```bash
   export LARD_DATA_ROOT_PATH='/home/yeli/yeli/data/lard'
   export LARD_PROJECT_ROOT_PATH='/home/yeli/Nextcloud/lard/lard-detection'
   ```
2. 激活环境变量 `source ~/.bashrc`

# 创建环境

```bash
conda create -n lard
conda activate lard

conda install python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

# 安装第三方包

## 其他

```bash
pip install fiftyone
pip install fiftyone-db-ubuntu2204
pip install scienceplots
```

## LARD

```bash
# 添加子模块
git submodule add https://github.com/flywithliye/LARD.git src/data/LARD
git submodule init
git submodule update
git add .gitmodules src/data/LARD
```

## mmdet

pip install -U openmim
mim install mmengine
pip install "mmcv>=2.0.0rc4,<2.1.0"

```bash
# 添加子模块
git submodule add https://github.com/flywithliye/mmdetection.git 3rdparty/mmdetection
git submodule init
git submodule update
git add .gitmodules 3rdparty/mmdetection
git commit -m "Added mmdetection submodule"

# 下载安装
git clone https://github.com/flywithliye/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## mmyolo

```bash
# 添加子模块
git submodule add https://github.com/flywithliye/mmyolo.git 3rdparty/mmyolo
git submodule init
git submodule update
git add .gitmodules 3rdparty/mmyolo
git commit -m "Added mmdetection submodule"
```

## ultralytics

```bash
# 添加子模块
git submodule add https://github.com/flywithliye/ultralytics.git 3rdparty/ultralytics
git submodule init
git submodule update
git add .gitmodules 3rdparty/ultralytics
git commit -m "Added ultralytics submodule"

# 下载安装
git clone https://github.com/flywithliye/ultralytics.git
cd ultralytics
pip install -v -e .
```

安装完成后，执行 `pip list | grep -e mmdetection -e mmyolo -e ultralytics`有如下输出

```bash
mmdet                         3.2.0                /home/yeli/workspace/lard/lard-detection/3rdparty/mmdetection
mmyolo                        0.6.0                /home/yeli/workspace/lard/lard-detection/3rdparty/mmyolo
ultralytics                   8.0.203              /home/yeli/workspace/lard/lard-detection/3rdparty/ultralytics
```

# LARD配置

1. lard_dataset.py文件import部分:

   ```python
   from src.labeling.labels import Labels
   ```

   修改为

   ```python
   from LARD.src.labeling.labels import Labels
   ```
2. lard_dataset.py文件注释掉224行补充以下内容:

   ```python
   dataset_dir = output_dir
   ```
3. labels.py文件import部分:

   ```python
   from src.labeling.export_config import CORNERS_NAMES
   ```

   修改为

   ```python
   from LARD.src.labeling.export_config import CORNERS_NAMES
   ```

# ultralytics配置

> 以下内容可通过执行 `src/code_ultralytics/prepare_ultralytics.ipynb`文件实现

1. ultralytics/cfg/default.yaml 追加配置：

   ```yaml
   use_custom_aug: False # 使用自定义albumentations
   path_transform:  # 自定义albumentations文件路径
   ```
2. ultralytics/data/augment.py 函数v8_transforms返回值 `Albumentations`部分修改为：

   ```python
   Albumentations(p=1.0, custom_aug=hyp.custom_aug, path_transform=hyp.path_transform, imgsz=hyp.imgsz)
   ```
3. ultralytics/data/augment.py `Albumentations`的初始化函数补充形参：

   ```
   def __init__(self, p=1.0, custom_aug=False, path_transform="", imgsz=640):
   ```
4. ultralytics/data/augment.py `Albumentations`的初始化函数中，注释掉 `check_version`部分。
5. ultralytics/data/augment.py `Albumentations`的初始化函数中，在 `self.transform`后面补充：

   ```python
   if custom_aug:
       T = A.load(path_transform)  # albumentations.core.composition.Compose
       T.transforms[4].transforms[0].width = imgsz  # RandomSizedBBoxSafeCrop
       T.transforms[4].transforms[0].height = imgsz
       T.transforms[-1].width = imgsz  # Resize
       T.transforms[-1].height = imgsz
   self.transform = T
   ```

# mmdet配置

> 以下内容可通过执行 `src/code_mmdet/prepare_mmdet.ipynb`文件实现

1. 在数据集 `mmdetection/mmdet/datasets/lard.py`下构建LARD数据集类:

   ```python
   from mmdet.registry import DATASETS
   from .coco import CocoDataset


   @DATASETS.register_module()
   class LardDataset(CocoDataset):
       """Dataset for LARD."""

       METAINFO = {
           'classes':
           ('runway',),
           # palette is a list of color tuples, which is used for visualization.
           'palette':
           [(220, 20, 60),]
       }
   ```
2. 在初始化部分 `mmdetection/mmdet/datasets/__init__.py`注册LARD数据集类：

   ```python
   from .lard import LardDataset

   __all__ = [
       'XMLDataset', 'CocoDataset', 'LardDataset', 'DeepFashionDataset', 'VOCDataset',
       'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset', 'LVISV1Dataset',
       'WIDERFaceDataset', 'get_loading_pipeline', 'CocoPanopticDataset',
       'MultiImageMixDataset', 'OpenImagesDataset', 'OpenImagesChallengeDataset',
       'AspectRatioBatchSampler', 'ClassAwareSampler', 'MultiSourceSampler',
       'GroupMultiSourceSampler', 'BaseDetDataset', 'CrowdHumanDataset',
       'Objects365V1Dataset', 'Objects365V2Dataset', 'DSDLDetDataset',
       'BaseVideoDataset', 'MOTChallengeDataset', 'TrackImgSampler',
       'ReIDDataset', 'YouTubeVISDataset', 'TrackAspectRatioBatchSampler',
       'ADE20KPanopticDataset', 'CocoCaptionDataset', 'RefCocoDataset',
       'BaseSegDataset', 'ADE20KSegDataset', 'CocoSegDataset',
       'ADE20KInstanceDataset', 'iSAIDDataset', 'V3DetDataset', 'ConcatDataset'
   ]
   ```
3. 在 `mmdetection/mmdet/evaluation/functional/class_names.py`中添加LARD类别名字定义:

   ```python
   def lard_classes() -> list:
       """Class names of LARD."""
       return [
           'runway',
       ]
   ```
4. 在 `mmdetection/mmdet/evaluation/functional/__init__.py`内注册LARD类别：

   ```python
   from .class_names import (cityscapes_classes, coco_classes, lard_classes,
                               coco_panoptic_classes, dataset_aliases, get_classes,
                               imagenet_det_classes, imagenet_vid_classes,
                               objects365v1_classes, objects365v2_classes,
                               oid_challenge_classes, oid_v6_classes, voc_classes)

   __all__ = [
       'voc_classes', 'imagenet_det_classes', 'imagenet_vid_classes',
       'coco_classes', 'lard_classes', 'cityscapes_classes', 'dataset_aliases', 'get_classes',
       'average_precision', 'eval_map', 'print_map_summary', 'eval_recalls',
       'print_recall_summary', 'plot_num_recall', 'plot_iou_recall',
       'oid_v6_classes', 'oid_challenge_classes', 'INSTANCE_OFFSET',
       'pq_compute_single_core', 'pq_compute_multi_core', 'bbox_overlaps',
       'objects365v1_classes', 'objects365v2_classes', 'coco_panoptic_classes',
       'evaluateImgLists', 'YTVIS', 'YTVISeval'
   ]
   ```
5. 修改 `mmdetection/mmdet/models/dense_heads/base_dense_head.py`文件

   **Before modification**

   ```python
   if getattr(self.loss_cls, 'custom_cls_channels', False):
       scores = self.loss_cls.get_activation(cls_score)
   elif self.use_sigmoid_cls:
       scores = cls_score.sigmoid()
   else:
       # remind that we set FG labels to [0, num_class-1]
       # since mmdet v2.0
       # BG cat_id: num_class
       scores = cls_score.softmax(-1)[:, :-1]
   ```

   **After modification**

   ```python
   if False: # getattr(self.loss_cls, 'custom_cls_channels', False):  # Change made here
       scores = self.loss_cls.get_activation(cls_score)
   elif self.use_sigmoid_cls:
       scores = cls_score.sigmoid()
   else:
       # remind that we set FG labels to [0, num_class-1]
       # since mmdet v2.0
       # BG cat_id: num_class
       scores = cls_score.softmax(-1)[:, :-1]
   ```

对子模块进行上述修改后，若出现diff文件并提示dirty，则在vscode中勾选 `git.ignoreSubmodules`

# 其他

* [ ] TODO
