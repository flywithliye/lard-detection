# 系统配置

1. 创建环境变量 `vim ~/.bashrc`

   ```bash
   export LARD_DATA_ROOT_PATH='/home/yeli/yeli/data/lard'
   export LARD_YOLO_ROOT_PATH='/home/yeli/Nextcloud/lard/yolov8'
   ```
2. 激活环境变量 `source ~/.bashrc`

# 安装第三方包

## mmdet

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

* [ ] TODO

# 其他

* [ ] TODO
