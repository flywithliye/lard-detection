# 系统配置

1. 创建环境变量 `vim ~/.bashrc`

   ```bash
   export LARD_DATA_ROOT_PATH='/home/yeli/yeli/data/lard'
   export LARD_YOLO_ROOT_PATH='/home/yeli/Nextcloud/lard/yolov8'
   ```
2. 激活环境变量 `source ~/.bashrc`

# ultralytics配置

1. ultralytics/cfg/default.yaml 追加配置 `custom_albumentation: False`
2. ultralytics/data/augment.py 函数v8_transforms返回值 `Albumentations`部分修改为 `Albumentations(p=1.0,custom_albumentation=hyp.custom_albumentation)`
3. ultralytics/data/augment.py `Albumentations`的初始化函数补充形参 `custom_albumentation=False`
4. ultralytics/data/augment.py `Albumentations`的初始化函数T构造部分后面补充
   ```python
   if custom_albumentation:
       T = [
           A.Blur(p=0.01),
       ]
   ```

# 其他
