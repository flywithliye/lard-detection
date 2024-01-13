# (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=128, enable=True)
backend_args = None

# 实验参数
model_name = 'centernet'
model_stru = ''
model_cfg = ''
exp_name = f'{model_name}{model_stru}{model_cfg}'

# 数据集
work_dir = f'runs/mmdetection/{exp_name}/train'
data_root = 'datasets/lard/'
dataset_type = 'LardDataset'
input_size = (
    512,
    512,
)

# 常用修改参数
num_workers = 8
num_epochs = 500
batch_size = dict(
    train=16,
    val=16,
    test=8
)

# 随机性控制
randomness = dict(
    seed = 0,
    diff_rank_seed=True,
    deterministic=True
)

# 一些参数
data_preprocessor=dict(
    bgr_to_rgb=True,
    mean=[
        122.00711516,
        141.11828193,
        164.56574534
    ],
    std=[
        46.91310377,
        54.8164231,
        70.38650678
    ],
    type='DetDataPreprocessor')
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        save_best='auto',
        max_keep_ckpts=1,
        type='CheckpointHook'),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=50,
        min_delta=0.005),
    logger=dict(interval=25, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

# 模型定义
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        norm_cfg=dict(type='BN'),
        norm_eval=False,
        type='ResNet'),
    bbox_head=dict(
        feat_channels=64,
        in_channels=64,
        loss_center_heatmap=dict(loss_weight=1.0, type='GaussianFocalLoss'),
        loss_offset=dict(loss_weight=1.0, type='L1Loss'),
        loss_wh=dict(loss_weight=0.1, type='L1Loss'),
        num_classes=1,
        type='CenterNetHead'),
    data_preprocessor=data_preprocessor,
    neck=dict(
        in_channels=512,
        num_deconv_filters=(
            256,
            128,
            64,
        ),
        num_deconv_kernels=(
            4,
            4,
            4,
        ),
        type='CTResNetNeck',
        use_dcn=True),
    test_cfg=dict(local_maximum_kernel=3, max_per_img=100, topk=100),
    train_cfg=None,
    type='CenterNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=True, end=3, start_factor=0.001, type='LinearLR'),
    dict(
        begin=3,
        by_epoch=True,
        end=num_epochs,
        gamma=0.1,
        milestones=[
            int(num_epochs*0.7),
            int(num_epochs*0.8),
        ],
        type='MultiStepLR'),
]

# 模型训练配置
resume = False
train_cfg = dict(
    max_epochs=num_epochs, 
    type='EpochBasedTrainLoop', 
    val_interval=1)
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.5,
            1.5,
        ),
        hue_delta=18,
        saturation_range=(
            0.5,
            1.5,
        ),
        type='PhotoMetricDistortion'),
    dict(
        crop_size=input_size,
        mean=[
            0,
            0,
            0,
        ],
        ratios=(
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
        ),
        std=[
            1,
            1,
            1,
        ],
        test_pad_mode=None,
        to_rgb=True,
        type='RandomCenterCropPad'),
    dict(keep_ratio=True, scale=input_size, type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=batch_size['train'],
    dataset=dict(
        dataset=dict(
            ann_file='annotations/instances_train.json',
            backend_args=None,
            data_prefix=dict(img='YoloFormat/train/images/'),
            data_root=data_root,
            filter_cfg=dict(filter_empty_gt=True, min_size=0),
            pipeline=train_pipeline,
            type=dataset_type),
        times=1,
        type='RepeatDataset'),
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

# 模型验证配置
val_cfg = dict(type='ValLoop')
val_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(
        border=None,
        mean=[
            0,
            0,
            0,
        ],
        ratios=None,
        std=[
            1,
            1,
            1,
        ],
        test_mode=True,
        test_pad_add_pix=1,
        test_pad_mode=[
            'logical_or',
            31,
        ],
        to_rgb=True,
        type='RandomCenterCropPad'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'border',
        ),
        type='PackDetInputs'),
]
val_dataloader = dict(
    batch_size=batch_size['val'],
    dataset=dict(
        ann_file='annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img='YoloFormat/val/images/'),
        data_root=data_root,
        pipeline=val_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=data_root+'annotations/instances_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

# 模型测试配置
test_cfg = dict(type='TestLoop')
test_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(
        border=None,
        mean=[
            0,
            0,
            0,
        ],
        ratios=None,
        std=[
            1,
            1,
            1,
        ],
        test_mode=True,
        test_pad_add_pix=1,
        test_pad_mode=[
            'logical_or',
            31,
        ],
        to_rgb=True,
        type='RandomCenterCropPad'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'border',
        ),
        type='PackDetInputs'),
]
test_dataloader = dict(
    batch_size=batch_size['test'],
    dataset=dict(
        ann_file='annotations/instances_test_synth.json',
        backend_args=None,
        data_prefix=dict(img='YoloFormat/test_synth/images/'),
        data_root=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=data_root+'annotations/instances_test_synth.json',
    backend_args=None,
    format_only=True,
    metric='bbox',
    outfile_prefix=f'runs/mmdetection/{exp_name}/test/coco_detection/prediction_test_synth',
    type='CocoMetric')

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=vis_backends)
