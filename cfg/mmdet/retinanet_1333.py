# (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16, enable=True)

# 实验参数
model_name = 'retinanet'
model_stru = ''
model_cfg = ''
img_size = 1333
exp_name = f'{model_name}{model_stru}{model_cfg}_{img_size}'

# 数据集
work_dir = f'runs/mmdetection/{exp_name}/train'
data_root = 'datasets/lard/'
dataset_type = 'LardDataset'
input_size = (
    1333,
    800,
)

# 常用修改参数
num_workers = 8
num_epochs = 500
batch_size = dict(
    train=2,
    val=2,
    test=2
)

# 随机性控制
randomness = dict(
    seed=0,
    diff_rank_seed=True,
    deterministic=True
)

# 一些参数
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        122.00711516,
        141.11828193,
        164.56574534
    ],
    pad_size_divisor=32,
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
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=1,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=data_preprocessor,
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=8,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')

# 优化器和调度器
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
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
            data_prefix=dict(img='detection/train/images/'),
            data_root=data_root,
            filter_cfg=dict(filter_empty_gt=True, min_size=0),
            pipeline=train_pipeline,
            type=dataset_type),
        times=1,
        type='RepeatDataset'
    ),
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

# 模型验证配置
val_cfg = dict(type='ValLoop')
val_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=input_size, type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
val_dataloader = dict(
    batch_size=batch_size['val'],
    dataset=dict(
        ann_file='annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img='detection/val/images/'),
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
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=input_size, type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
test_dataloader = dict(
    batch_size=batch_size['test'],
    dataset=dict(
        ann_file='annotations/instances_test_synth.json',
        backend_args=None,
        data_prefix=dict(img='detection/test_synth/images/'),
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
