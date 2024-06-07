_backend_args = None

# 实验参数
model_name = 'yolov8n'
model_stru = ''
model_cfg = ''
img_size = 1280
exp_name = f'{model_name}{model_stru}{model_cfg}_{img_size}'

work_dir = f'runs/mmyolo/{exp_name}/train'

# 数据集
data_root = 'datasets/lard/'
dataset_type = 'YOLOv5LardDataset'
img_scale = (
    img_size,
    img_size,
)

# 常用修改参数
num_workers = 8
num_epochs = 500
batch_size = dict(
    train=16,
    val=16,
    test=16
)

# 随机性控制
randomness = dict(
    seed = 0,
    diff_rank_seed=True,
    deterministic=True
)

# 一些参数
base_lr = 0.01 * (batch_size['train']*10) / (16*8)
lr_factor = 0.01
affine_scale = 0.5
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
last_transform = [
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=albu_train_transforms,
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
close_mosaic_epochs = 10

# 钩子定义
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=490,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=img_scale, type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=img_scale,
                type='LetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='YOLOv5RandomAffine'),
            *last_transform
        ],
        type='mmdet.PipelineSwitchHook'),
]
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=1, save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=25, type='LoggerHook'),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=50,
        min_delta=0.005),
    param_scheduler=dict(
        lr_factor=lr_factor,
        max_epochs=num_epochs,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_aspect_ratio = 100

# 模型定义
widen_factor = 0.25
deepen_factor = 0.33
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=deepen_factor,
        last_stage_out_channels=1024,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv8CSPDarknet',
        widen_factor=widen_factor),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                1024,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=1,
            reg_max=16,
            type='YOLOv8HeadModule',
            widen_factor=widen_factor),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=loss_bbox_weight,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=loss_cls_weight,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=loss_dfl_weight,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOv8Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=deepen_factor,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            1024,
        ],
        type='YOLOv8PAFPN',
        widen_factor=widen_factor),
    test_cfg=dict(
        max_per_img=8,
        multi_label=False,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-09,
            num_classes=1,
            topk=tal_topk,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLODetector')
num_det_layers = 3

# 优化器和调度器
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=batch_size['train'],
        lr=base_lr,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = None
resume = False

# 模型训练
train_cfg = dict(
    dynamic_intervals=[
        (
            490,
            1,
        ),
    ],
    max_epochs=num_epochs,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    *last_transform,
]
train_dataloader = dict(
    batch_size=batch_size['train'],
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        dataset=dict(
            ann_file='annotations/instances_train.json',
            data_prefix=dict(img='detection/train/images/'),
            data_root=data_root,
            filter_cfg=dict(filter_empty_gt=False, min_size=0),
            pipeline=train_pipeline,
            type=dataset_type),
        times=1,
        type='RepeatDataset'),
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=img_scale, type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=img_scale,
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    *last_transform,
]

# 模型验证
val_cfg = dict(type='ValLoop')
val_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=img_scale, type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=img_scale,
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
val_dataloader = dict(
    batch_size=batch_size['val'],
    dataset=dict(
        ann_file='annotations/instances_val.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='detection/val/images/'),
        data_root=data_root,
        pipeline=val_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=data_root+'annotations/instances_val.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    format_only=False,
    type='mmdet.CocoMetric')
val_interval_stage2 = 1

# 模型测试
test_cfg = dict(type='TestLoop')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=img_scale, type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=img_scale,
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
test_dataloader = dict(
    batch_size=batch_size['test'],
    dataset=dict(
        ann_file='annotations/instances_test_synth.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='detection/test_synth/images/'),
        data_root=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=data_root+'annotations/instances_test_synth.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    format_only=True,
    outfile_prefix=f'runs/mmyolo/{exp_name}/test/coco_detection/prediction_test_synth',
    type='mmdet.CocoMetric')

# 测试时增强配置
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                v=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=8, nms=dict(iou_threshold=0.6, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends)
