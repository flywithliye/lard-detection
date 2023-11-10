auto_scale_lr = dict(base_batch_size=128, enable=False)
backend_args = None

model_name = 'yolov3'
model_stru = ''
model_cfg = ''
exp_name = f'{model_name}{model_stru}{model_cfg}'

work_dir = f'runs/mmdetection/{exp_name}/train'
data_root = 'datasets/lard/'
dataset_type = 'LardDataset'
input_size = (
    608,
    608,
)
num_workers = 48
num_epochs = 300
batch_size = dict(
    train=64,
    val=64,
    test=64
)

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
    logger=dict(interval=50, type='LoggerHook'),
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
model = dict(
    backbone=dict(
        depth=53,
        init_cfg=dict(checkpoint='open-mmlab://darknet53', type='Pretrained'),
        out_indices=(
            3,
            4,
            5,
        ),
        type='Darknet'),
    bbox_head=dict(
        anchor_generator=dict(
            base_sizes=[
                [(120, 76), (198, 113), (366, 188)],
                [(42, 22), (46, 41), (75, 52)],
                [(10, 14), (19, 19), (26, 31)]
            ],
            strides=[
                32,
                16,
                8,
            ],
            type='YOLOAnchorGenerator'),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[
            32,
            16,
            8,
        ],
        in_channels=[
            512,
            256,
            128,
        ],
        loss_cls=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_conf=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_wh=dict(loss_weight=2.0, reduction='sum', type='MSELoss'),
        loss_xy=dict(
            loss_weight=2.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=1,
        out_channels=[
            1024,
            512,
            256,
        ],
        type='YOLOV3Head'),
    data_preprocessor=data_preprocessor,
    neck=dict(
        in_channels=[
            1024,
            512,
            256,
        ],
        num_scales=3,
        out_channels=[
            512,
            256,
            128,
        ],
        type='YOLOV3Neck'),
    test_cfg=dict(
        conf_thr=0.005,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.2),
    train_cfg=dict(
        assigner=dict(
            min_pos_iou=0,
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='GridAssigner')),
    type='YOLOV3')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=3, start_factor=0.001, type='LinearLR'),
    dict(
        begin=3,
        by_epoch=True,
        end=num_epochs,
        gamma=0.1,
        milestones=[
            int(num_epochs*0.8),
            int(num_epochs*0.9),
        ],
        type='MultiStepLR'),
]
resume = False
train_cfg = dict(
    max_epochs=num_epochs,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(mean=[
        0,
        0,
        0,
    ], ratio_range=(
        1,
        2,
    ), to_rgb=True, type='Expand'),
    dict(
        min_crop_size=0.3,
        min_ious=(
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),
        type='MinIoURandomCrop'),
    dict(keep_ratio=True, scale=input_size, type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
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
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=vis_backends)
