import numpy as np
import matplotlib.pyplot as plt

plt.style.use('science')


def plot_pr_curve(
    coco_evals: dict,
    iou_indices: list = [0, 2, 4, 6, 8],
    exp_name: str = "",
):

    for data_type, coco_evaluator in coco_evals.items():

        assert data_type in [
            'test_synth',
            'test_real_nominal',
            'test_real_edge'
        ]

        print(data_type)

        precision = coco_evaluator.eval['precision']
        # recall = coco_evaluator.eval['recall']

        # precision.shape=(10, 101, 1, 4, 3), recall.shape=(10, 1, 4, 3)
        # print(f"{precision.shape=}, {recall.shape=}")

        # Assuming we have 5D precision data and 4D recall data from coco_eval
        # precision.shape should be (10, 101, num_classes, 4, 3)
        # recall.shape should be (10, num_classes, 4, 3)

        # num_iou_thresholds = 10  # Usually 10 IoU thresholds from 0.5 to 0.95
        # num_recall_levels = 101  # Usually 101 recall levels from 0 to 1
        # num_classes = 2  # Assuming 2 classes for demonstration
        # num_areas = 4   # Usually 4 areas ('all', 'small', 'medium', 'large')
        # num_max_dets = 3  # Usually 3 max detections [1, 10, 100]

        # 选择具体类别/区域/最大检测量
        class_index = 0
        area_index = 0
        max_dets_index = 2

        # 初始化图像
        plt.figure(figsize=(8, 4))

        # Precision-Recall曲线
        plt.subplot(1, 2, 1)
        for iou_index in iou_indices:
            selected_precision = precision[iou_index, :, class_index, area_index, max_dets_index] # noqa
            selected_recall = np.linspace(0, 1, len(selected_precision))
            plt.plot(selected_recall, selected_precision, label=f'IoU = {0.5 + iou_index * 0.05:.2f}') # noqa
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(frameon=True)
        plt.grid(True)

        # F1-Recall曲线
        plt.subplot(1, 2, 2)
        for iou_index in iou_indices:
            selected_precision = precision[iou_index, :, class_index, area_index, max_dets_index] # noqa
            selected_recall = np.linspace(0, 1, len(selected_precision))
            F1 = 2 * (selected_precision * selected_recall) / (selected_precision + selected_recall + 1e-6)  # noqa
            plt.plot(selected_recall, F1, label=f'IoU = {0.5 + iou_index * 0.05:.2f}')  # noqa
        plt.xlabel('Recall')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Recall')
        plt.legend(frameon=True)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'results/figs/test/pr_curve_{exp_name}_{data_type}.jpg', dpi=600, bbox_inches='tight')
        plt.show()


def plot_mmdet_fastern_rcnn_train_log(df_train, df_val, exp_name: str):

    _, axs = plt.subplots(2, 2, figsize=(7, 6), dpi=100)
    axs = axs.flatten()
    if len(df_train):
        axs[0].plot(df_train.index, df_train['lr'], label='lr')
        axs[0].legend(frameon=True)
        axs[0].set_title('Learning Rate vs Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Learning Rate')
        axs[0].set_xlim(left=0)

        axs[1].plot(
            df_train.index,
            df_train['loss'],
            label='loss')
        axs[1].plot(
            df_train.index,
            df_train['loss_rpn_cls'],
            label='loss_rpn_cls')
        axs[1].plot(
            df_train.index,
            df_train['loss_rpn_bbox'],
            label='loss_rpn_bbox')
        axs[1].plot(
            df_train.index,
            df_train['loss_cls'],
            label='loss_cls')
        axs[1].plot(
            df_train.index,
            df_train['loss_bbox'],
            label='loss_bbox')
        axs[1].legend(frameon=True)
        axs[1].set_title('Loss vs Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlim(left=0)

        axs[2].plot(
            df_train.index,
            df_train['acc'],
            label='acc')
        axs[2].legend(frameon=True)
        axs[2].set_title('ACC vs Epoch')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('ACC')
        axs[2].set_xlim(left=0)

    if len(df_val):
        axs[3].plot(
            df_val.index,
            df_val['coco/bbox_mAP'],
            label='bbox_mAP',
            linewidth=5)
        axs[3].plot(
            df_val.index,
            df_val['coco/bbox_mAP_50'],
            label='bbox_mAP_50')
        axs[3].plot(
            df_val.index,
            df_val['coco/bbox_mAP_75'],
            label='bbox_mAP_75')
        axs[3].legend(frameon=True)
        axs[3].set_title('coco/bbox_mAP vs Step')
        axs[3].set_xlabel('Epoch')
        axs[3].set_ylabel('coco/bbox_mAP')
        axs[3].set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(
        f'results/figs/train/train_log_{exp_name}.jpg',
        dpi=600, bbox_inches='tight')
    plt.show()


def plot_mmdet_ssd_train_log(df_train, df_val, exp_name: str):

    _, axs = plt.subplots(2, 2, figsize=(7, 6), dpi=100)
    axs = axs.flatten()
    if len(df_train):
        axs[0].plot(df_train.index, df_train['lr'], label='lr')
        axs[0].legend(frameon=True)
        axs[0].set_title('Learning Rate vs Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Learning Rate')
        axs[0].set_xlim(left=0)

        axs[1].plot(df_train.index, df_train['loss'], label='loss')
        axs[1].plot(df_train.index, df_train['loss_cls'], label='loss_cls')
        axs[1].plot(df_train.index, df_train['loss_bbox'], label='loss_bbox')
        axs[1].legend(frameon=True)
        axs[1].set_title('Loss vs Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlim(left=0)

    if len(df_val):
        axs[3].plot(
            df_val.index,
            df_val['coco/bbox_mAP'],
            label='bbox_mAP',
            linewidth=5)
        axs[3].plot(
            df_val.index,
            df_val['coco/bbox_mAP_50'],
            label='bbox_mAP_50')
        axs[3].plot(
            df_val.index,
            df_val['coco/bbox_mAP_75'],
            label='bbox_mAP_75')
        axs[3].legend(frameon=True)
        axs[3].set_title('coco/bbox_mAP vs Step')
        axs[3].set_xlabel('Epoch')
        axs[3].set_ylabel('coco/bbox_mAP')
        axs[3].set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(
        f'results/figs/train/train_log_{exp_name}.jpg',
        dpi=600, bbox_inches='tight')
    plt.show()


def plot_mmdet_yolov3_train_log(df_train, df_val, exp_name: str):

    _, axs = plt.subplots(2, 2, figsize=(7, 6), dpi=100)
    axs = axs.flatten()
    if len(df_train):
        axs[0].plot(df_train.index, df_train['lr'], label='lr')
        axs[0].legend(frameon=True)
        axs[0].set_title('Learning Rate vs Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Learning Rate')
        axs[0].set_xlim(left=0)

        axs[1].plot(df_train.index, df_train['grad_norm'], label='grad_norm')
        axs[1].legend(frameon=True)
        axs[1].set_title('Gradient Norm vs Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Gradient Norm')
        axs[1].set_xlim(left=0)

        axs[2].plot(df_train.index, df_train['loss'], label='loss')
        axs[2].plot(df_train.index, df_train['loss_cls'], label='loss_cls')
        axs[2].plot(df_train.index, df_train['loss_conf'], label='loss_conf')
        axs[2].plot(df_train.index, df_train['loss_xy'], label='loss_xy')
        axs[2].plot(df_train.index, df_train['loss_wh'], label='loss_wh')
        axs[2].legend(frameon=True)
        axs[2].set_title('Loss vs Epoch')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Loss')
        axs[2].set_xlim(left=0)

    if len(df_val):
        axs[3].plot(
            df_val.index,
            df_val['coco/bbox_mAP'],
            label='bbox_mAP',
            linewidth=5)
        axs[3].plot(
            df_val.index,
            df_val['coco/bbox_mAP_50'],
            label='bbox_mAP_50')
        axs[3].plot(
            df_val.index,
            df_val['coco/bbox_mAP_75'],
            label='bbox_mAP_75')
        axs[3].legend(frameon=True)
        axs[3].set_title('coco/bbox_mAP vs Step')
        axs[3].set_xlabel('Epoch')
        axs[3].set_ylabel('coco/bbox_mAP')
        axs[3].set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(
        f'results/figs/train/train_log_{exp_name}.jpg',
        dpi=600, bbox_inches='tight')
    plt.show()


def plot_ultralytics_yolov8_train_log(df_train_val, exp_name: str):

    _, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=100)
    axes = axes.flatten()

    axes[0].plot(df_train_val.index, df_train_val['train/box_loss'],
                label='Training bounding box loss')
    axes[0].plot(df_train_val.index, df_train_val['train/cls_loss'],
                label='Training classification loss')
    axes[0].plot(df_train_val.index, df_train_val['train/dfl_loss'],
                label='Training DFL loss')
    axes[0].legend(loc='upper right', frameon=True)

    axes[1].plot(df_train_val.index, df_train_val['val/box_loss'],
                label='Validation bounding box loss')
    axes[1].plot(df_train_val.index, df_train_val['val/cls_loss'],
                label='Validation classification loss')
    axes[1].plot(df_train_val.index, df_train_val['val/dfl_loss'],
                label='Validation DFL loss')
    axes[1].legend(loc='upper right', frameon=True)

    axes[2].plot(df_train_val.index, df_train_val['metrics/precision(B)'], label='Precision')
    axes[2].plot(df_train_val.index, df_train_val['metrics/recall(B)'], label='Recall')
    axes[2].plot(df_train_val.index, df_train_val['metrics/mAP50(B)'], label='mAP50')
    axes[2].plot(df_train_val.index, df_train_val['metrics/mAP50-95(B)'], label='mAP50-95')
    axes[2].legend(loc='lower right', ncol=2, frameon=True)

    axes[3].plot(df_train_val.index, df_train_val['lr/pg0'], label='Learning rate for group 0')
    axes[3].plot(df_train_val.index, df_train_val['lr/pg1'], label='Learning rate for group 1')
    axes[3].plot(df_train_val.index, df_train_val['lr/pg2'], label='Learning rate for group 2')
    axes[3].legend(loc='lower right', frameon=True)

    axes[0].set_xlabel('epoch')
    axes[1].set_xlabel('epoch')
    axes[2].set_xlabel('epoch')
    axes[3].set_xlabel('epoch')

    axes[0].set_ylabel('loss')
    axes[1].set_ylabel('loss')
    axes[2].set_ylabel('metrics')
    axes[3].set_ylabel('learning rate')

    axes[0].set_title('Train loss')
    axes[1].set_title('Validation loss')
    axes[2].set_title('Validation metrics')
    axes[3].set_title('Learning rate')

    plt.tight_layout()
    plt.savefig(f'results/figs/train/train_log_{exp_name}.jpg',
                dpi=600, bbox_inches='tight')
    plt.show()
