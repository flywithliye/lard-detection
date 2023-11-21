import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False


def draw_gt_pred(image, anns, preds, catid_2_catname: dict):
    
    # 字体和绘图参数
    FONT_SCALE = 2
    FONT = cv2.FONT_HERSHEY_TRIPLEX
    LINE_WIDTH = 5
    COLOR_TEXT = (255, 255, 255)
    COLOR_GT = (255, 192, 203)  # 粉色
    COLOR_PRED = (135, 206, 235)  # 天蓝
    ALPHA = 0.4

    # 备份图像
    img = image.copy()

    # 绘制 GT bbox
    for ann in anns:

        # bbox位置
        x_left, y_top, w, h = [int(x) for x in ann['bbox']]
        x_right = x_left + w
        y_bottom = y_top + h
        label_id = ann['category_id']

        # 类别label
        label_text = catid_2_catname[label_id]

        # 文本 [类名称]
        text = label_text.capitalize()

        # 文字尺寸
        (w_text, h_text), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 2)

        # 绘制bbox
        cv2.rectangle(
            img,
            (x_left, y_top),
            (x_right, y_bottom),
            COLOR_GT, LINE_WIDTH)
        # MASK
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (x_left, y_top),
            (x_right, y_bottom),
            COLOR_GT, -1)
        cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0, img)

        # 绘制文字背景 bbox上部
        bg_x_right = x_right if w_text < w else x_left + w_text
        h_bias = 30
        # 填充
        cv2.rectangle(
            img,
            (x_left, y_top),  # 左下角
            (bg_x_right, y_top-h_text-h_bias),  # 右上角
            COLOR_GT, -1)
        # 边缘
        cv2.rectangle(
            img,
            (x_left, y_top),  # 左下角
            (bg_x_right, y_top-h_text-h_bias),  # 右上角
            COLOR_GT, LINE_WIDTH)

        # 绘制文本
        cv2.putText(
            img,
            text,
            (x_left, y_top-20),
            FONT, FONT_SCALE, COLOR_TEXT, 2)

    # 绘制 Pred bbox
    for pred in preds:

        # bbox位置
        x_left, y_top, w, h = [int(x) for x in pred['bbox']]
        x_right = x_left + w
        y_bottom = y_top + h

        # 类别label
        label_id = pred['category_id']
        label_text = catid_2_catname[label_id]

        # 置信度
        score = pred['score']

        # 文本 [类别+置信度]
        text = f"{label_text} ({score:.2f})".capitalize()

        # 文字尺寸
        (w_text, h_text), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 2)

        # 绘制bbox
        cv2.rectangle(
            img,
            (x_left, y_top),
            (x_right, y_bottom),
            COLOR_PRED, LINE_WIDTH)
        # MASK
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (x_left, y_top),
            (x_right, y_bottom),
            COLOR_PRED, -1)
        cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0, img)

        # 绘制文字背景 bbox上部
        bg_x_right = x_right if w_text < w else x_left + w_text
        h_bias = 30
        # 填充
        cv2.rectangle(
            img,
            (x_left, y_bottom),  # 左上角
            (bg_x_right, y_bottom + h_text + h_bias),  # 右下角
            COLOR_PRED, -1)
        # 边缘
        cv2.rectangle(
            img,
            (x_left, y_bottom),  # 左上角
            (bg_x_right, y_bottom + h_text + h_bias),  # 右下角
            COLOR_PRED, LINE_WIDTH)
        # 绘制文本
        cv2.putText(
            img,
            text,
            (x_left, y_bottom + h_text + 10),
            FONT, FONT_SCALE, COLOR_TEXT, 2)

    # INFO
    num_gt = len(anns)
    num_pred = len(preds)
    info = f"{num_gt} Ground {'Truth' if num_gt in [0, 1] else 'Truths'} In Total | {num_pred} {'Prediction' if num_pred in [0, 1] else 'Predictions'} Found"
    (w_text, h_text), _ = cv2.getTextSize(info, FONT, 3, 2)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.imshow(img)
    ax.text(
        50, 50, info,
        bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'),
        color='blue', va='top', ha='left')
    ax.axis('off')
    
    return fig


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
        axs[0].set_title('Learning Rate')
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
        axs[1].set_title('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlim(left=0)

        axs[2].plot(
            df_train.index,
            df_train['acc'],
            label='acc')
        axs[2].legend(frameon=True)
        axs[2].set_title('ACC')
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
        axs[3].set_title('coco/bbox_mAP')
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
        axs[0].set_title('Learning Rate')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Learning Rate')
        axs[0].set_xlim(left=0)

        axs[1].plot(df_train.index, df_train['loss'], label='loss')
        axs[1].plot(df_train.index, df_train['loss_cls'], label='loss_cls')
        axs[1].plot(df_train.index, df_train['loss_bbox'], label='loss_bbox')
        axs[1].legend(frameon=True)
        axs[1].set_title('Loss')
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
        axs[3].set_title('coco/bbox_mAP')
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
        axs[0].set_title('Learning Rate')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Learning Rate')
        axs[0].set_xlim(left=0)

        axs[1].plot(df_train.index, df_train['grad_norm'], label='grad_norm')
        axs[1].legend(frameon=True)
        axs[1].set_title('Gradient Norm')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Gradient Norm')
        axs[1].set_xlim(left=0)

        axs[2].plot(df_train.index, df_train['loss'], label='loss')
        axs[2].plot(df_train.index, df_train['loss_cls'], label='loss_cls')
        axs[2].plot(df_train.index, df_train['loss_conf'], label='loss_conf')
        axs[2].plot(df_train.index, df_train['loss_xy'], label='loss_xy')
        axs[2].plot(df_train.index, df_train['loss_wh'], label='loss_wh')
        axs[2].legend(frameon=True)
        axs[2].set_title('Loss')
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
        axs[3].set_title('coco/bbox_mAP')
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
