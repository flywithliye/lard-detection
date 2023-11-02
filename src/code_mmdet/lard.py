
from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class LardDataset(CocoDataset):
    """Dataset for LARD."""

    METAINFO = {
        'classes':
        ('runway'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }
