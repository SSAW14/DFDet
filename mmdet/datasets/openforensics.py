from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class OpenForensicsDataset(CocoDataset):

    CLASSES = ('Real', 'Fake')
