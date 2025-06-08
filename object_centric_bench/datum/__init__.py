from ..utils import register_module
from .dataset import DataLoader
from .dataset_coco import MSCOCO
from .transform import (
    Lambda,
    Normalize,
    Rearrange,
    Clone,
    RandomFlip,
    RandomCrop,
    CenterCrop,
    Resize,
    TupleToNumber,
    Detach,
)

[register_module(_) for _ in locals().values() if isinstance(_, type)]
