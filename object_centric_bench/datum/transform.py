import random

from einops import rearrange
import numpy as np
import torch as pt
import torchvision.transforms.v2 as ptvt2

from ..utils import DictTool


class Lambda:

    def __init__(self, keys, func=lambda _: _):
        self.keys = keys
        if type(func) is str:
            func = eval(func)
        self.func = func

    def __call__(self, **pack: dict) -> dict:
        # pack = pack.copy()
        for key in self.keys:
            input = DictTool.getattr(pack, key)
            output = self.func(input)
            DictTool.setattr(pack, key, output)
        return pack


class Normalize:
    """Support any tensor shape, as long as mean and std broadcastable!"""

    def __init__(self, keys, mean=None, std=None):
        self.keys = keys
        self.mean = pt.from_numpy(np.array(mean, "float32"))
        self.std = pt.from_numpy(np.array(std, "float32"))

    def __call__(self, **pack: dict) -> dict:
        # pack = pack.copy()
        for key in self.keys:
            input = DictTool.getattr(pack, key)
            mean = input.mean() if self.mean is None else self.mean
            std = input.std() if self.std is None else self.std
            output = (input - mean) / std
            DictTool.setattr(pack, key, output)
        return pack


class Rearrange:
    """Can work as Flatten."""

    def __init__(self, keys, pattern, **kwds):
        self.keys = keys
        self.pattern = pattern
        self.kwds = kwds

    def __call__(self, **pack: dict) -> dict:
        # pack = pack.copy()
        for key in self.keys:
            input = DictTool.getattr(pack, key)
            output = rearrange(input, self.pattern, **self.kwds)
            DictTool.setattr(pack, key, output)
        return pack


class Clone:

    def __init__(self, keys, keys2):
        assert len(keys) == len(keys2)
        self.keys = keys
        self.keys2 = keys2

    def __call__(self, **pack: dict) -> dict:
        for key, key2 in zip(self.keys, self.keys2):
            input = DictTool.getattr(pack, key)
            output = input.clone()
            DictTool.setattr(pack, key2, output)
        return pack


class RandomFlip:
    """Flip tensor randomly along one of the given dimensions.
    Support bbox shape (..,c=4)."""

    def __init__(self, keys, dims: list, bbox_key=None, p=0.5):
        """
        - dims: dimensions to flip
        - bbox_key: l-t-r-b, both-side normalized; shape=(..,c=4)
        - prob: probability to flip
        """
        self.keys = keys
        self.bbox_key = bbox_key
        self.dims = dims
        self.p = p

    def __call__(self, **pack: dict) -> dict:
        if random.random() > self.p:
            return pack
        # pack = pack.copy()
        dim = random.choice(self.dims)
        for key in self.keys:
            input = DictTool.getattr(pack, key)
            output = input.flip(dim)
            DictTool.setattr(pack, key, output)
        if self.bbox_key:
            assert dim in [-2, -1]  # h, w
            bbox = DictTool.getattr(pack, self.bbox_key)  # ltrb
            bbox2 = bbox.clone()
            if dim == -2:  # height vertical t-b
                bbox2[..., 1::2] = 1 - bbox[..., 1::2]
            if dim == -1:  # width horizontal l-r
                bbox2[..., 0::2] = 1 - bbox[..., 0::2]
            DictTool.setattr(pack, self.bbox_key, bbox2)
        return pack


class Detach:
    """Detach keyed tensors from their graph."""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, **pack: dict) -> dict:
        # pack = pack.copy()
        for key in self.keys:
            input = DictTool.getattr(pack, key)
            output = input.detach()
            DictTool.setattr(pack, key, output)
        return pack


class TupleToNumber:
    """Convert tuple indexs to number indexes. Support tensor shape (..,g,..)."""

    def __init__(self, keys, groups, gdim):
        """
        - groups: sizes of all groups
        - gdim: which is the group dimension; will be eliminated after conversion
        """
        self.keys = keys
        self.groups = groups
        self.gdim = gdim

    def __call__(self, **pack: dict) -> dict:
        # pack = pack.copy()
        for key in self.keys:
            input = DictTool.getattr(pack, key)
            output = __class__.tuple_to_number(input, self.groups, self.gdim)
            DictTool.setattr(pack, key, output)
        return pack

    @staticmethod
    def tuple_to_number(tidx, groups: list, gdim=1):
        """
        tidx: shape=(b,g,..)
        """
        nidx = 0
        device = tidx.device
        base = 1
        for i, g in enumerate(groups):
            i = pt.from_numpy(np.array(i, "int64"), device=device)
            idx_g = tidx.index_select(gdim, i).squeeze(gdim)
            nidx += idx_g * base
            base *= g
        return nidx


INTERPOLATS = {_.value: _ for _ in ptvt2.InterpolationMode}


class Resize:
    """Support tensor shape (..,c,h,w). Can skip unnecessary resizing.
    ??? To support bounding box tensor=(..,4) ???
    """

    def __init__(self, keys, size, interp="bilinear", max_size=None, c=1):
        """
        - size: two-tuple height-width or int; resize along the short side if it is int
        - max_size: int; resize along the long side
        - c: 1 means tensor shape=(..,c,h,w); 0 means tensor=(..,h,w)
        """
        self.keys = keys
        self.interp = INTERPOLATS[interp]
        self.resize = ptvt2.Resize(
            size, self.interp, max_size, antialias=interp != "nearest-exact"
        )
        self.c = c  # input has c or not

    def __call__(self, **pack: dict) -> dict:
        # pack = pack.copy()
        for key in self.keys:
            input = DictTool.getattr(pack, key)
            if not self.c:
                input = input[..., None, :, :]  # (..,c=1,h,w)
            output = self.resize(input)
            if not self.c:
                output = output[..., 0, :, :]
            DictTool.setattr(pack, key, output)
        return pack


class RandomCrop:
    """Support tensor shape (..,h,w) and bbox shape (..,c=4).
    RandomResizedCrop can be achieved by combining RandomCrop(size=None) with Resize.
    Its scale is re-scaled in runtime by the maximum square crop of the original image,
    which is better than not (the original implementation).
    """

    def __init__(
        self,
        keys,
        size=None,
        scale=(0.75, 1.0),
        ratio=(3 / 4, 4 / 3),
        bbox_key=None,
        value=0,
    ):
        """
        - size: two-tuple height-width, int or None.
            If int then crop in square; if None then crop in by scale range
        - scale: area range of random crop; valid when size is None
        - ratio: aspect ratio range of random crop; valid when size is None
        - bbox_key: l-t-r-b, both-side normalized; shape=(..,c=4)
        - value: reset out-crop bbox to this value, not remove them
        """
        # https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/lib/preprocessing.py#L1039
        assert "flow" not in keys  # TODO XXX not support optical flow crop
        self.keys = keys
        self.size = size
        if size is not None:  # random crop by given size
            self.random_crop = ptvt2.RandomCrop(size)
        else:  # random crop by given scale range
            self.random_crop = None  # ptvt2.RandomResizedCrop([1, 1], scale, ratio)
        self.scale = scale  # re-scale
        self.ratio = ratio
        self.bbox_key = bbox_key
        self.value = value

    def __call__(self, **pack: dict) -> dict:
        # pack = pack.copy()
        image = DictTool.getattr(pack, self.keys[0])
        h0, w0 = image.shape[-2:]
        if self.size is None:
            h0, w0 = image.shape[-2:]
            scale_factor = min(h0, w0) ** 2 / (h0 * w0)  # re-scale
            scale2 = [_ * scale_factor for _ in self.scale]
            self.random_crop = ptvt2.RandomResizedCrop([1, 1], scale2, self.ratio)
        params = self.random_crop.make_params(image)
        t, l, h, w = [params[_] for _ in ["top", "left", "height", "width"]]
        for key in self.keys:
            input = DictTool.getattr(pack, key)
            output = input[..., t : t + h, l : l + w]
            DictTool.setattr(pack, key, output)
        if self.bbox_key:
            bbox = DictTool.getattr(pack, self.bbox_key)  # ltrb
            bbox2 = __class__.crop_bbox(bbox, h0, w0, t, l, h, w, self.value)
            DictTool.setattr(pack, self.bbox_key, bbox2)
        return pack

    @staticmethod
    def crop_bbox(bbox: pt.Tensor, h0, w0, t, l, h, w, value=0) -> pt.Tensor:
        """suppose bbox l-t-r-b is normalized; only zero out-crop bboxs, not remove them
        https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/lib/preprocessing.py#L76

        - bbox: shape=(..,c=4). both-side normalized, not short-side normalized
        - h0, w0, t, l, h, w: absolute coordinates
        """
        # Transform the box coordinates.
        a = pt.tensor([w0, h0], dtype=pt.float32)
        b = pt.tensor([l, t], dtype=pt.float32)
        c = pt.tensor([w, h], dtype=pt.float32)
        bbox = ((bbox.unflatten(-1, [2, 2]) * a - b) / c).flatten(-2)
        # Filter the valid boxes.
        bbox = bbox.clip(0, 1)
        cond = (bbox[..., 2:] - bbox[..., :2] <= 0).any(-1)
        bbox[cond] = value
        return bbox


class CenterCrop:
    """Support tensor shape (..,h,w) and bbox shape (..,c=4) lrtb."""

    def __init__(self, keys, size: list = None, bbox_key=None, value=0):
        """
        - size: two-tuple height-width or int.
            if int then crop in square; if None then crop in max square
        - bbox_key: l-t-r-b, both-side normalized; shape=(..,c=4)
        - value: reset out-crop bbox to this value, not remove them
        """
        # https://github.com/google-research/slot-attention-video/blob/ba8f15ee19472c6f9425c9647daf87910f17b605/savi/lib/preprocessing.py#L1039
        assert "flow" not in keys  # TODO XXX not support optical flow crop
        self.keys = keys
        assert (
            isinstance(size, int)
            or (isinstance(size, (list, tuple)) and len(size) == 2)
            or size is None
        )
        self.size = [size] * 2 if isinstance(size, int) else size
        # self.center_crop = ptvt2.CenterCrop(size)
        self.bbox_key = bbox_key
        self.value = value

    def __call__(self, **pack: dict) -> dict:
        # pack = pack.copy()
        image = DictTool.getattr(pack, self.keys[0])
        h0, w0 = image.shape[-2:]
        if self.size is None:
            self_size = [min(h0, w0)] * 2
        else:
            self_size = self.size
        t, l, b, r = __class__.calc_params(h0, w0, self_size)
        for key in self.keys:
            input = DictTool.getattr(pack, key)
            output = input[..., t:b, l:r]
            DictTool.setattr(pack, key, output)
        if self.bbox_key:
            bbox = DictTool.getattr(pack, self.bbox_key)
            bbox2 = RandomCrop.crop_bbox(
                bbox, h0, w0, t, l, self_size[0], self_size[1], self.value
            )
            DictTool.setattr(pack, self.bbox_key, bbox2)
        return pack

    @staticmethod
    def calc_params(h, w, size):
        assert size[0] <= h and size[1] <= w
        t = (h - size[0]) // 2
        l = (w - size[1]) // 2
        b = t + size[0]
        r = l + size[1]
        return t, l, b, r
