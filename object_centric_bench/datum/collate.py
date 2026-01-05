"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

import torch.utils.data as ptud

from .transform import PadTo1
from ..util import DictTool


class ClPadToMax1:
    """
    Pad a dimension of a list of tensors to the max size.
    A counterpart to ``DefaultCollate``.
    Not composable by ``Compose``.

    TODO XXX support ``keys=["bbox", "clazz"], pkeys=["bmask", "cmask"]``
    """

    def __init__(
        self, keys: list, dims: list, plus: list = None, mode="right", value=0
    ):
        assert len(keys) == len(dims)
        self.keys = keys
        self.dims = dims
        if plus is not None:
            assert len(keys) == len(plus)
        self.plus = plus  # 1 for padding ``bbox`` as its background
        self.mode = mode
        self.value = value

    def __call__(self, samples: list) -> list:
        for i, (key, dim) in enumerate(zip(self.keys, self.dims)):
            inputs = [DictTool.getattr(_, key) for _ in samples]
            size = max(_.size(dim) for _ in inputs)
            if self.plus is not None:
                size += self.plus[i]

            for sample, input in zip(samples, inputs):
                left, right = PadTo1.calc_padding(size, input.size(dim), self.mode)
                output = PadTo1.pad1(input, dim, left, right, self.value)
                DictTool.setattr(sample, key, output)

        return samples
        return ptud.default_collate(samples)


class ClPadTo1:
    """Collate Pad To 1 dimension."""

    def __init__(self, keys: list, dims: list, num: list = None, mode="right", value=0):
        assert len(keys) == len(dims)
        self.keys = keys
        self.dims = dims
        if num is not None:  # None: auto pad to max ``dim`` size
            assert len(keys) == len(num)
        self.num = num
        self.mode = mode
        self.value = value

    def __call__(self, samples: list) -> list:
        for i, (key, dim) in enumerate(zip(self.keys, self.dims)):
            inputs = [DictTool.getattr(_, key) for _ in samples]
            if self.num is None:
                numi = max(_.size(dim) for _ in inputs)
            else:
                numi = self.num[i]

            for sample, input in zip(samples, inputs):
                left, right = PadTo1.calc_padding(numi, input.size(dim), self.mode)
                output = PadTo1.pad1(input, dim, left, right, self.value)
                DictTool.setattr(sample, key, output)

        return samples
        return ptud.default_collate(samples)


class DefaultCollate:
    """
    default_collate from PyTorch"""

    def __call__(self, samples: list) -> dict:
        return ptud.default_collate(samples)
