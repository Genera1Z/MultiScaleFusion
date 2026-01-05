"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

import torch as pt
import torch.nn.functional as ptnf


@pt.inference_mode()
def interpolat_argmax_attent(attent, size, mode="bilinear"):
    """Already optimized with PyTorch inference mode.

    - attent: shape=(b,..,s,h,w), dtype=float
    - segment: shape=(b,..,h,w), dtype=int; index segment
    """
    shape0 = attent.shape[:-3]
    attent_ = attent.flatten(0, -4)  # (b*..,s,h,w)
    attent_ = ptnf.interpolate(attent_, size=size, mode=mode)
    segment_ = attent_.argmax(1).byte()  # (b*..,h,w)
    segment = segment_.unflatten(0, shape0)
    return segment
