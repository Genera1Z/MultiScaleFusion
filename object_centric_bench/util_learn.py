"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

from scipy.optimize import linear_sum_assignment
import numpy as np
import torch as pt


@pt.inference_mode()
def intersection_over_union(oh_pd, oh_gt):
    """
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    oh_pd: shape=(b,n,c), dtype=bool, onehot segment
    oh_gt: shape=(b,n,d), dtype=bool, onehot segment
    return: shape=(b,c,d), dtype=float
    """
    # the following boolean op is even slower than floating point op
    # double -> float
    intersect = pt.einsum("bnc,bnd->bcd", oh_pd.double(), oh_gt.double()).float()
    # intersect = (oh_pd[:, :, :, None] & oh_gt[:, :, None, :]).sum(1)  # oom
    # long, will be auto-cast to float
    union = oh_pd.sum(1)[:, :, None] + oh_gt.sum(1)[:, None, :] - intersect
    # union = (oh_pd[:, :, :, None] | oh_gt[:, :, None, :]).sum(1)  # oom
    iou = intersect / union

    invalid = union == 0
    iou[invalid] = 0
    return iou


@pt.inference_mode()
def hungarian_matching(iou_all, maximize=True):
    """
    https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

    iou_all: shape=(b,c,d), dtype=float32
    """
    iou_all_ = iou_all.detach().cpu().contiguous().numpy()
    # print(iou_all_.shape)
    rcidx = list(map(lambda _: linear_sum_assignment(_, maximize=maximize), iou_all_))

    iou = list(map(lambda t, i: t[i[0], i[1]], iou_all_, rcidx))

    # ``pt.from_numpy+np.array`` faster than ``pt.as_tensor``
    iou = pt.from_numpy(np.array(iou, "single")).to(iou_all.device)
    rcidx = pt.from_numpy(np.array(rcidx, "long")).to(iou_all.device)
    rcidx = rcidx.permute(0, 2, 1)  # (b,2,min(c,d)) -> (b,min(c,d),2)
    # print(rcidx.min().item(), rcidx.max().item())
    # print(rcidx.cpu().numpy().tolist())
    return iou, rcidx  # (b,min(c,d)) (b,min(c,d),2)
