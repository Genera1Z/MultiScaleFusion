"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""
from abc import ABC, abstractmethod

import lpips
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from ..util import DictTool
from ..util_learn import intersection_over_union, hungarian_matching


class MetricWrap(nn.Module):
    """"""

    def __init__(self, detach=False, **metrics):
        super().__init__()
        self.detach = detach
        self.metrics = metrics

    def forward(self, **pack: dict) -> dict:
        if self.detach:
            with pt.inference_mode():
                return self._forward(**pack)
        else:
            return self._forward(**pack)

    def _forward(self, **pack: dict) -> dict:
        metrics = {}
        for key, value in self.metrics.items():
            kwds = {t: DictTool.getattr(pack, s) for t, s in value["map"].items()}
            # if self.detach:  # inference_mode has done this
            #     kwds = {k: v.detach() for k, v in kwds.items()}
            if "transform" in value:
                kwds = value["transform"](**kwds)
            assert isinstance(value["metric"], Metric)
            metric = value["metric"](**kwds)  # (loss/acc, valid)
            if "weight" in value:
                metric = (metric[0] * value["weight"], metric[1])
            metrics[key] = metric
        return metrics


class Metric(ABC, nn.Module):
    """
    mean all or specified dimensions; always keep batch dimension
    """

    def __init__(self, mean=()):
        """
        - mean: only support ``mean``; other operations like ``sum`` are seldomly used.
            - None: do nothing
            - len(mean) == 0: mean all non-batch/first dimensions
            - len(mean) > 0: mean the specified dimensions
        """
        super().__init__()
        assert mean is None or isinstance(mean, (list, tuple))
        if isinstance(mean, (list, tuple)) and len(mean) > 0:
            assert 0 not in mean  # batch/first dimension should not be included
        self.mean = mean

    @abstractmethod
    def forward(self, *args, **kwds) -> tuple:
        ...
        metric, valid = self.finaliz(...)
        return metric, valid  # loss/acc (b,..), valid (b,)

    def finaliz(self, metric, valid=None) -> tuple:
        """mean ``metric`` along dimensions ``self.mean``; flag ``valid`` samples

        - metric: shape=(b,..); dtype=float
        - valid: shape=(b,); dtype=bool
        """
        if self.mean is None:
            metric2 = metric  # (b,..)
        elif len(self.mean) == 0:
            if metric.ndim > 1:
                metric2 = metric.flatten(1).mean(1)  # (b,)
            else:
                metric2 = metric
        else:
            metric2 = metric.mean(self.mean)  # (b,..)
        if valid is not None:
            valid2 = valid
        else:
            valid2 = pt.ones(metric.size(0), dtype=pt.bool, device=metric.device)
        return metric2, valid2


####


class CrossEntropyLoss(Metric):
    """``nn.CrossEntropyLoss``."""

    def forward(self, input, target):
        """
        - input: shape=(b,c,..), dtype=float
        - target: shape=(b,..), dtype=int64;
            or shape=(b,c,..), dtype=float
        """
        # loss = ptnf.cross_entropy(input, target, reduction="none")  # (b,..)
        loss = ptnf.cross_entropy(input, target)[None]  # (b=1,)
        return self.finaliz(loss)  # (b,) (b,)


class CrossEntropyLossGrouped(CrossEntropyLoss):
    """"""

    def __init__(self, groups, mean=()):
        super().__init__(mean)
        self.groups = groups

    def forward(self, input, target):
        """
        input: shape=(b,g*c,..), dtype=float
        target: shape=(b,g,..), dtype=int64
        """
        assert input.ndim == target.ndim
        start = 0
        loss = []
        for g, gc in enumerate(self.groups):
            end = start + gc
            ce = super().forward(input[:, start:end], target[:, g])  # (b,) (b,)
            start = end
            loss.append(ce)
        assert end == input.size(1)
        loss_sum = sum(_[0] for _ in loss)  # (b,..)
        return self.finaliz(loss_sum)  # (b,) (b,)


class MSELoss(Metric):
    """``nn.MSELoss``."""

    def forward(self, input, target):
        assert input.ndim == target.ndim >= 1
        # TODO XXX why outside-mean is no better than builtin-mean ??? TODO XXX
        # loss = ptnf.mse_loss(input, target, reduction="none")  # (b,..)
        loss = ptnf.mse_loss(input, target)[None]  # (b=1,)
        return self.finaliz(loss)  # (b,) (b,)


class LPIPSLoss(Metric):
    """"""

    def __init__(self, net="vgg", mean=()):
        super().__init__(mean)
        self.lpips = lpips.LPIPS(pretrained=True, net=net, eval_mode=True)
        for p in self.lpips.parameters():
            p.requires_grad = False
        self.lpips.compile()
        # self.lpips = pt.quantization.quantize_dynamic(self.lpips)  # slow

    def forward(self, input, target):
        """
        input: shape=(b,c,h,w), dtype=float
        target: shape=(b,c,h,w), dtype=float
        """
        assert input.ndim == target.ndim == 4
        # assert input.dtype == target.dtype == pt.float
        self.lpips.to(input.device)  # to the same device, won't repeat once done
        # lpips = self.lpips(target, input)  # (b,c,h,w)
        lpips = self.lpips(target, input).mean()[None]  # (b=1,)
        return self.finaliz(lpips)  # (b,) (b,)


####


class ARI(Metric):
    """"""

    def __init__(self, skip=[], mean=()):
        super().__init__(mean)
        self.skip = pt.from_numpy(np.array(skip, "int64"))

    def forward(self, input, target):
        """
        - input: shape=(b,n,c), onehot segment
        - target: shape=(b,n,d), onehot segment
        """
        assert input.ndim == target.ndim == 3
        if self.skip.numel():
            self.skip = self.skip.cuda()
            target = __class__.skip_segment(target, self.skip)
        ari = __class__.adjusted_rand_index(input, target)  # (b,)
        valid = ARI.find_valid(target)  # (b,)
        return self.finaliz(ari, valid)  # (b,) (b,)

    @pt.inference_mode()
    @staticmethod
    def adjusted_rand_index(oh_pd, oh_gt):
        """
        https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        # the following boolean op is even slower than floating point op
        # N = (oh_gt[:, :, :, None] & oh_pd[:, :, None, :]).sum(1)  # (b,d,c)  # oom
        # long, will be auto-cast to float  # int overflow
        N = pt.einsum("bnc,bnd->bcd", oh_pd.double(), oh_gt.double()).long()  # (b,c,d)
        # the following fixed point op shows negligible speedup vesus floating point op
        A = N.sum(1)  # (b,d)
        B = N.sum(2)  # (b,c)
        num = A.sum(1)  # (b,)

        ridx = (N * (N - 1)).sum([1, 2])  # (b,)
        aidx = (A * (A - 1)).sum(1)  # (b,)
        bidx = (B * (B - 1)).sum(1)  # (b,)

        expect_ridx = (aidx * bidx) / (num * (num - 1)).clip(1)
        max_ridx = (aidx + bidx) / 2
        denominat = max_ridx - expect_ridx
        ari = (ridx - expect_ridx) / denominat  # (b,)

        # two cases ``denominat == 0``
        # - both pd and gt assign all pixels to a single cluster
        #    max_ridx == expect_ridx == ridx == num * (num - 1)
        # - both pd and gt assign max 1 point to each cluster
        #    max_ridx == expect_ridx == ridx == 0
        # we want the ARI score to be 1.0
        ari[denominat == 0] = 1
        return ari

    @pt.inference_mode()
    @staticmethod
    def skip_segment(oh_gt, skip_idx):
        """
        - oh_gt: shape=(b,n,d), onehot segment
        - skip_idx: shape=(d,), dtype=long
        """
        b, n, d = oh_gt.shape
        arange = pt.arange(d, dtype=skip_idx.dtype, device=oh_gt.device)
        mask = ~pt.isin(arange, skip_idx)
        return oh_gt[:, :, mask]

    @pt.inference_mode()
    @staticmethod
    def find_valid(oh_gt):
        assert oh_gt.ndim == 3
        valid = oh_gt.flatten(1).any(1)
        return valid


class mBO(Metric):
    """"""

    def __init__(self, skip=[], mean=()):
        super().__init__(mean)
        self.skip = pt.from_numpy(np.array(skip, "int64"))

    def forward(self, input, target):
        """
        - input: shape=(b,n,c), onehot segment
        - target: shape=(b,n,d), onehot segment
        """
        assert input.ndim == target.ndim == 3
        if self.skip.numel():
            self.skip = self.skip.cuda()
            target = ARI.skip_segment(target, self.skip)
        mbo = __class__.mean_best_overlap(input, target)  # (b,)
        valid = ARI.find_valid(target)  # (b,)
        return self.finaliz(mbo, valid)  # (b,) (b,)

    @pt.inference_mode()
    @staticmethod
    def mean_best_overlap(oh_pd, oh_gt):
        """
        https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, idx = iou_all.max(1)  # (b,d)
        num_gt = oh_gt.any(1).sum(1)  # (b,)
        return iou.sum(1) / num_gt  # (b,)


class mIoU(Metric):
    """"""

    def __init__(self, skip=[], mean=()):
        super().__init__(mean)
        self.skip = pt.from_numpy(np.array(skip, "int64"))

    def forward(self, input, target):
        """
        input: shape=(b,n,c), onehot segment
        target: shape=(b,n,d), onehot segment
        """
        assert input.ndim == target.ndim == 3
        if self.skip.numel():
            self.skip = self.skip.cuda()
            target = ARI.skip_segment(target, self.skip)
        miou = __class__.mean_intersection_over_union(input, target)  # (b,)
        valid = ARI.find_valid(target)  # (b,)
        return self.finaliz(miou, valid)  # (b,) (b,)

    @pt.inference_mode()
    @staticmethod
    def mean_intersection_over_union(oh_pd, oh_gt):
        """
        https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, idx = hungarian_matching(iou_all, maximize=True)  # (b,d)
        num_gt = oh_gt.any(1).sum(1)  # (b,)
        return iou.sum(1) / num_gt  # (b,)
