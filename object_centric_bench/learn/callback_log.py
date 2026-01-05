"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

from pathlib import Path
import json

import numpy as np
import torch as pt

from .callback import Callback


class AverageLog(Callback):
    """"""

    def __init__(
        self, log_file=None, epoch_key="epoch", loss_key="loss", acc_key="acc"
    ):
        super().__init__()
        self.log_file = log_file
        self.epoch_key = epoch_key
        self.loss_key = loss_key
        self.acc_key = acc_key
        self.idx = None
        self.state_dict = {}

    @pt.inference_mode()
    def index(self, epoch, isval):
        self.idx = f"{epoch}" if not isval else f"{epoch}/val"
        self.state_dict.clear()

    @pt.inference_mode()
    def append(self, loss, acc):
        for k, v in {**loss, **acc}.items():
            assert len(v) == 2  # (loss/acc,valid)
            v = tuple([_.detach().cpu().numpy() for _ in v])
            if k in self.state_dict:
                self.state_dict[k].append(v)
            else:
                self.state_dict[k] = [v]

    @pt.inference_mode()
    def mean(self):
        avg_dict = {}
        for k, v in self.state_dict.items():
            metric, valid = list(zip(*v))
            metric = np.concatenate(metric, 0)  # concat all batches  # (b,..)
            valid = np.concatenate(valid, 0)  # (b,)
            metric2 = metric[valid]  # keep the valid ones  # (?,..)
            v2 = np.mean(metric2, 0)  # .round(8) can cause nan
            avg_dict[k] = v2.tolist()
        if self.log_file:
            __class__.save(self.idx, avg_dict, self.log_file)
        print(self.idx, avg_dict)
        return avg_dict

    @pt.inference_mode()
    @staticmethod
    def save(key, avg_dict, log_file):
        line = json.dumps({key: avg_dict})
        with open(log_file, "a") as f:
            f.write(line + "\n")

    @pt.inference_mode()
    def before_epoch(self, isval, **pack):
        epoch = pack[self.epoch_key]
        self.index(epoch, isval)
        return pack

    @pt.inference_mode()
    def after_step(self, **pack):
        loss = pack[self.loss_key]
        acc = pack[self.acc_key]
        self.append(loss, acc)
        return pack

    @pt.inference_mode()
    def after_epoch(self, **pack):
        self.mean()
        return pack


class SaveModel(Callback):
    """"""

    def __init__(
        self,
        save_dir=None,
        since_step=0,
        weights_only=True,
        key=r".*",
        epoch_key="epoch",
        step_count_key="step_count",
        model_key="model",
    ):
        super().__init__()
        self.save_dir = save_dir
        self.since_step = since_step  # self.after_step is taken
        self.weights_only = weights_only
        self.key = key
        self.epoch_key = epoch_key
        self.step_count_key = step_count_key
        self.model_key = model_key

    @pt.inference_mode()
    def __call__(self, epoch, step_count, model):
        if step_count >= self.since_step:
            save_file = Path(self.save_dir) / f"{epoch:04d}.pth"
            model.save(save_file, self.weights_only, self.key)

    @pt.inference_mode()
    def after_epoch(self, **pack):
        epoch = pack[self.epoch_key]
        step_count = pack[self.step_count_key]
        model = pack[self.model_key]
        self.__call__(epoch, step_count, model)
        return pack
