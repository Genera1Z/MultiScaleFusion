"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""
from argparse import ArgumentParser
from pathlib import Path
import os
import random
import shutil
import time

import numpy as np
import torch as pt
import tqdm

from object_centric_bench.datum import DataLoader
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # reproducibility
pt._dynamo.config.suppress_errors = True  # one_hot, interplolate


def train_epoch(pack):
    t0 = time.time()
    pack.model.train()
    pack.isval = False
    [_.before_epoch(**pack) for _ in pack.callback_t]

    for batch in tqdm.tqdm(pack.dataset_t):
        if pack.step_count + 1 > pack.total_step:
            break
        pack.batch = batch

        [_.before_step(**pack) for _ in pack.callback_t]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)
            [_.after_forward(**pack) for _ in pack.callback_t]
            pack.loss = pack.loss_fn_t(**pack)  # {k:(loss,valid),..}
        # for pack.loss/acc
        # - value: dtype=float, shape=(b,). but actually (b=1,) for loss
        # - valid: dtype=bool, shape=(b,). but actually (b=1,) for loss
        pack.acc = pack.acc_fn_t(**pack)  # in autocast may cause inf

        flag = True
        for loss_i, valid_i in pack.loss.values():
            if valid_i.sum() == 0:
                print("no valid sample in batch")  # then will not back prop
                flag = False
                break

        if flag:
            with pt.autocast("cuda", enabled=True):
                loss_mean_sum = sum(_l[_v].mean() for _l, _v in pack.loss.values())

            pack.optimiz.zero_grad()
            pack.optimiz.gscale.scale(loss_mean_sum).backward()
            if pack.optimiz.gclip is not None:
                pack.optimiz.gscale.unscale_(pack.optimiz)
                pack.optimiz.gclip(pack.model.parameters())
            pack.optimiz.gscale.step(pack.optimiz)
            pack.optimiz.gscale.update()

        [_.after_step(**pack) for _ in pack.callback_t]

        pack.step_count += 1

    [_.after_epoch(**pack) for _ in pack.callback_t]
    print("b/s:", len(pack.dataset_t) / (time.time() - t0))


@pt.inference_mode()
def val_epoch(pack):
    pack.model.eval()
    pack.isval = True
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for batch in pack.dataset_v:
        pack.batch = batch

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn_v(**pack)
        pack.acc = pack.acc_fn_v(**pack)  # in autocast may cause inf

        [_.after_step(**pack) for _ in pack.callback_v]

    [_.after_epoch(**pack) for _ in pack.callback_v]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)


def main(args):
    seed = args.seed
    cfg_file = Path(args.cfg_file)
    data_path = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    ckpt_file = args.ckpt_file
    print(args)

    if ckpt_file is None:
        pass
    elif isinstance(ckpt_file, str):
        ckpt_file = Path(ckpt_file)
    else:
        assert isinstance(ckpt_file, (list, tuple))
        ckpt_file = [Path(_) for _ in ckpt_file]

    assert cfg_file.name.endswith(".py")
    assert cfg_file.is_file()
    cfg_name = cfg_file.name.split(".")[0]
    cfg = Config.fromfile(cfg_file)

    save_path = Path(save_dir) / cfg_name / str(seed)
    save_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_file, save_path.parent)

    set_seed(seed)  # for reproducibility
    pt.backends.cudnn.benchmark = False  # XXX True: faster but stochastic
    pt.backends.cudnn.deterministic = True  # for cuda devices
    pt.use_deterministic_algorithms(True, warn_only=True)  # for all devices

    ## datum init

    work_init_fn = lambda _: set_seed(seed)  # for reproducibility
    rng = pt.Generator()
    rng.manual_seed(seed)

    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_path

    dataset_t = build_from_config(cfg.dataset_t)
    dataload_t = DataLoader(
        dataset_t,
        cfg.batch_size_t,
        shuffle=True,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_t),
        pin_memory=True,
        worker_init_fn=work_init_fn,
        generator=rng,
    )
    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
        worker_init_fn=work_init_fn,
        generator=rng,
    )

    ## model init

    model = build_from_config(cfg.model)
    print(model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    if ckpt_file:
        if isinstance(ckpt_file, (list, tuple)):
            if len(ckpt_file) == 1:
                cfg.ckpt_map = [cfg.ckpt_map]
            assert len(ckpt_file) == len(cfg.ckpt_map)
            [model.load(_, __) for _, __ in zip(ckpt_file, cfg.ckpt_map)]
        else:
            model.load(ckpt_file, cfg.ckpt_map)
    if cfg.freez:
        model.freez(cfg.freez)

    model = model.cuda()
    model.compile()  # TODO XXX comment this for debugging

    ## learn init

    if cfg.param_groups:
        cfg.optimiz.params = model.group_params(**cfg.param_groups)
    else:
        cfg.optimiz.params = model.parameters()
    optimiz = build_from_config(cfg.optimiz)
    optimiz.gscale = build_from_config(cfg.gscale)
    optimiz.gclip = build_from_config(cfg.gclip)

    loss_fn_t = MetricWrap(**build_from_config(cfg.loss_fn_t))
    loss_fn_v = MetricWrap(**build_from_config(cfg.loss_fn_v))
    # loss_fn.compile()  # sometimes nan ???
    acc_fn_t = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_t))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))
    # acc_fn.compile()  # sometimes nan ???

    for cb in cfg.callback_t + cfg.callback_v:
        if cb.type.__name__ in ["AverageLog", "HandleLog"]:
            cb.log_file = f"{save_path}.txt"
        elif cb.type.__name__ == "SaveModel":
            cb.save_dir = save_path
    callback_t = build_from_config(cfg.callback_t)
    callback_v = build_from_config(cfg.callback_v)

    ## train loop

    pack = Config({})
    pack.dataset_t = dataload_t
    pack.dataset_v = dataload_v
    pack.model = model
    pack.optimiz = optimiz
    pack.loss_fn_t = loss_fn_t
    pack.loss_fn_v = loss_fn_v
    pack.acc_fn_t = acc_fn_t
    pack.acc_fn_v = acc_fn_v
    pack.callback_t = callback_t
    pack.callback_v = callback_v
    pack.total_step = cfg.total_step
    pack.val_interval = cfg.val_interval

    epoch_count = 0
    epoch_count_v = 0
    pack.step_count = 0
    [_.before_train(**pack) for _ in pack.callback_t]

    while pack.step_count < pack.total_step:
        pack.epoch = epoch_count
        pt.cuda.empty_cache()
        train_epoch(pack)

        flag1 = pack.step_count >= (epoch_count_v + 1) * pack.val_interval
        flag2 = pack.step_count >= pack.total_step
        if flag1 or flag2:
            pt.cuda.empty_cache()
            val_epoch(pack)
            epoch_count_v += 1

        epoch_count += 1

    assert pack.step_count == pack.total_step
    [_.after_train(**pack) for _ in pack.callback_t]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,  # TODO XXX
    )
    parser.add_argument(
        "--cfg_file",
        type=str,  # TODO XXX
        default="config-slotdiffusion/vqvae-coco-c4-msf.py",
    )
    parser.add_argument(  # TODO XXX
        "--data_dir", type=str, default="/media/GeneralZ/Storage/Static/datasets"
    )
    parser.add_argument("--save_dir", type=str, default="save")
    parser.add_argument(
        "--ckpt_file",
        type=str,
        nargs="+",
        # default="archive-randsfq-tsim/randsfq_r-ytvis/42-0155.pth",
        # default=[
        #     "archive-hwm/vqvae-ytvis-c256/best.pth",
        #     "archive-hwm/spott_r_randar-ytvis/best.pth",
        # ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
