"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch as pt
import tqdm

from object_centric_bench.datum import DataLoader
from object_centric_bench.util_datum import draw_segmentation_np
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config


@pt.inference_mode()
def val_epoch(
    cfg, dataset_v, model, loss_fn_v, acc_fn_v, callback_v, is_viz=False, is_img=False
):
    pack = Config({})
    pack.dataset_v = dataset_v
    pack.model = model
    pack.loss_fn_v = loss_fn_v
    pack.acc_fn_v = acc_fn_v
    pack.callback_v = callback_v
    pack.epoch = 0

    pack2 = Config({})

    mean = pt.from_numpy(np.array(cfg.IMAGENET_MEAN, "float32"))
    std = pt.from_numpy(np.array(cfg.IMAGENET_STD, "float32"))
    cnt = 0

    pack.isval = True
    pack.model.eval()
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for i, batch in enumerate(tqdm.tqdm(pack.dataset_v)):
        pack.batch = {k: v.cuda() for k, v in batch.items()}

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn_v(**pack)
        pack.acc = pack.acc_fn_v(**pack)

        if is_viz:
            # mkdir
            save_dn = Path(cfg.name)
            if not Path(save_dn).exists():
                save_dn.mkdir(exist_ok=True)
            # read gt image and segment
            img_key = "image" if is_img else "video"
            imgs_gt = (  # image video
                (pack.batch[img_key] * std.cuda() + mean.cuda()).clip(0, 255).byte()
            )
            segs_gt = pack.batch["segment"]
            # read pd attent -> pd segment
            segs_pd = pack.output["segment"]
            # visualize gt image or video
            for img_gt, seg_gt, seg_pd in zip(imgs_gt, segs_gt, segs_pd):
                if is_img:
                    img_gt, seg_gt, seg_pd = [  # warp img as vid
                        _[None] for _ in (img_gt, seg_gt, seg_pd)
                    ]
                for tcnt, (igt, sgt, spd) in enumerate(zip(img_gt, seg_gt, seg_pd)):
                    igt = igt.permute(1, 2, 0).cpu().numpy()
                    igt = cv2.cvtColor(igt, cv2.COLOR_RGB2BGR)
                    sgt = sgt.cpu().numpy()
                    spd = spd.cpu().numpy()
                    save_path = save_dn / f"{cnt:06d}-{tcnt:06d}"
                    cv2.imwrite(f"{save_path}-i.png", igt)
                    cv2.imwrite(
                        f"{save_path}-s.png", draw_segmentation_np(igt, sgt, alpha=0.9)
                    )
                    cv2.imwrite(
                        f"{save_path}-p.png", draw_segmentation_np(igt, spd, alpha=0.9)
                    )
                cnt += 1

        [_.after_step(**pack) for _ in pack.callback_v]

    [_.after_epoch(**pack) for _ in pack.callback_v]

    for cb in pack.callback_v:
        if cb.__class__.__name__ == "AverageLog":
            pack2.log_info = cb.mean()
            break
        elif cb.__class__.__name__ == "HandleLog":
            pack2.log_info = cb.handle()
            break

    return pack2


def main(args):
    cfg_file = Path(args.cfg_file)
    data_path = Path(args.data_dir)
    ckpt_file = Path(args.ckpt_file)
    is_viz = args.is_viz
    is_img = args.is_img
    pt.backends.cudnn.benchmark = True

    assert cfg_file.name.endswith(".py")
    assert cfg_file.is_file()
    cfg_name = cfg_file.name.split(".")[0]
    cfg = Config.fromfile(cfg_file)
    cfg.name = cfg_name

    ## datum init

    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_path

    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
    )

    ## model init

    model = build_from_config(cfg.model)
    # print(model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    if ckpt_file:
        model.load(ckpt_file, None, verbose=False)
    if cfg.freez:
        model.freez(cfg.freez, verbose=False)

    model = model.cuda()
    # model.compile()

    ## learn init

    loss_fn_v = MetricWrap(**build_from_config(cfg.loss_fn_v))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))

    cfg.callback_v = [_ for _ in cfg.callback_v if _.type.__name__ != "SaveModel"]
    for cb in cfg.callback_v:
        if cb.type.__name__ in ["AverageLog", "HandleLog"]:
            cb.log_file = None
    callback_v = build_from_config(cfg.callback_v)

    ## do eval

    pack2 = val_epoch(
        cfg, dataload_v, model, loss_fn_v, acc_fn_v, callback_v, is_viz, is_img
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg_file",
        type=str,  # TODO XXX
        default="config-smoothsa/smoothsa_r-coco.py",
    )
    parser.add_argument(  # TODO XXX
        "--data_dir", type=str, default="/media/GeneralZ/Storage/Static/datasets"
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,  # TODO XXX
        default="archive-smoothsa/smoothsa_r-coco/42-0027.pth",
    )
    parser.add_argument(
        "--is_viz",
        type=bool,  # TODO XXX
        default=False,
    )
    parser.add_argument(
        "--is_img",  # image or video
        type=bool,  # TODO XXX
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
