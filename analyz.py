from collections import namedtuple
from pathlib import Path
import json
import shutil

from einops import rearrange
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torchvision.models as ptvm

from object_centric_bench.datum import Resize
from object_centric_bench.datum.utils import (
    draw_segmentation_np,
    generate_spectrum_colors,
)
from object_centric_bench.model import *
from object_centric_bench.utils import Config, build_from_config


DataFrame = namedtuple("DataFrame", ["data", "idxs", "cols"])


def pick_columns(log_lines, col_keys, mode="train"):
    idxs = []
    data = []

    for log_line in log_lines:
        log_dict = json.loads(log_line.strip())
        assert len(log_dict.keys()) == 1
        idx0 = list(log_dict.keys())[0]

        if mode == "val" and "val" in idx0:
            idx = int(idx0.split("/")[0])
        elif mode == "train" and "val" not in idx0:
            idx = int(idx0)
        else:
            assert (mode == "train" and "val" in idx0) or (
                mode == "val" and "val" not in idx0
            )
            continue

        value = [log_dict[idx0].get(_, np.nan) for _ in col_keys]

        idxs.append(idx)
        data.append(value)

    record = DataFrame(
        data=np.array(data),  # (i,c,..)
        idxs=idxs,
        cols=[f"{_}-{mode}" for _ in col_keys],
    )
    return record


def main_visualiz_log():
    base_dir = "save"
    groups0 = [
        # "vqdino-clevrtex-c256",
        #"vqdino-coco-c256",
        #"vqdino-voc-c256",
        #"vqdino-movi_d-c256",
        # "vqdino-clevrtex-c4",
        #"vqdino-coco-c4",
        #"vqdino-voc-c4",
        #"vqdino-ytvis-c256",
        #"vqvae-coco-c256-gdr",
        "vqvae-coco-c256-msf",
        #"vqvae-coco-c4-gdr",
        "vqvae-coco-c4-msf",
        # "vqvae-clevrtex-c256",
        # "vqvae-coco-c256",
        # "vqvae-voc-c256",
        # "vqvae-movi_d-c256",
    ]
    groups1 = [
        # "dinosaur_r-clevrtex",
        # "dinosaur_r-coco",
        # "dinosaur_r-voc",
        # "slate_r_vqvae-clevrtex",
        # "slate_r_vqvae-coco",
        # "slate_r_vqvae-voc",
        # "steve_c_vqvae-movi_d",
        # "slotdiffusion_r_vqvae-clevrtex",
        # "slotdiffusion_r_vqvae-coco",
        # "slotdiffusion_r_vqvae-voc",
        #"slate_r_vqvae-coco-gdr",
        "slate_r_vqvae-coco-msf",
        #"slotdiffusion_r_vqvae-coco-gdr",
        "slotdiffusion_r_vqvae-coco-msf",
        # "vqdino_tfd_r-clevrtex",
        # "vqdino_tfd_r-coco",
        # "vqdino_tfd_r-voc",
        # "vqdino_tfdt_c-movi_d",
        # "vqdino_mlp_r-clevrtex",
        # "vqdino_mlp_r-coco",
        # "vqdino_mlp_r-voc",
        # "vqdino_dfz_r-clevrtex",
        # "vqdino_dfz_r-coco",
        # "vqdino_dfz_r-voc",
        # "vqdino_smdt_r-ytvis",
        # "spott_r_randar-ytvis",
    ]
    groups = groups0
    # ckeys, akeys, sign = ["recon", "align", "commit", "lpips"], [0, 3], -1
    # ckeys, akeys, sign = ["recon", "align", "commit"], [0], -1
    ckeys, akeys, sign = ["recon0", "align0", "commit0"], [0], -1
    # ckeys, akeys, sign = ["align"], [0], -1
    #ckeys, akeys, sign = ["ari", "ari_fg", "mbo", "miou"], [1, 2, 3], 1

    copy_ckpt = 1  # TODO XXX False

    results = {}
    for group in groups:
        result_g = {}

        log_files = list(Path(base_dir).glob(f"{group}/*.txt"))

        # pick columns from files train val and merge

        trials_t = []
        trials_v = []

        for log_file in log_files:
            with open(log_file, "r") as f:
                log_content = f.readlines()
            trial_t = pick_columns(log_content, ckeys, "train")  # (epoch,metric,..)
            trial_v = pick_columns(log_content, ckeys, "val")

            trials_t.append(trial_t)
            trials_v.append(trial_v)

        points_t = np.stack([_.data for _ in trials_t], axis=2)  # (e,m,t,..)
        points_v = np.stack([_.data for _ in trials_v], axis=2)

        result_g["trials_t"] = DataFrame(points_t, trial_t.idxs, trial_t.cols)
        result_g["trials_v"] = DataFrame(points_v, trial_v.idxs, trial_v.cols)

        # calculate mean and std at every t over all b

        smean_t = np.mean(points_t, axis=2)  # (e,m,..)
        sstd_t = np.std(points_t, axis=2)
        stripe_t = np.stack([smean_t, sstd_t], axis=2)  # (e,m,2,..)

        smean_v = np.mean(points_v, axis=2)
        sstd_v = np.std(points_v, axis=2)
        stripe_v = np.stack([smean_v, sstd_v], axis=2)

        result_g["stripe_t"] = DataFrame(stripe_t, trial_t.idxs, trial_t.cols)
        result_g["stripe_v"] = DataFrame(stripe_v, trial_v.idxs, trial_v.cols)

        # find best over all t at every b

        # calculate mean and std over all b

        def mean_std_of_best(points, akeys):
            acc = np.mean(  # (e,m,t,..) -> (e,t,..) -> (e,t)  # XXX mean all accs
                sum(points[:, _, ...] for _ in akeys),
                axis=tuple(range(2, points.ndim - 1)),
            )
            acc *= sign
            acc[: int((0.5 + 0.1) * acc.shape[0]), ...] = -1e9  # XXX latter half

            bbidx = np.argmax(np.max(acc, axis=0), axis=0)  # ()
            bidx = np.argmax(acc, axis=0)  # .tolist()  # (t,)
            # XXX TODO XXX TODO ??? why after indexing, t & m switches ???  XXX TODO XXX TODO
            best = points[bidx, :, np.arange(bidx.shape[0]), ...]  # (t,m,..)

            bmean = np.mean(best, axis=0)  # (m,..)
            bstd = np.std(best, axis=0)
            return bbidx, bidx, best, bmean, bstd

        if akeys is not None:
            bbidx_t, bidx_t, best_t, bmean_t, bstd_t = mean_std_of_best(points_t, akeys)
            bbidx_v, bidx_v, best_v, bmean_v, bstd_v = mean_std_of_best(points_v, akeys)
            # print(f"best val of {group}:", bmean_v.round(4) * 100)
            print(group, *[f"{_:.2f}" for _ in bmean_v.round(4) * 100], sep=",")

            if copy_ckpt:
                print(
                    f"best val of {group}:", bmean_v[-2:].mean().round(4) * 100, bidx_v
                )
                for i, (lfile, biv) in enumerate(zip(log_files, bidx_v)):
                    bpath = str(lfile)[:-4]
                    epoch = trial_v.idxs[biv]
                    shutil.copy(  # copy best of every trials out
                        f"{bpath}/{epoch:04d}.pth", f"{bpath}-{epoch:04d}.pth"
                    )
                    if i == bbidx_v:  # copy best of all trials out
                        shutil.copy(
                            f"{bpath}/{epoch:04d}.pth", f"{lfile.parent}/best.pth"
                        )

            result_g["bmeanstd_t"] = [bmean_t, bstd_t]
            result_g["bmeanstd_v"] = [bmean_v, bstd_v]

        results[group] = result_g

    width = len(ckeys)
    _, axs = plt.subplots(2, width)
    if axs.ndim == 1:
        axs = axs[:, None]

    def draw_groups(gkey, axs, titles, x, ys, ds, alpha=0.1):
        for ax, title, y, d in zip(axs, titles, ys, ds):
            ax.set_title(title)
            ax.plot(x, y, label=gkey)
            y = np.array(y)
            d = np.array(d)
            ax.fill_between(x, y - d, y + d, alpha=alpha)

    for gkey, gvalue in results.items():
        meanstd_t = gvalue["stripe_t"].data.swapaxes(0, 1)  # (e,m,2,..) -> (m,e,2,..)
        yst = meanstd_t[:, :, 0]
        dst = meanstd_t[:, :, 1]
        draw_groups(
            gkey, axs[0], gvalue["stripe_t"].cols, gvalue["stripe_t"].idxs, yst, dst
        )
        meanstd_v = gvalue["stripe_v"].data.swapaxes(0, 1)
        ysv = meanstd_v[:, :, 0]
        dsv = meanstd_v[:, :, 1]
        draw_groups(
            gkey, axs[1], gvalue["stripe_v"].cols, gvalue["stripe_v"].idxs, ysv, dsv
        )

    [_.legend() for _ in axs.flatten()]
    plt.show()


if __name__ == "__main__":
    main_visualiz_log()
    # main_count_intra_inter_object_distance()
