import colorsys

import numpy as np
import torch as pt
import torchvision.utils as ptvu
import torch.nn.functional as ptnf


def index_segment_to_bbox(segment_idx: np.ndarray):
    """
    segment_idx: shape=(h,w)
    bbox: shape=(n,c=4), ltrb
    """
    assert segment_idx.ndim == 2 and segment_idx.dtype == np.uint8
    idxs = np.unique(segment_idx).tolist()
    idxs.sort()
    if 0 in idxs:
        idxs.remove(0)  # not include the bbox for background
    bbox = np.zeros([len(idxs), 4], dtype="float32")
    for i, idx in enumerate(idxs):
        y, x = np.where(segment_idx == idx)
        bbox[i, 0] = np.min(x)  # left
        bbox[i, 1] = np.min(y)  # top
        bbox[i, 2] = np.max(x)  # right
        bbox[i, 3] = np.max(y)  # bottom
    return bbox


def generate_spectrum_colors(num_color):
    spectrum = []
    for i in range(num_color):
        hue = i / float(num_color)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        spectrum.append([int(255 * c) for c in rgb])
    return np.array(spectrum, dtype="uint8")  # (n,c=3)


def draw_segmentation_np(
    image: np.ndarray, segment: np.ndarray, max_num=0, alpha=0.5, colors=None
):
    """
    image: in shape (h,w,c)
    segment: in shape (h,w)
    """
    if not max_num:
        max_num = int(segment.max() + 1)
    if colors is None:
        colors = generate_spectrum_colors(max_num)  # len(np.unique(segment))
    mask = ptnf.one_hot(pt.from_numpy(segment.astype("int64")), max_num)
    image2 = ptvu.draw_segmentation_masks(
        image=pt.from_numpy(image).permute(2, 0, 1),
        masks=mask.bool().permute(2, 0, 1),
        alpha=alpha,
        colors=colors.tolist(),
    )
    return image2.permute(1, 2, 0).numpy()
