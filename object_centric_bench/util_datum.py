"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

import colorsys

import numpy as np
import torch as pt
import torchvision.utils as ptvu


def mask_segment_to_bbox_np(segment):
    """
    - segment: mask format, shape=(h,w,s)
    - bbox: ltrb format, shape=(s,c=4)
    """
    assert segment.ndim == 3 and segment.dtype == bool
    h, w, s = segment.shape
    y = np.arange(h)[:, None, None]
    x = np.arange(w)[None, :, None]
    l = np.amin(np.where(segment, x, np.inf), (0, 1))
    t = np.amin(np.where(segment, y, np.inf), (0, 1))
    r = np.amax(np.where(segment, x, -np.inf), (0, 1))
    b = np.amax(np.where(segment, y, -np.inf), (0, 1))
    bbox = np.stack([l, t, r, b], 1)
    valid = segment.any((0, 1))
    bbox[~valid] = 0
    bbox = bbox.astype("int32")
    # assert ((l <= r) & (t <= b)).all()  # has strange error for float64
    assert (bbox[:, :2] <= bbox[:, 2:]).all()  # left-closed and right-closed
    return bbox


def generate_spectrum_colors(num_color):
    spectrum = []
    for i in range(num_color):
        hue = i / float(num_color)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        spectrum.append([int(255 * c) for c in rgb])
    return np.array(spectrum, dtype="uint8")  # (s,c=3)


def draw_segmentation_np(image: np.ndarray, segment: np.ndarray, alpha=0.5, color=None):
    """
    - image: shape=(h,w,c)
    - segment: shape=(h,w,s), dtype=bool; in mask format, not index format
    - color: shape=(s,c=3)
    """
    h, w, c = image.shape
    h2, w2, s = segment.shape
    assert h == h2 and w == w2

    if color is None:
        color = generate_spectrum_colors(s)
    image2 = ptvu.draw_segmentation_masks(
        image=pt.from_numpy(image).permute(2, 0, 1),
        masks=pt.from_numpy(segment).permute(2, 0, 1),
        alpha=alpha,
        colors=color.tolist(),
    )
    return image2.permute(1, 2, 0).numpy()
