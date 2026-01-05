"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

from pathlib import Path
import json
import pickle as pkl
import time

import cv2
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from .dataset import lmdb_open_read, lmdb_open_write
from ..util_datum import draw_segmentation_np, mask_segment_to_bbox_np


class MSCOCO(ptud.Dataset):
    """
    Common Objects in COntext  https://cocodataset.org

    Example
    ```
    dataset = MSCOCO(
        data_file="coco/train.lmdb",
        extra_keys=["segment", "bbox", "clazz"],
        mode="instance",
        base_dir=Path("/media/GeneralZ/Storage/Static/datasets"),
    )
    for sample in dataset:
        dataset.visualiz(
            image=sample["image"].permute(1, 2, 0).numpy(),
            segment=sample["segment"].numpy(),
            bbox=sample["bbox"].numpy(),
            clazz=sample["clazz"].numpy(),
        )
    ```
    """

    def __init__(
        self,
        data_file,
        extra_keys=["segment", "bbox", "clazz"],
        transform=lambda **_: _,
        mode="instance",  # instance panoptic
        base_dir: Path = None,
    ):
        if base_dir:
            data_file = base_dir / data_file
        self.data_file = data_file

        env = lmdb_open_read(data_file)
        with env.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))
        env.close()

        self.extra_keys = extra_keys
        self.transform = transform
        assert mode in ["instance", "panoptic"]
        self.mode = mode

    def __getitem__(self, index):
        """
        - image: shape=(c=3,h,w), uint8 | float32
        - segment: shape=(h,w,s), uint8 -> bool
        - bbox: shape=(s,c=4), float32, ltrb
        - clazz: shape=(s,), uint8
        """
        if not hasattr(self, "env"):  # torch>2.6
            self.env = lmdb_open_read(self.data_file)

        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.idxs[index]))
        sample1 = {}

        image0 = cv2.cvtColor(
            cv2.imdecode(  # cvtColor will unify images to 3 channels safely
                np.frombuffer(sample0["image"], "uint8"), cv2.IMREAD_UNCHANGED
            ),
            cv2.COLOR_BGR2RGB,
        )
        image = pt.from_numpy(image0).permute(2, 0, 1)
        sample1["image"] = image  # (c,h,w) uint8

        if "segment" in self.extra_keys:
            segment = pt.from_numpy(
                cv2.imdecode(sample0["segment"], cv2.IMREAD_GRAYSCALE)
            )
            sample1["segment"] = segment  # (h,w) uint8

            if "clazz" in self.extra_keys:
                clazz = pt.from_numpy(sample0["clazz"])
                sample1["clazz"] = clazz  # (s,) uint8

            isthing = pt.from_numpy(sample0["isthing"])  # (s,) bool

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]  # (h,w); index format
            # (h,w,s); mask format
            segment3 = ptnf.one_hot(segment2.long(), isthing.shape[0] + 1).bool()
            segment3 = segment3[:, :, 1:]  # remove unannotated area

            h, w, s = segment3.shape
            # assert s > 0  # some images have no annotation

            # ``RandomCrop`` and ``CenterCrop`` can diminish segments
            cond = segment3.any([0, 1])  # (s,)

            segment3 = segment3[:, :, cond]
            sample2["segment"] = segment3  # (h,w,s) bool

            if "bbox" in self.extra_keys:
                bbox2 = pt.from_numpy(mask_segment_to_bbox_np(segment3.numpy())).float()
                bbox2[:, 0::2] /= w  # normalize
                bbox2[:, 1::2] /= h
                sample2["bbox"] = bbox2  # (s,c=4) float32

            if "clazz" in self.extra_keys:
                clazz2 = sample2["clazz"][cond]
                sample2["clazz"] = clazz2  # (s,) uint8

        if self.mode == "instance":

            if "segment" in sample2:
                isthing = isthing[cond]
                isstuff = ~isthing

                segment9 = sample2["segment"]
                segment_bg = segment9[:, :, isstuff].any(2, True)  # merge stuff as bg
                # if isstuff.sum() == 0:
                #     segment_bg = segment_bg[:, :, :0]
                segment_fg = segment9[:, :, isthing]
                segment9 = pt.concat([segment_bg, segment_fg], 2)
                sample2["segment"] = segment9

            if "bbox" in sample2:
                assert "segment" in sample2
                bbox9 = sample2["bbox"][isthing]
                assert segment9.shape[2] == bbox9.shape[0] + 1
                sample2["bbox"] = bbox9

            if "clazz" in sample2:
                assert "segment" in sample2
                clazz9 = sample2["clazz"][isthing]
                assert segment9.shape[2] == clazz9.shape[0] + 1
                sample2["clazz"] = clazz9

        return sample2

    def __len__(self):
        return len(self.idxs)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw/mscoco"),
        dst_dir=Path("coco"),
    ):
        """
        Download dataset MSCOCO:
        - 2017 Train images [118K/18GB]
            http://images.cocodataset.org/zips/train2017.zip
        - 2017 Val images [5K/1GB]
            http://images.cocodataset.org/zips/val2017.zip
        - 2017 Panoptic Train/Val annotations [821MB]
            http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
        - panoptic_coco_categories.json [12.9KB]
            https://github.com/cocodataset/panopticapi

        Structure dataset as follows and run it!
        - annotations
          - panoptic_coco_categories.json
          - panoptic_train2017.json
          - panoptic_train2017
            - *.png
          - panoptic_val2017.json
          - panoptic_val2017
            - *.png
        - tain2017
          - *.jpg
        - val2017
          - *.jpg
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        category_file = src_dir / "annotations" / "panoptic_coco_categories.json"
        with open(category_file, "r") as f:
            categories = json.load(f)
        categories = {category["id"]: category for category in categories}

        splits = dict(
            train=["train2017", "annotations/panoptic_train2017"],
            val=["val2017", "annotations/panoptic_val2017"],
        )

        for split, [image_dn, segment_dn] in splits.items():
            print(split, image_dn, segment_dn)

            annotation_file = src_dir / f"{segment_dn}.json"
            with open(annotation_file, "r") as f:
                annotations = json.load(f)
            annotations = annotations["annotations"]

            dst_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb_open_write(dst_file)

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            # https://github.com/cocodataset/panopticapi/blob/master/converters/panoptic2detection_coco_format.py
            for cnt, annotat in enumerate(annotations):
                sids0 = [_["id"] for _ in annotat["segments_info"]]
                assert len(sids0) == len(set(sids0)) < 256
                sinfo0 = dict(zip(sids0, annotat["segments_info"]))

                fn = annotat["file_name"].split(".")[0]
                image_file = src_dir / image_dn / f"{fn}.jpg"
                segment_file = src_dir / segment_dn / f"{fn}.png"

                with open(image_file, "rb") as f:
                    image_b = f.read()
                segment_bgr = cv2.imread(str(segment_file))  # (h,w,c=3)
                segment_rgb = cv2.cvtColor(segment_bgr, cv2.COLOR_BGR2RGB)
                segment0 = (  # (h,w)
                    (segment_rgb * [[[256**0, 256**1, 256**2]]]).sum(2).astype("int32")
                )
                # remove unannotated segmentation index 0
                sidxs = list(set(np.unique(segment0).tolist()) - {0})
                sidxs.sort()
                assert set(sids0) == set(sidxs)

                segment = np.zeros_like(segment0, dtype="uint8")
                assert len(sids0) < 255  # uint8
                clazz = []
                isthing = []

                for si, sidx in enumerate(sidxs):
                    ci = sinfo0[sidx]["category_id"]
                    it = categories[ci]["isthing"]
                    assert it in [0, 1]
                    segment[segment0 == sidx] = si + 1  # shift segment to index + 1
                    clazz.append(ci)
                    isthing.append(it)

                clazz = np.array(clazz, "uint8")  # (s,)
                isthing = np.array(isthing, "bool")  # (s,)

                # image = cv2.imdecode(  # there are some grayscale images
                #     np.frombuffer(image_b, "uint8"), cv2.IMREAD_COLOR
                # )
                # segment_pt = pt.from_numpy(segment).long()
                # mask = ptnf.one_hot(segment_pt).bool().numpy()
                # if 0 in segment_pt.unique():  # if there is invalid annotation
                #     mask = mask[:, :, 1:]
                # __class__.visualiz(image, mask, None, clazz, wait=0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                assert type(image_b) == bytes
                assert segment.ndim == 2 and segment.dtype == np.uint8
                assert clazz.ndim == 1 and clazz.dtype == np.uint8
                assert isthing.ndim == 1 and isthing.dtype == bool
                assert (
                    len(set(np.unique(segment)) - {0})
                    == clazz.shape[0]
                    == isthing.shape[0]
                )

                sample_dict = dict(
                    image=image_b,  # (h,w,c=3) bytes
                    segment=cv2.imencode(".webp", segment)[1],  # (h,w) uint8
                    clazz=clazz,  # (s,) uint8
                    isthing=isthing,  # (s,) bool
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

            txn.commit()
            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

            print(f"total={cnt + 1}, time={time.time() - t0}")

    @staticmethod
    def visualiz(image, segment=None, bbox=None, clazz=None, wait=0):
        """
        - image: rgb format, shape=(h,w,c=3), uint8
        - segment: mask format, shape=(h,w,s), bool
        - bbox: both-side normalized ltrb, shape=(s,c=4), float32
        - clazz: shape=(s,), uint8
        """
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8

        cv2.imshow("i", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        segment_viz = None
        if segment is not None and segment.shape[2]:
            assert segment.ndim == 3 and segment.dtype == bool
            segment_viz = draw_segmentation_np(image, segment, alpha=0.75)

            if bbox is not None:
                ds = segment.shape[2] - bbox.shape[0]
                assert ds in [0, 1]
                assert (
                    bbox.ndim == 2 and bbox.shape[1] == 4 and bbox.dtype == np.float32
                )
                if clazz is not None:
                    assert clazz.shape[0] == bbox.shape[0]

                bbox = bbox * np.tile(segment_viz.shape[:2][::-1], 2)  # de-normalize
                for box in np.round(bbox).astype("int"):
                    segment_viz = cv2.rectangle(
                        segment_viz, tuple(box[:2]), tuple(box[2:]), (0, 0, 0), 2
                    )

            if clazz is not None:
                ds = segment.shape[2] - clazz.shape[0]
                assert ds in [0, 1]
                assert clazz.ndim == 1 and clazz.dtype == np.uint8
                if bbox is not None:
                    assert clazz.shape[0] == bbox.shape[0]

                for i, clz in enumerate(clazz):
                    mask = segment[
                        :, :, i + ds
                    ]  # skip annotated area (panoptic) or bg (instance)
                    total = float(np.sum(mask))
                    assert total > 0
                    ys, xs = np.indices(mask.shape)  # centroid
                    cx = int(round((xs * mask).sum() / total))
                    cy = int(round((ys * mask).sum() / total))

                    segment_viz = cv2.putText(
                        segment_viz,
                        f"{clz}",
                        [cx, cy],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255] * 3,
                    )

            cv2.imshow("s", cv2.cvtColor(segment_viz, cv2.COLOR_RGB2BGR))

        cv2.waitKey(wait)
        return image, segment_viz
