from pathlib import Path
import json
import pickle as pkl
import time

import cv2
import lmdb
import numpy as np
import torch as pt
import torch.utils.data as ptud

from .utils import draw_segmentation_np


class MSCOCO(ptud.Dataset):
    """
    Common Objects in COntext  https://cocodataset.org
    """

    def __init__(
        self,
        data_file,
        extra_keys=["bbox", "segment", "clazz"],
        transform=lambda **_: _,
        max_spare=4,
        base_dir: Path = None,
    ):
        if base_dir:
            data_file = base_dir / data_file
        self.env = lmdb.open(
            str(data_file),
            subdir=False,
            readonly=True,
            readahead=False,
            meminit=False,
            max_spare_txns=max_spare,
            lock=False,
        )
        with self.env.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))
        self.extra_keys = extra_keys
        self.transform = transform

    def __getitem__(self, index, compact=True):
        """
        - image: shape=(c=3,h,w), uint8
        - segment: shape=(h,w), uint8
        - bbox: shape=(n-1,c=4), float32; not include bg; ltrb
        - clazz: shape=(n-1,), uint8; not include bg
        """
        # load sample pack
        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.idxs[index]))
        sample1 = {}

        # load image and segment
        image0 = cv2.cvtColor(
            cv2.imdecode(  # cvtColor will unify images to 3 channels safely
                np.frombuffer(sample0["image"], "uint8"), cv2.IMREAD_UNCHANGED
            ),
            cv2.COLOR_BGR2RGB,
        )
        image = pt.from_numpy(image0).permute(2, 0, 1)
        sample1["image"] = image

        if "segment" in self.extra_keys:
            segment = pt.from_numpy(
                cv2.imdecode(sample0["segment"], cv2.IMREAD_GRAYSCALE)
            )
            sample1["segment"] = segment

            sidxs0_ = segment.unique(sorted=True)
            sidxs0 = sidxs0_.numpy().tolist()

            # load bbox and clazz for set prediction
            if "bbox" in self.extra_keys:
                bbox = pt.from_numpy(sample0["bbox"])  # (s,c=4)
                assert bbox.size(0) + (0 in sidxs0) == len(sidxs0)
                sample1["bbox"] = bbox
            if "clazz" in self.extra_keys:
                clazz = pt.from_numpy(sample0["clazz"])  # (s,)
                assert clazz.size(0) + (0 in sidxs0) == len(sidxs0)
                sample1["clazz"] = clazz

        # conduct transformation
        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]
            sidxs2_ = segment2.unique(sorted=True)
            sidxs2 = sidxs2_.numpy().tolist()

            cond = pt.isin(sidxs0_, sidxs2_)
            assert cond.ndim == 1
            if sidxs0_[0] == 0:
                cond = cond[1:]

            # ``RandomCrop`` and ``CenterCrop`` can produce invalid boxes, which are set to all-zero.
            # Let us filter them out, as well as the corresponding clazz.
            if "bbox" in self.extra_keys:
                bbox2 = sample2["bbox"][cond]  # (n,c=4) -> (?,c=4)
                assert bbox2.size(-1) == 4 and bbox2.ndim == 2
                assert bbox2.size(0) + (0 in sidxs2) == len(sidxs2)
                sample2["bbox"] = bbox2

                if "clazz" in self.extra_keys:
                    clazz2 = sample2["clazz"][cond]  # (n,) -> (?,)
                    assert clazz2.ndim == 1 and clazz2.size(0) == bbox2.size(0)
                    assert clazz2.size(0) + (0 in sidxs2) == len(sidxs2)
                    sample2["clazz"] = clazz2

            # compact segment idxs to be continuous
            if compact:
                segment3 = sample2["segment"]
                # remove background index 0
                sidxs3 = list(set(segment3.unique().tolist()) - {0})
                sidxs3.sort()
                cnt = 1  # index 0 means background
                for sidx3 in sidxs3:
                    segment3[segment3 == sidx3] = cnt
                    cnt += 1
                sample2["segment"] = segment3

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
        - 2017 Train images [118K/18GB] http://images.cocodataset.org/zips/train2017.zip
        - 2017 Val images [5K/1GB] http://images.cocodataset.org/zips/val2017.zip

        Structure dataset as follows and run it!
        - annotations
          - panoptic_coco_categories.json  # download from https://github.com/cocodataset/panopticapi
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
            lmdb_env = lmdb.open(
                str(dst_file),
                map_size=1024**4,
                subdir=False,
                readonly=False,
                meminit=False,
            )

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
                segment0 = (
                    (segment_rgb * [[[256**0, 256**1, 256**2]]]).sum(2).astype("int32")
                )
                # remove unannotated index 0
                sidxs = list(set(np.unique(segment0).tolist()) - {0})
                sidxs.sort()
                assert set(sids0) == set(sidxs)

                segment = np.zeros_like(segment0, "uint8")  # (h,w)
                both_side = np.tile(segment.shape[:2][::-1], 2).astype("float32")
                bbox = []
                clazz = []

                for si, sidx in enumerate(sidxs):
                    ci = sinfo0[sidx]["category_id"]
                    it = categories[ci]["isthing"]
                    assert it in [0, 1]
                    # merge stuff into background as index 0; shift things to index + 1
                    segment[segment0 == sidx] = (si + 1) if it else 0

                    if it:  # only keep the bbox and clazz of things
                        bb = sinfo0[sidx]["bbox"]  # xywh
                        bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]  # ltrb
                        bbox.append(bb)
                        clazz.append(ci)

                # bbox = index_segment_to_bbox(segment).reshape(-1, 4)
                bbox = np.array(bbox, "float32").reshape(-1, 4)  # in case no elements
                bbox = bbox / both_side  # whwh
                clazz = np.array(clazz, "uint8")

                # image = cv2.imdecode(  # there are some grayscale images
                #     np.frombuffer(image_b, "uint8"), cv2.IMREAD_COLOR
                # )
                # print(bbox.shape, clazz.shape)
                # __class__.visualiz(image, bbox, segment, clazz, wait=0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                assert type(image_b) == bytes
                assert (
                    bbox.ndim == 2 and bbox.shape[1] == 4 and bbox.dtype == np.float32
                )
                assert segment.ndim == 2 and segment.dtype == np.uint8
                assert clazz.ndim == 1 and clazz.dtype == np.uint8
                assert (
                    len(set(np.unique(segment).tolist()) - {0})
                    == bbox.shape[0]
                    == clazz.shape[0]
                )

                sample_dict = dict(
                    image=image_b,  # (h,w,c=3) bytes
                    bbox=bbox,  # (n,c=4) float32 ltrb
                    # re-encoding consumes less space than segment_b
                    segment=cv2.imencode(".webp", segment)[1],  # (h,w) uint8
                    clazz=clazz,  # (n,) uint8
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
    def visualiz(image, bbox=None, bbox_gt=None, segment=None, clazz=None, wait=0):
        """
        - image: bgr format, shape=(h,w,c=3), uint8
        - bbox: both normalized ltrb, shape=(n,c=4), float32
        - segment: index format, shape=(h,w), uint8
        - clazz: shape=(n,), uint8
        """
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8

        if bbox is not None and bbox.shape[0]:
            assert (
                bbox.ndim == 2
                and bbox.shape[1] == 4
                and bbox.dtype in [np.float16, np.float32]
            )
            bbox = bbox * np.tile(image.shape[:2][::-1], 2)
            for box in bbox.astype("int"):
                image = cv2.rectangle(
                    image, tuple(box[:2]), tuple(box[2:]), (0, 0, 0), 2
                )
                # image = cv2.circle(
                #     image, tuple(box[:2]), radius=5, color=(0, 0, 0), thickness=-1
                # )
                # image = cv2.circle(
                #     image, tuple(box[2:]), radius=5, color=(255, 255, 255), thickness=-1
                # )

        if bbox_gt is not None and bbox_gt.shape[0]:
            assert (
                bbox_gt.ndim == 2
                and bbox_gt.shape[1] == 4
                and bbox_gt.dtype in [np.float16, np.float32]
            )
            bbox_gt = bbox_gt * np.tile(image.shape[:2][::-1], 2)
            for box_gt in bbox_gt.astype("int"):
                image = cv2.rectangle(
                    image, tuple(box_gt[:2]), tuple(box_gt[2:]), (63, 127, 255), 2
                )

        cv2.imshow("i", image)

        segment_viz = None
        if segment is not None:
            assert segment.ndim == 2 and segment.dtype == np.uint8
            segment_viz = draw_segmentation_np(image, segment, alpha=0.75)

            if clazz is not None and bbox.shape[0]:
                assert clazz.ndim == 1 and clazz.dtype == np.uint8
                nseg = list(set(np.unique(segment).tolist()) - {0})
                nseg.sort()
                assert len(nseg) == len(clazz)
                for iseg, iclz in zip(nseg, clazz):
                    y, x = np.where(segment == iseg)
                    l = np.min(x)
                    t = np.min(y)
                    r = np.max(x)
                    b = np.max(y)
                    segment_viz = cv2.putText(
                        segment_viz,
                        f"{iclz}",
                        [int((l + r) / 2), int((t + b) / 2)],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255] * 3,
                    )

            cv2.imshow("s", segment_viz)

        cv2.waitKey(wait)
        return image, segment_viz
