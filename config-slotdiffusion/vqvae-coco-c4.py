from object_centric_bench.datum import (
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    MSCOCO,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    ClipGradNorm,
    MSELoss,
    LPIPSLoss,
    CbLinearCosine,
    Callback,
    AverageLog,
    SaveModel,
)
from object_centric_bench.model import (
    Sequential,
    Interpolate,
    VQVAE,
    EncoderTAESD,
    GroupNorm,
    Conv2d,
    Codebook,
    DecoderTAESD,
)
from object_centric_bench.util import Compose

### global

resolut0 = [256, 256]
num_code = 4096
emb_dim0 = 4

total_step = 30000  # 50000 worse ???
val_interval = total_step // 40
batch_size_t = 64
batch_size_v = batch_size_t
num_work = 4
lr = 2e-3

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(type=RandomCrop, keys=["image"], size=None, scale=[0.75, 1]),
    dict(type=Resize, keys=["image"], size=resolut0, interp="bilinear"),
    dict(type=RandomFlip, keys=["image"], dims=[-1], p=0.5),
    dict(type=Normalize, keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
transform_v = [
    dict(type=CenterCrop, keys=["image"], size=None),
    dict(type=Resize, keys=["image"], size=resolut0, interp="bilinear"),
    dict(type=Normalize, keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
dataset_t = dict(
    type=MSCOCO,
    data_file="coco/train.lmdb",
    extra_keys=[],
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type=MSCOCO,
    data_file="coco/val.lmdb",
    extra_keys=[],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = None
collate_fn_v = None

### model

model = dict(
    type=VQVAE,
    encode=dict(
        type=Sequential,
        modules=[
            # ~=EncoderAKL (w/o mid); >>ResNet18, naive CNN
            dict(type=Interpolate, scale_factor=0.5, interp="bicubic"),
            dict(type=EncoderTAESD, se=[0, 14], gn=0),  # more convs in between: bad
            dict(type=GroupNorm, num_groups=1, num_channels=64),
            dict(type=Conv2d, in_channels=64, out_channels=emb_dim0, kernel_size=1),
        ],
    ),
    decode=dict(
        type=Sequential,
        modules=[
            dict(type=Conv2d, in_channels=emb_dim0, out_channels=64, kernel_size=1),
            dict(type=GroupNorm, num_groups=1, num_channels=64),
            dict(type=DecoderTAESD, se=[2, 19], gn=0),  # >> naive cnn  # in case oom
        ],
    ),
    codebook=dict(type=Codebook, num_embed=num_code, embed_dim=emb_dim0),
)
model_imap = dict(input="batch.image")
model_omap = ["encode", "zidx", "quant", "decode"]
ckpt_map = None
freez = [r"^m\.encode\.1\.(?:[0-9]|10)\..*"]  # train whole decode is the best

### learn

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)
gclip = dict(type=ClipGradNorm, max_norm=1)

loss_fn_t = loss_fn_v = dict(
    recon=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.decode", target="batch.image"),
        transform=dict(
            type=Resize,
            keys=["target"],
            size=[_ // 2 for _ in resolut0],
            interp="bicubic",
        ),
    ),
    align=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.quant", target="output.encode"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
    commit=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.encode", target="output.quant"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
        weight=0.25,
    ),
    lpips=dict(
        metric=dict(type=LPIPSLoss, net="alex"),
        map=dict(input="output.decode", target="batch.image"),
        transform=dict(
            type=Resize,
            keys=["target"],
            size=[_ // 2 for _ in resolut0],
            interp="bicubic",
        ),
    ),
)
acc_fn_t = acc_fn_v = dict()

before_step = [
    dict(type=Lambda, ikeys=[["batch.image"]], func=lambda _: _.cuda()),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step),
    dict(type=AverageLog, log_file=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1]),
    callback_t[1],
    dict(type=SaveModel, save_dir=..., since_step=total_step * 0.5),
]
