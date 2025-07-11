resolut0 = [256, 256]
num_code = 256 * 256
embed_dim0 = 4
num_group = 2  # must be 2
groups = [int(num_code ** (1 / num_group))] * num_group
expanz = 1  # 2
num_scale = 3

total_step = 30000
val_interval = total_step // 40
batch_size_t = 64
batch_size_v = batch_size_t
num_work = 4
lr = 2e-3

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    dict(type="RandomCrop", keys=["image"], size=None, scale=[0.75, 1]),
    dict(type="Resize", keys=["image"], size=resolut0, interp="bilinear"),
    dict(type="RandomFlip", keys=["image"], dims=[-1], p=0.5),
    dict(type="Normalize", keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
transform_v = [
    dict(type="CenterCrop", keys=["image"], size=None),
    dict(type="Resize", keys=["image"], size=resolut0, interp="bilinear"),
    dict(type="Normalize", keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
dataset_t = dict(
    type="MSCOCO",
    data_file="coco/train.lmdb",
    instance=True,
    extra_keys=[],
    transform=dict(type="Compose", transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type="MSCOCO",
    data_file="coco/val.lmdb",
    instance=True,
    extra_keys=[],
    transform=dict(type="Compose", transforms=transform_v),
    base_dir=...,
)
collate_fn_t = None
collate_fn_v = None

### model

model = dict(
    type="VQVAEZMultiScale",
    encode=dict(
        type="Sequential",
        modules=[
            dict(type="Interpolate", scale_factor=0.5, interp="bicubic"),
            dict(type="EncoderTAESD", se=[0, 14], gn=0),
            dict(type="GroupNorm", num_groups=1, num_channels=64),
            dict(type="Conv2d", in_channels=64, out_channels=embed_dim0, kernel_size=1),
        ],
    ),
    decode=dict(
        type="Sequential",
        modules=[
            dict(type="Conv2d", in_channels=embed_dim0, out_channels=64, kernel_size=1),
            dict(type="GroupNorm", num_groups=1, num_channels=64),
            dict(type="DecoderTAESD", se=[2, 19], gn=0),
        ],
    ),
    quant=dict(
        type="ModuleList",
        modules=[
            dict(
                type="QuantiZ",
                num_code=int(groups[0] / 2 ** max(0, _ - 1)),
                code_dim=int(embed_dim0 * expanz),
                in_dim=int(1024),
                std=0,
            )
            for _ in range(1 + num_scale)
        ],
    ),
    project=None,  # TODO
    alpha=0.0,
    retr=False,
    eaq=False,
)
model_imap = dict(input="image")
model_omap = ["encode", "zidx", "quant", "residual", "decode"]
ckpt_map = None
freez = [r"m\.encode\.1\.(?:[0-9]|10)\..*"]

### learn

param_groups = None
optimiz = dict(type="Adam", params=param_groups, lr=lr)
gscale = dict(type="GradScaler")
gclip = dict(type="ClipGradNorm", max_norm=1)

loss_fn = dict(
    **{
        f"recon{_}": dict(
            metric=dict(type="MSELoss"),
            map=dict(input=f"output.decode.{_}", target=f"batch.image"),
            transform=dict(
                type="Resize",
                keys=["target"],
                size=[int(r / 2**_ / 2) for r in resolut0],
                interp="bicubic",
            ),
        )
        for _ in range(num_scale)
    },
    **{
        f"align{_}": dict(
            metric=dict(type="MSELoss"),
            map=dict(input=f"output.quant.{_}", target=f"output.encode.{_}"),
            transform=dict(type="Detach", keys=["target"]),
        )
        for _ in range(num_scale)
    },
    **{
        f"commit{_}": dict(
            metric=dict(type="MSELoss"),
            map=dict(input=f"output.encode.{_}", target=f"output.residual.{_}"),
            transform=dict(type="Detach", keys=["target"]),
            weight=0.25,
        )
        for _ in range(num_scale)
    },
    **{
        f"norm_e{_}": dict(
            metric=dict(type="MSELoss"),
            map=dict(input=f"output.encode.{_}", target=f"output.encode.{_}"),
            transform=dict(
                type="Lambda",
                keys=["target"],
                func="lambda _: ptnf.group_norm(_.detach(), num_groups=1)",
            ),
            weight=0.1,
        )
        for _ in range(num_scale)
    },
    **{
        f"lpips{_}": dict(
            metric=dict(type="LPIPSLoss", net="alex"),
            map=dict(input=f"output.decode.{_}", target=f"batch.image"),
            transform=dict(
                type="Resize",
                keys=["target"],
                size=[int(r / 2**_ / 2) for r in resolut0],
                interp="bicubic",
            ),
        )
        for _ in range(1)  # num_scale
    },
)
metric_fn_t = metric_fn_v = dict()

before_step = [
    dict(
        type="CbLnCosine",
        assigns=[
            f"model.m.quant[{_}].std.data[...]=value" for _ in range(1 + num_scale)
        ],
        ntotal=total_step,
        vbase=1,
        vfinal=2,
    ),
    dict(
        type="CbCosineLinear",
        assigns=["model.m.alpha.data[...]=value"],
        ncos=total_step // 2,
        ntotal=total_step,
        vbase=1,
        vmid=0,
        vfinal=0,
    ),
    dict(
        type="CbLinearCosine",
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,
    ),
]
callback_t = [
    dict(type="Callback", before_step=before_step),
    dict(type="AverageLog", log_file=...),
]
callback_v = [
    dict(type="Callback", before_step=None),
    callback_t[1],
    dict(type="SaveModel", save_dir=..., since_step=total_step * 0.5),
]
