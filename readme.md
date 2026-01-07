# `MSF` Multi-Scale Fusion for Object Representation



<br>
<br>

## ⚗️ (2026/01/06) Update !!!

Please check our brand new OCL works:
- **[RandSF.Q](https://github.com/Genera1Z/RandSF.Q)**: significantly surpasses state-of-the-art video OCL, e.g., **SlotContrast**, by **up to 10 points**!
- **[SmoothSA](https://github.com/Genera1Z/SmoothSA)**: improves the state of the art **even further**, e.g., **SPOT** / **DIAS** (images) and **SlotContrast** / **RandSF.Q** (videos), with **minimal modifications**!

<br>
<br>
<br>

---



## About

Official implementation of ICLR 2025 paper "*Multi-Scale Fusion for Object Representation*" available on [arXiv:2410.01539](https://arxiv.org/abs/2410.01539).

**Please note that `MSF` is *re-implemented* upon codebase 🤗 [VQ-VFM-OCL](https://github.com/Genera1Z/VQ-VFM-OCL), different from the version described in the paper. For more details, models, checkpoints, datasets and results, please visit this repo.**

Quantitative results of object discovery on COCO: (Encoding with backbone **DINO2/S-14** at resolution **256x256/224x224**)

<img src="res/quantitative_results.png" style="width:40%;">



## Converted Datasets 🚀

Dataset [COCO](https://cocodataset.org) is available on [dataset-coco](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-coco), which is converted in LMDB database format and can be used off-the-shelf in this repo.



## Model Checkpoints 🌟

The checkpoints for the models are available.
- [slate-msf-coco](https://github.com/Genera1Z/MultiScaleFusion/releases/tag/slate-msf-coco): SLATE-MSF, i.e., `MSF-Tfd`, on COCO.
- [slotdiffusion-msf-coco](https://github.com/Genera1Z/MultiScaleFusion/releases/tag/slotdiffusion-msf-coco): SlotDiffusion-MSF, i.e., `MSF-Dfz`, on COCO.



## How to Use

Take SLATE-MSF on COCO as an example.

**(1) Environment**

To set up the environment, run:
```shell
# python 3.11
pip install -r requirements.txt
```

**(2) Dataset**

To prepare the dataset, download ***Converted Datasets*** and unzip to `path/to/your/dataset/`. Or convert them by yourself according to ```XxxDataset.convert_dataset()``` docs.

**(3) Train**

To train the model, run:
```shell
# 1. pretrain the MSF VAE module
python train.py \
    --seed 42 \
    --cfg_file config-slatesteve/vqvae-coco-c256-msf.py \
    --data_dir path/to/your/dataset \
    --save_dir save

# *. place the best VAE checkpoint at archive-slatesteve/vqvae-coco-c256-msf/best.pth
mv save archive-slatesteve

# 2. train the SLATE-MSF OCL model
python train.py \
    --seed 42 \
    --cfg_file config-slatesteve/slate_r_vqvae-coco-msf.py \
    --data_dir path/to/your/dataset \
    --save_dir save \
    --ckpt_file archive-slatesteve/vqvae-coco-c256-msf/best.pth
```

**(4) Evaluate**

To evaluate the model, run:
```shell
python eval.py \
    --cfg_file config-slatesteve/slate_r_vqvae-coco-msf.py \
    --data_dir path/to/your/dataset \
    --ckpt_file archive-slatesteve/slate_r_vqvae-coco-msf/best.pth \
    --is_viz True
# object discovery accuracy values will be printed in the terminal
# object discovery visualization will be saved to ./slate_r_vqvae-coco-msf/
```

## Support

If you have any issues on this repo or cool ideas on OCL, please do not hesitate to contact me!
- page: https://genera1z.github.io
- email: rongzhen.zhao@aalto.fi, zhaorongzhenagi@gmail.com



## Citation

If you find this repo useful, please cite our work.
```
@article{zhao2025msf,
  title={{Multi-Scale Fusion for Object Representation}},
  author={Zhao, Rongzhen and Wang, Vivienne and Kannala, Juho and Pajarinen, Joni},
  journal={ICLR},
  year={2025}
}
```
