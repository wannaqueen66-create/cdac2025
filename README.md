# cdac2025

目标：训练模型实现 **热力图（heatmap）→ 屋顶平面图（roof plan）** 的图像到图像生成（paired 1-to-1）。

Goal: Train models for **heatmap → roof plan** image-to-image translation (paired 1-to-1).

---

## 目录 / Table of Contents
- [项目背景 / Background](#项目背景--background)
- [数据格式 / Data Layout](#数据格式--data-layout)
- [安装 / Installation](#安装--installation)
- [训练脚本 1：ControlNet（Stable Diffusion）/ Training Script 1: ControlNet (Stable Diffusion)](#训练脚本-1controlnetstable-diffusion--training-script-1-controlnet-stable-diffusion)
- [训练脚本 2：SA-Img2Img（Self-Attention GAN）/ Training Script 2: SA-Img2Img (Self-Attention GAN)](#训练脚本-2sa-img2imgself-attention-gan--training-script-2-sa-img2img-self-attention-gan)
- [输出与产物 / Outputs](#输出与产物--outputs)
- [小样本建议（40 对）/ Tips for Small Data (40 pairs)](#小样本建议40-对--tips-for-small-data-40-pairs)

---

## 项目背景 / Background

- 需要训练两类模型：
  1) **ControlNet + Stable Diffusion**（用 heatmap 作为条件）
  2) **Self-Attention GAN（SA-Img2Img）**（条件 GAN，pix2pix 风格，加入 self-attention）
- 数据：40 组 **屋顶平面图 ↔ 热力图** 一对一配对。
- 目标：从热力图反推屋顶平面图。

We train two model families:
1) **ControlNet + Stable Diffusion** (heatmap as conditioning)
2) **Self-Attention GAN (SA-Img2Img)** (pix2pix-like conditional GAN + self-attention)

Dataset: 40 paired samples (**roof plan ↔ heatmap**). Goal: generate roof plan from heatmap.

---

## 数据格式 / Data Layout

将数据按如下结构放置，文件名必须一一对应：

Put paired images under one root, with identical filenames:

```
DATA_ROOT/
  heatmap/
    0001.png
    0002.png
  roof/
    0001.png
    0002.png
```

支持扩展名 / Supported: `png/jpg/jpeg`。

---

## 安装 / Installation

```bash
pip install -r requirements.txt
```

---

## 训练脚本 1：ControlNet（Stable Diffusion） / Training Script 1: ControlNet (Stable Diffusion)

脚本 / Script：`scripts/train_controlnet.py`

示例 / Example：
```bash
accelerate launch scripts/train_controlnet.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --output_dir outputs/controlnet_roof \
  --data_root /path/to/DATA_ROOT \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_train_steps 2000 \
  --validation_steps 200 \
  --prompt "a roof plan" \
  --seed 42
```

说明 / Notes：
- 默认只训练 ControlNet（UNet/VAE/TextEncoder 冻结）。
- 需要文本 prompt；小样本建议用固定 prompt（例如 `a roof plan`）。

---

## 训练脚本 2：SA-Img2Img（Self-Attention GAN） / Training Script 2: SA-Img2Img (Self-Attention GAN)

脚本 / Script：`scripts/train_sa_img2img_gan.py`

示例 / Example：
```bash
python scripts/train_sa_img2img_gan.py \
  --data_root /path/to/DATA_ROOT \
  --out_dir outputs/sagan_roof \
  --image_size 256 \
  --batch_size 4 \
  --lr 2e-4 \
  --epochs 200 \
  --seed 42
```

说明 / Notes：
- 条件输入为 heatmap，输出为 roof。
- Loss = GAN loss + L1（默认 `lambda_l1=100`）。

---

## 预处理：裁剪屋顶白边并同步裁热力图 / Preprocess: crop roof white margins and sync-crop heatmaps

脚本 / Script：`scripts/preprocess_dataset.py`

示例 / Example：
```bash
python scripts/preprocess_dataset.py \
  --data_root /path/to/DATA_ROOT \
  --out_root /path/to/DATA_ROOT_CROPPED \
  --out_size 512 \
  --white_thresh 245 \
  --pad 16
```

然后训练时把 `--data_root` 指向裁剪后的目录。

Then point training `--data_root` to the cropped directory.

## 输出与产物 / Outputs

- ControlNet：
  - `outputs/controlnet_roof/controlnet_step_*/`（阶段 checkpoint）
  - `outputs/controlnet_roof/controlnet_final/`（最终权重）
  - `outputs/controlnet_roof/validation/`（验证生成图）

- SA-Img2Img：
  - `out_dir/checkpoints/`（`ckpt_*.pt`）
  - `out_dir/samples/`（训练过程中采样图：conditioning / fake / real 三列）

---

## 小样本建议（40 对） / Tips for Small Data (40 pairs)

- 尽量降低学习率、增加正则与数据增强（翻转/轻微颜色扰动等）。
- 频繁验证并早停，避免过拟合。
- ControlNet 建议：只训 ControlNet、固定 prompt、少步数多观察。

- Use low LR, add augmentation and early stopping.
- Validate frequently; small paired sets overfit quickly.
- For ControlNet: train ControlNet only, fixed prompt, fewer steps.

