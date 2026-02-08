# cdac2025

目标：训练模型实现 **热力图（heatmap）→ 屋顶平面图（roof plan）** 的图像到图像生成（paired 1-to-1，小样本约 40 对）。

Goal: Train models for **heatmap → roof plan** image-to-image translation (paired 1-to-1, small data ~40 pairs).

---

## 目录 / Table of Contents

- [项目概览 / Overview](#项目概览--overview)
- [快速开始（推荐流程）/ Quickstart (recommended)](#快速开始推荐流程-quickstart-recommended)
- [Google Colab 新手教程 / Beginner Colab Guide](#google-colab-新手教程--beginner-colab-guide)
- [数据格式 / Data Layout](#数据格式--data-layout)
- [预处理：同步裁剪（强烈推荐）/ Preprocess: synced cropping (recommended)](#预处理同步裁剪强烈推荐-preprocess-synced-cropping-recommended)
- [训练方案 1：ControlNet（Stable Diffusion）/ Training Option 1: ControlNet (Stable Diffusion)](#训练方案-1controlnetstable-diffusion--training-option-1-controlnet-stable-diffusion)
- [训练方案 2：SA-Img2Img（Self-Attention GAN）/ Training Option 2: SA-Img2Img (Self-Attention GAN)](#训练方案-2sa-img2imgself-attention-gan--training-option-2-sa-img2img-self-attention-gan)
- [输出与产物 / Outputs](#输出与产物--outputs)
- [小样本训练建议（40 对）/ Tips for small data (40 pairs)](#小样本训练建议40-对--tips-for-small-data-40-pairs)
- [常见问题 / FAQ](#常见问题--faq)

---

## 项目概览 / Overview

- 输入 / Input：热力图（heatmap，常见为 512×512 伪彩/网格）
- 输出 / Output：屋顶平面图（roof plan，常见为 590×590、白底细线）
- 任务 / Task：heatmap → roof plan（配对监督）

本仓库提供两条训练路线（你可以任选其一，或都跑来对比）：

1) **ControlNet + Stable Diffusion（Diffusers）**：`scripts/train_controlnet.py`
2) **SA-Img2Img（Self-Attention 条件 GAN，pix2pix 风格）**：`scripts/train_sa_img2img_gan.py`

This repo provides two training options:
1) **ControlNet + Stable Diffusion (Diffusers)**
2) **SA-Img2Img (Self-Attention conditional GAN, pix2pix-style)**

---

## 快速开始（推荐流程）/ Quickstart (recommended)

> 推荐：先做预处理（同步裁剪白边/有效区域），再训练。对小样本非常关键。

### Step 1) 安装依赖 / Install
```bash
pip install -r requirements.txt
```

### Step 2) 准备数据 / Prepare data
把数据按目录放好（同名配对）：

```
DATA_ROOT/
  heatmap/
    0001.png
    0002.png
  roof/
    0001.png
    0002.png
```

### Step 3) 预处理（同步裁剪）/ Preprocess (synced crop)
```bash
python scripts/preprocess_dataset.py \
  --data_root /path/to/DATA_ROOT \
  --out_root  /path/to/DATA_ROOT_CROPPED \
  --out_size 512 \
  --white_thresh 245 \
  --pad 16
```

### Step 4) 训练（任选其一）/ Train (pick one)

**Option A: SA-Img2Img GAN（推荐先跑通）**
```bash
python scripts/train_sa_img2img_gan.py \
  --data_root /path/to/DATA_ROOT_CROPPED \
  --out_dir outputs/sagan_roof \
  --image_size 512 \
  --batch_size 2 \
  --lr 2e-4 \
  --epochs 200 \
  --seed 42 \
  --val_ratio 0.2 \
  --split_seed 42
```

**Option B: ControlNet（Stable Diffusion）**
```bash
accelerate launch scripts/train_controlnet.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --output_dir outputs/controlnet_roof \
  --data_root /path/to/DATA_ROOT_CROPPED \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_train_steps 2000 \
  --validation_steps 200 \
  --prompt "a roof plan" \
  --seed 42 \
  --val_ratio 0.2 \
  --split_seed 42
```

---

## Google Colab 新手教程 / Beginner Colab Guide

> 适合从零开始。你只需要 Google 账号和浏览器。

### 0) 准备你本地的数据
先在本地整理为：

```
DATA_ROOT/
  heatmap/
    0001.png
  roof/
    0001.png
```

然后把 `DATA_ROOT` 压缩为 `DATA_ROOT.zip`（里面要直接是 heatmap/roof 两个文件夹）。

### 1) 打开 Colab 并启用 GPU
1. 打开 <https://colab.research.google.com>
2. 新建 Notebook
3. 菜单：`Runtime` → `Change runtime type` → `Hardware accelerator` 选 `GPU`

### 2) 克隆项目并安装依赖
在 Colab 单元运行：

```bash
!git clone https://github.com/wannaqueen66-create/cdac2025.git
%cd cdac2025
!pip -q install -r requirements.txt
```

### 3) 上传数据 zip 到 Colab
左侧文件面板上传 `DATA_ROOT.zip`，然后解压：

```bash
!mkdir -p /content/data_raw
!unzip -q /content/DATA_ROOT.zip -d /content/data_raw
```

> 如果你 zip 后多了一层目录，确认最终路径是：`/content/data_raw/DATA_ROOT/heatmap` 和 `/content/data_raw/DATA_ROOT/roof`。

### 4) 预处理（强烈推荐）
```bash
!python scripts/preprocess_dataset.py \
  --data_root /content/data_raw/DATA_ROOT \
  --out_root /content/data_cropped \
  --out_size 512 \
  --white_thresh 245 \
  --pad 16
```

### 5) 先跑 SA-Img2Img（新手推荐）
```bash
!python scripts/train_sa_img2img_gan.py \
  --data_root /content/data_cropped \
  --out_dir /content/outputs/sagan_roof \
  --image_size 512 \
  --batch_size 2 \
  --lr 2e-4 \
  --epochs 100 \
  --seed 42 \
  --val_ratio 0.2 \
  --split_seed 42
```

训练中重点看：
- `/content/outputs/sagan_roof/samples/`（可视化结果）
- `/content/outputs/sagan_roof/metrics.csv`（L1/PSNR/SSIM）

### 6) （可选）再跑 ControlNet
> ControlNet 更吃显存。Colab 免费卡可能要减小 step 或降低验证频率。

```bash
!accelerate launch scripts/train_controlnet.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --output_dir /content/outputs/controlnet_roof \
  --data_root /content/data_cropped \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_train_steps 1000 \
  --validation_steps 250 \
  --prompt "a roof plan" \
  --seed 42 \
  --val_ratio 0.2 \
  --split_seed 42
```

### 7) 下载训练结果到本地
```bash
!cd /content && zip -qr outputs.zip outputs
```
然后在左侧文件面板下载 `/content/outputs.zip`。

### 8) 常见报错（新手版）
- **没开 GPU / CUDA 报错**：确认 Runtime 已切到 GPU。
- **找不到数据目录**：确认解压后路径是否真的有 `heatmap/` 和 `roof/`。
- **显存不足（OOM）**：先减 `batch_size`，再减 `max_train_steps` 或 `resolution`。
- **会话断开**：Colab 空闲会断，建议定期把 `/content/outputs` 打包下载。

---

## 数据格式 / Data Layout

- `DATA_ROOT/heatmap/`：条件输入图（heatmap）
- `DATA_ROOT/roof/`：目标输出图（roof plan）
- 文件名必须一一对应（例如 `0001.png` 同时存在于两个目录）。

- `DATA_ROOT/heatmap/`: conditioning images (heatmaps)
- `DATA_ROOT/roof/`: target images (roof plans)
- Filenames must match 1-to-1.

支持扩展名 / Supported: `png/jpg/jpeg`。

---

## 预处理：同步裁剪（强烈推荐）/ Preprocess: synced cropping (recommended)

脚本 / Script：`scripts/preprocess_dataset.py`

目的 / Motivation：
- roof 图经常有大量白底边框，热力图也可能有无效背景。
- 小样本下，先裁掉无效区域会显著提高学习效率与对齐度。

本预处理流程会：
1) 在 roof 图上找“非白色区域”的 bbox（可调阈值 `--white_thresh`），并加 padding（`--pad`）
2) 将 bbox 按分辨率比例映射到 heatmap 图上并同步裁剪
3) 两边裁剪后 pad 成正方形并 resize 到 `--out_size`（默认 512）
4) 输出到新的目录 `DATA_ROOT_CROPPED/heatmap` 与 `DATA_ROOT_CROPPED/roof`

The script:
1) finds the non-white bbox on roof
2) scales that bbox into heatmap space and crops heatmap accordingly
3) pads to square and resizes to `--out_size`
4) writes to `DATA_ROOT_CROPPED/`

---

## 训练方案 1：ControlNet（Stable Diffusion） / Training Option 1: ControlNet (Stable Diffusion)

脚本 / Script：`scripts/train_controlnet.py`

要点 / Key points：
- 使用 Diffusers 组件：VAE / UNet / CLIPTextEncoder + ControlNet
- 默认仅训练 ControlNet（UNet/VAE/TextEncoder 冻结），对小样本更稳
- 需要文本 prompt；小样本建议固定 prompt（如 `a roof plan`）

---

## 训练方案 2：SA-Img2Img（Self-Attention GAN） / Training Option 2: SA-Img2Img (Self-Attention GAN)

脚本 / Script：`scripts/train_sa_img2img_gan.py`

要点 / Key points：
- pix2pix 风格 conditional GAN（PatchGAN 判别器）
- 生成器为 U-Net，并在 bottleneck 加 self-attention
- Loss = GAN + L1（默认 `lambda_l1=100`）

---

## 输出与产物 / Outputs

### ControlNet
- `outputs/controlnet_roof/controlnet_step_*/`：阶段 checkpoint
- `outputs/controlnet_roof/controlnet_final/`：最终权重
- `outputs/controlnet_roof/validation/`：固定验证集生成图
- `outputs/controlnet_roof/val_split.txt`：固定验证集样本名
- `outputs/controlnet_roof/metrics.csv`：自动评估指标（L1 / PSNR / SSIM）
- `outputs/controlnet_roof/train_config.json`：训练配置与 git commit

### SA-Img2Img
- `outputs/sagan_roof/checkpoints/`：训练 checkpoint（`.pt`）
- `outputs/sagan_roof/samples/`：采样图（conditioning / fake / real 三列）
- `outputs/sagan_roof/val_split.txt`：固定验证集样本名
- `outputs/sagan_roof/metrics.csv`：自动评估指标（L1 / PSNR / SSIM）
- `outputs/sagan_roof/train_config.json`：训练配置与 git commit

---

## 小样本训练建议（40 对）/ Tips for small data (40 pairs)

- 先用 **512** 跑通（你热力图本来就是 512），roof 经预处理后统一 512。
- 过拟合很快：建议频繁查看 sample/validation，必要时提前停。
- 如果线条细节不够：
  - 先保证结构对齐（预处理很关键）
  - 再考虑更高分辨率（640/768）或 tile/patch 训练

---

## 常见问题 / FAQ

**Q: 为什么要先做预处理？**
- A: roof 的白边会稀释有效信号；同步裁剪能提高 heatmap→roof 的空间对齐，尤其小样本更明显。

**Q: 两个训练脚本都需要预处理吗？**
- A: 是的。预处理是数据层面的通用步骤，两个方案都建议使用同一份裁剪后的数据根目录。

