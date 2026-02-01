# cdac2025

Goal: train models that map **heatmap → roof plan** (paired, 1-to-1).

You asked for two training scripts:
1) **ControlNet (Stable Diffusion + ControlNet)** fine-tuning
2) **Self-Attention GAN (SA-Img2Img)** (pix2pix-style conditional GAN with self-attention)

## Data layout (expected)
Put paired images under one root, same filenames:

```
DATA_ROOT/
  heatmap/
    0001.png
    0002.png
  roof/
    0001.png
    0002.png
```

Supported extensions: png/jpg/jpeg.

## Setup
```bash
pip install -r requirements.txt
```

## 1) Train ControlNet
Script: `scripts/train_controlnet.py`

Example:
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

Notes for small data (40 pairs): keep LR low, use augmentation, consider early stopping.

## 2) Train SA-Img2Img (Self-Attention GAN)
Script: `scripts/train_sa_img2img_gan.py`

Example:
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

Outputs:
- checkpoints under `out_dir/checkpoints/`
- sample grids under `out_dir/samples/`

