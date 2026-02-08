"""Train ControlNet for heatmap -> roof plan (paired) with fixed validation split and metrics."""

import argparse
import csv
import json
import math
import os
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.transforms.functional import to_tensor

from common_dataset import PairedImageDataset, discover_pairs, split_pairs
from metrics import l1 as metric_l1, psnr as metric_psnr, ssim_global


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--lr_warmup_steps", type=int, default=200)
    p.add_argument("--validation_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str, default="a roof plan")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def run_validation_and_metrics(
    accelerator: Accelerator,
    pipe: StableDiffusionControlNetPipeline,
    val_batch,
    prompt: str,
    step: int,
    out_dir: str,
):
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    cond = val_batch["conditioning"].to(accelerator.device)
    real = val_batch["target"].to(accelerator.device)

    cond_01 = (cond * 0.5 + 0.5).clamp(0, 1)
    real_01 = (real * 0.5 + 0.5).clamp(0, 1)

    images = pipe(
        prompt=[prompt] * cond_01.shape[0],
        image=cond_01,
        num_inference_steps=30,
        guidance_scale=7.0,
    ).images

    os.makedirs(os.path.join(out_dir, "validation"), exist_ok=True)
    pred_tensors = []
    for i, im in enumerate(images):
        im.save(os.path.join(out_dir, "validation", f"step_{step:06d}_{i}.png"))
        pred_tensors.append(to_tensor(im).to(accelerator.device))

    pred = torch.stack(pred_tensors, dim=0).clamp(0, 1)

    return {
        "l1": metric_l1(pred, real_01).item(),
        "psnr": metric_psnr(pred, real_01).item(),
        "ssim": ssim_global(pred, real_01).item(),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )

    set_seed(args.seed)

    if accelerator.is_main_process:
        pairs = discover_pairs(args.data_root)
        train_pairs, val_pairs = split_pairs(pairs, val_ratio=args.val_ratio, seed=args.split_seed)
        with open(os.path.join(args.output_dir, "val_split.txt"), "w", encoding="utf-8") as f:
            for p in val_pairs:
                f.write(p.name + "\n")
        with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "git_commit": get_git_commit(),
                    "num_total": len(pairs),
                    "num_train": len(train_pairs),
                    "num_val": len(val_pairs),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(os.path.join(args.output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "val_l1", "val_psnr", "val_ssim"])
    else:
        train_pairs, val_pairs = [], []

    # broadcast split from main process
    train_pairs = accelerator.gather_for_metrics(train_pairs) if False else train_pairs

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    controlnet = ControlNetModel.from_unet(unet)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # rebuild split on each process deterministically
    pairs = discover_pairs(args.data_root)
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=args.val_ratio, seed=args.split_seed)

    train_ds = PairedImageDataset(None, image_size=args.resolution, random_flip=True, pairs=train_pairs)
    val_ds = PairedImageDataset(None, image_size=args.resolution, random_flip=False, pairs=val_pairs)

    dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=min(args.train_batch_size, len(val_ds)), shuffle=False, num_workers=args.num_workers, drop_last=False)
    val_batch = next(iter(val_dl))

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(dl) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    controlnet, optimizer, dl, lr_scheduler = accelerator.prepare(controlnet, optimizer, dl, lr_scheduler)

    text_inputs = tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

    global_step = 0

    for _epoch in range(num_train_epochs):
        for _step, batch in enumerate(dl):
            with accelerator.accumulate(controlnet):
                cond = batch["conditioning"].to(accelerator.device)
                target = batch["target"].to(accelerator.device)

                with torch.no_grad():
                    latents = vae.encode(target).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                cond_01 = (cond * 0.5 + 0.5).clamp(0, 1)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.repeat(bsz, 1, 1),
                    controlnet_cond=cond_01,
                    return_dict=False,
                )

                with torch.no_grad():
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states.repeat(bsz, 1, 1),
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                loss = torch.mean((model_pred - noise) ** 2)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and global_step % 50 == 0:
                accelerator.print(f"step {global_step}: loss={loss.detach().float().item():.4f}")

            if accelerator.is_main_process and global_step % args.validation_steps == 0 and global_step > 0:
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    controlnet=accelerator.unwrap_model(controlnet),
                    safety_checker=None,
                )
                metrics = run_validation_and_metrics(
                    accelerator=accelerator,
                    pipe=pipe,
                    val_batch=val_batch,
                    prompt=args.prompt,
                    step=global_step,
                    out_dir=args.output_dir,
                )
                accelerator.print(
                    f"[val] step={global_step} l1={metrics['l1']:.4f} psnr={metrics['psnr']:.2f} ssim={metrics['ssim']:.4f}"
                )
                with open(os.path.join(args.output_dir, "metrics.csv"), "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        global_step,
                        f"{loss.detach().float().item():.6f}",
                        f"{metrics['l1']:.6f}",
                        f"{metrics['psnr']:.6f}",
                        f"{metrics['ssim']:.6f}",
                    ])
                del pipe

            if accelerator.is_main_process and global_step % 500 == 0 and global_step > 0:
                save_path = os.path.join(args.output_dir, f"controlnet_step_{global_step:06d}")
                accelerator.unwrap_model(controlnet).save_pretrained(save_path)

            global_step += 1
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "controlnet_final")
        accelerator.unwrap_model(controlnet).save_pretrained(final_path)


if __name__ == "__main__":
    main()
