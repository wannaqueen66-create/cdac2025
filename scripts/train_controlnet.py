"""Train ControlNet for heatmap -> roof plan (paired).

This script is a compact wrapper around Diffusers components. It fine-tunes a
ControlNet on a small paired dataset where:
- conditioning image: heatmap
- target image: roof plan

For small data (40 pairs), treat this as a starting point:
- keep learning rate low
- consider fewer steps and frequent validation
- consider freezing more components (by default we train ControlNet only)

Notes:
- You still need a text prompt. Use a fixed prompt like "a roof plan".
"""

import argparse
import os
import math
import random

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

from common_dataset import PairedImageDataset


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
    return p.parse_args()


@torch.no_grad()
def log_validation(accelerator: Accelerator, pipe: StableDiffusionControlNetPipeline, batch, step: int, out_dir: str):
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    cond = batch["conditioning"]
    # dataset returns [-1,1], pipeline expects PIL or [0,1] tensor; convert
    cond_01 = (cond * 0.5 + 0.5).clamp(0, 1)

    images = pipe(
        prompt=["a roof plan"] * cond_01.shape[0],
        image=cond_01,
        num_inference_steps=30,
        guidance_scale=7.0,
    ).images

    os.makedirs(os.path.join(out_dir, "validation"), exist_ok=True)
    for i, im in enumerate(images):
        im.save(os.path.join(out_dir, "validation", f"step_{step:06d}_{i}.png"))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )

    set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Initialize ControlNet from the UNet weights (common starting point)
    controlnet = ControlNetModel.from_unet(unet)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Freeze everything except ControlNet
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    controlnet.train()

    ds = PairedImageDataset(args.data_root, image_size=args.resolution, random_flip=True)
    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.learning_rate)

    # LR scheduler
    num_update_steps_per_epoch = math.ceil(len(dl) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare
    controlnet, optimizer, dl, lr_scheduler = accelerator.prepare(controlnet, optimizer, dl, lr_scheduler)

    # text embeddings for fixed prompt
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

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dl):
            with accelerator.accumulate(controlnet):
                cond = batch["conditioning"].to(accelerator.device)
                target = batch["target"].to(accelerator.device)

                # VAE encode target images to latents
                with torch.no_grad():
                    latents = vae.encode(target).latent_dist.sample() * vae.config.scaling_factor

                # sample noise & timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ControlNet expects conditioning image in [0,1]; our dataset is [-1,1]
                cond_01 = (cond * 0.5 + 0.5).clamp(0, 1)

                # ControlNet forward
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.repeat(bsz, 1, 1),
                    controlnet_cond=cond_01,
                    return_dict=False,
                )

                # UNet forward (frozen) to predict noise
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
                # create pipeline for validation
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    controlnet=accelerator.unwrap_model(controlnet),
                    safety_checker=None,
                )
                pipe = pipe.to(accelerator.device)
                log_validation(accelerator, pipe, batch, global_step, args.output_dir)
                del pipe

            if accelerator.is_main_process and global_step % 500 == 0 and global_step > 0:
                # save checkpoint
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
