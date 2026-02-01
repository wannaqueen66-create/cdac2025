"""SA-Img2Img: a minimal conditional GAN (pix2pix-like) with Self-Attention.

Input: heatmap (3ch)  -> Output: roof (3ch)

This is intentionally lightweight and hackable for small data (40 pairs).
"""

import argparse
import os
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from common_dataset import PairedImageDataset


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w)          # (b, c//8, hw)
        k = self.key(x).view(b, -1, h * w)            # (b, c//8, hw)
        v = self.value(x).view(b, -1, h * w)          # (b, c, hw)

        attn = torch.bmm(q.permute(0, 2, 1), k)        # (b, hw, hw)
        attn = F.softmax(attn / math.sqrt(q.shape[1]), dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.gamma * out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GeneratorUNetSA(nn.Module):
    """U-Net generator with a self-attention bottleneck."""

    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()
        # Encoder
        self.d1 = DownBlock(in_ch, base, norm=False)   # 128
        self.d2 = DownBlock(base, base * 2)            # 64
        self.d3 = DownBlock(base * 2, base * 4)        # 32
        self.d4 = DownBlock(base * 4, base * 8)        # 16
        self.d5 = DownBlock(base * 8, base * 8)        # 8

        self.sa = SelfAttention2d(base * 8)

        # Decoder
        self.u1 = UpBlock(base * 8, base * 8, dropout=True)   # 16
        self.u2 = UpBlock(base * 16, base * 4, dropout=False) # 32
        self.u3 = UpBlock(base * 8, base * 2, dropout=False)  # 64
        self.u4 = UpBlock(base * 4, base, dropout=False)      # 128

        self.out = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        b = self.sa(d5)

        u1 = self.u1(b)
        u1 = torch.cat([u1, d4], dim=1)
        u2 = self.u2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u3 = self.u3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.u4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        return self.out(u4)


class DiscriminatorPatch(nn.Module):
    """PatchGAN discriminator, conditioned on input heatmap."""

    def __init__(self, in_ch=3, cond_ch=3, base=64):
        super().__init__()
        ch = in_ch + cond_ch
        self.net = nn.Sequential(
            nn.Conv2d(ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 4, base * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, img, cond):
        x = torch.cat([img, cond], dim=1)
        return self.net(x)


@dataclass
class TrainCfg:
    data_root: str
    out_dir: str
    image_size: int
    batch_size: int
    lr: float
    epochs: int
    lambda_l1: float
    num_workers: int
    seed: int
    device: str


def save_samples(out_dir: str, step: int, cond, fake, real):
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    # [-1,1] -> [0,1]
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)

    grid = make_grid(torch.cat([denorm(cond), denorm(fake), denorm(real)], dim=0), nrow=cond.shape[0])
    save_image(grid, os.path.join(out_dir, "samples", f"step_{step:06d}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lambda_l1", type=float, default=100.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainCfg(
        data_root=args.data_root,
        out_dir=args.out_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        lambda_l1=args.lambda_l1,
        num_workers=args.num_workers,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "checkpoints"), exist_ok=True)

    ds = PairedImageDataset(cfg.data_root, image_size=cfg.image_size, random_flip=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    G = GeneratorUNetSA().to(cfg.device)
    D = DiscriminatorPatch().to(cfg.device)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        for batch in dl:
            cond = batch["conditioning"].to(cfg.device)
            real = batch["target"].to(cfg.device)

            # ----------------- Train D -----------------
            with torch.no_grad():
                fake = G(cond)

            pred_real = D(real, cond)
            pred_fake = D(fake.detach(), cond)
            d_loss = 0.5 * (bce(pred_real, torch.ones_like(pred_real)) + bce(pred_fake, torch.zeros_like(pred_fake)))

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # ----------------- Train G -----------------
            fake = G(cond)
            pred_fake = D(fake, cond)
            g_adv = bce(pred_fake, torch.ones_like(pred_fake))
            g_l1 = l1(fake, real) * cfg.lambda_l1
            g_loss = g_adv + g_l1

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            if step % 200 == 0:
                print(f"epoch={epoch} step={step} d={d_loss.item():.4f} g={g_loss.item():.4f} adv={g_adv.item():.4f} l1={g_l1.item():.4f}")
                save_samples(cfg.out_dir, step, cond[:4].cpu(), fake[:4].cpu(), real[:4].cpu())

            if step % 2000 == 0 and step > 0:
                ckpt = {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "cfg": cfg.__dict__,
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, "checkpoints", f"ckpt_{step:06d}.pt"))

            step += 1

    # final
    ckpt = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "step": step,
        "epoch": cfg.epochs,
        "cfg": cfg.__dict__,
    }
    torch.save(ckpt, os.path.join(cfg.out_dir, "checkpoints", "ckpt_final.pt"))


if __name__ == "__main__":
    main()
