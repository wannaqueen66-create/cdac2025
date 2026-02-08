"""SA-Img2Img: conditional GAN (pix2pix-like) with Self-Attention + fixed val split and metrics."""

import argparse
import csv
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from common_dataset import PairedImageDataset, discover_pairs, split_pairs
from metrics import l1 as metric_l1, psnr as metric_psnr, ssim_global


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w)
        k = self.key(x).view(b, -1, h * w)
        v = self.value(x).view(b, -1, h * w)

        attn = torch.bmm(q.permute(0, 2, 1), k)
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
    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()
        self.d1 = DownBlock(in_ch, base, norm=False)
        self.d2 = DownBlock(base, base * 2)
        self.d3 = DownBlock(base * 2, base * 4)
        self.d4 = DownBlock(base * 4, base * 8)
        self.d5 = DownBlock(base * 8, base * 8)

        self.sa = SelfAttention2d(base * 8)

        self.u1 = UpBlock(base * 8, base * 8, dropout=True)
        self.u2 = UpBlock(base * 16, base * 4, dropout=False)
        self.u3 = UpBlock(base * 8, base * 2, dropout=False)
        self.u4 = UpBlock(base * 4, base, dropout=False)

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
    val_ratio: float
    split_seed: int
    eval_every: int
    save_every: int


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_samples(out_dir: str, tag: str, cond, fake, real):
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    grid = make_grid(torch.cat([denorm(cond), denorm(fake), denorm(real)], dim=0), nrow=cond.shape[0])
    save_image(grid, os.path.join(out_dir, "samples", f"{tag}.png"))


@torch.no_grad()
def evaluate(G, val_loader, device, out_dir: str, step: int):
    G.eval()
    l1_vals, psnr_vals, ssim_vals = [], [], []
    first = True

    for batch in val_loader:
        cond = batch["conditioning"].to(device)
        real = batch["target"].to(device)
        fake = G(cond)

        fake01 = denorm(fake)
        real01 = denorm(real)

        l1_vals.append(metric_l1(fake01, real01).item())
        psnr_vals.append(metric_psnr(fake01, real01).item())
        ssim_vals.append(ssim_global(fake01, real01).item())

        if first:
            save_samples(out_dir, f"val_step_{step:06d}", cond[:4].cpu(), fake[:4].cpu(), real[:4].cpu())
            first = False

    G.train()
    return {
        "l1": float(np.mean(l1_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
    }


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
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)
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
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )

    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "checkpoints"), exist_ok=True)

    pairs = discover_pairs(cfg.data_root)
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=cfg.val_ratio, seed=cfg.split_seed)

    with open(os.path.join(cfg.out_dir, "val_split.txt"), "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(p.name + "\n")

    run_meta = {
        **asdict(cfg),
        "git_commit": get_git_commit(),
        "num_total": len(pairs),
        "num_train": len(train_pairs),
        "num_val": len(val_pairs),
    }
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    train_ds = PairedImageDataset(None, image_size=cfg.image_size, random_flip=True, pairs=train_pairs)
    val_ds = PairedImageDataset(None, image_size=cfg.image_size, random_flip=False, pairs=val_pairs)

    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=min(cfg.batch_size, len(val_ds)), shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    G = GeneratorUNetSA().to(cfg.device)
    D = DiscriminatorPatch().to(cfg.device)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    metrics_path = os.path.join(cfg.out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "d_loss", "g_loss", "g_adv", "g_l1", "val_l1", "val_psnr", "val_ssim"])

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        for batch in dl:
            cond = batch["conditioning"].to(cfg.device)
            real = batch["target"].to(cfg.device)

            with torch.no_grad():
                fake = G(cond)

            pred_real = D(real, cond)
            pred_fake = D(fake.detach(), cond)
            d_loss = 0.5 * (bce(pred_real, torch.ones_like(pred_real)) + bce(pred_fake, torch.zeros_like(pred_fake)))

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            fake = G(cond)
            pred_fake = D(fake, cond)
            g_adv = bce(pred_fake, torch.ones_like(pred_fake))
            g_l1 = l1_loss(fake, real) * cfg.lambda_l1
            g_loss = g_adv + g_l1

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            if step % 50 == 0:
                print(f"epoch={epoch} step={step} d={d_loss.item():.4f} g={g_loss.item():.4f} adv={g_adv.item():.4f} l1={g_l1.item():.4f}")

            if step % cfg.eval_every == 0:
                val_metrics = evaluate(G, val_dl, cfg.device, cfg.out_dir, step)
                print(f"[val] step={step} l1={val_metrics['l1']:.4f} psnr={val_metrics['psnr']:.2f} ssim={val_metrics['ssim']:.4f}")
                with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        step,
                        epoch,
                        f"{d_loss.item():.6f}",
                        f"{g_loss.item():.6f}",
                        f"{g_adv.item():.6f}",
                        f"{g_l1.item():.6f}",
                        f"{val_metrics['l1']:.6f}",
                        f"{val_metrics['psnr']:.6f}",
                        f"{val_metrics['ssim']:.6f}",
                    ])

            if step % cfg.save_every == 0 and step > 0:
                ckpt = {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "cfg": run_meta,
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, "checkpoints", f"ckpt_{step:06d}.pt"))

            step += 1

    ckpt = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "step": step,
        "epoch": cfg.epochs,
        "cfg": run_meta,
    }
    torch.save(ckpt, os.path.join(cfg.out_dir, "checkpoints", "ckpt_final.pt"))


if __name__ == "__main__":
    main()
