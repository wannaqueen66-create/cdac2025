"""Preprocess paired dataset: auto-crop roof white borders and sync-crop heatmaps.

Assumptions:
- roof images may contain large white margins.
- heatmap images correspond spatially to roof images, but have different resolution.
- filenames are paired 1-to-1 under:
    DATA_ROOT/heatmap/*.png
    DATA_ROOT/roof/*.png

This script:
1) finds a bounding box on roof image where pixels are not near-white.
2) applies padding.
3) scales bbox coordinates from roof space -> heatmap space.
4) crops both roof and heatmap.
5) pads to square and resizes to target size.
6) writes processed images to OUTPUT_ROOT/heatmap and OUTPUT_ROOT/roof.

Example:
python scripts/preprocess_dataset.py \
  --data_root /path/to/DATA_ROOT \
  --out_root /path/to/DATA_ROOT_CROPPED \
  --out_size 512 \
  --white_thresh 245 \
  --pad 16
"""

import argparse
import os
from typing import Tuple

from PIL import Image


def find_nonwhite_bbox(img: Image.Image, white_thresh: int) -> Tuple[int, int, int, int]:
    """Return bbox (left, top, right, bottom) of pixels with gray < white_thresh."""
    g = img.convert("L")
    px = g.load()
    w, h = g.size
    minx, miny = w, h
    maxx, maxy = -1, -1
    for y in range(h):
        for x in range(w):
            if px[x, y] < white_thresh:
                if x < minx: minx = x
                if y < miny: miny = y
                if x > maxx: maxx = x
                if y > maxy: maxy = y
    if maxx < 0:
        # all white; fallback to full image
        return (0, 0, w, h)
    # PIL bbox right/bottom are exclusive
    return (minx, miny, maxx + 1, maxy + 1)


def clamp_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    l, t, r, btm = b
    l = max(0, min(l, w))
    r = max(0, min(r, w))
    t = max(0, min(t, h))
    btm = max(0, min(btm, h))
    if r <= l: r = min(w, l + 1)
    if btm <= t: btm = min(h, t + 1)
    return (l, t, r, btm)


def pad_to_square(img: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill)
    # center
    x0 = (side - w) // 2
    y0 = (side - h) // 2
    out.paste(img, (x0, y0))
    return out


def process_pair(roof_path: str, heat_path: str, out_size: int, white_thresh: int, pad: int):
    roof = Image.open(roof_path).convert("RGB")
    heat = Image.open(heat_path).convert("RGB")

    rw, rh = roof.size
    hw, hh = heat.size

    # bbox on roof
    l, t, r, btm = find_nonwhite_bbox(roof, white_thresh)
    l -= pad; t -= pad; r += pad; btm += pad
    l, t, r, btm = clamp_bbox((l, t, r, btm), rw, rh)

    roof_crop = roof.crop((l, t, r, btm))

    # scale bbox to heatmap space
    sx = hw / rw
    sy = hh / rh
    hl = int(round(l * sx))
    ht = int(round(t * sy))
    hr = int(round(r * sx))
    hb = int(round(btm * sy))
    hl, ht, hr, hb = clamp_bbox((hl, ht, hr, hb), hw, hh)
    heat_crop = heat.crop((hl, ht, hr, hb))

    # pad to square then resize
    roof_sq = pad_to_square(roof_crop, fill=(255, 255, 255))
    heat_sq = pad_to_square(heat_crop, fill=(0, 0, 0))

    roof_out = roof_sq.resize((out_size, out_size), resample=Image.Resampling.LANCZOS)
    heat_out = heat_sq.resize((out_size, out_size), resample=Image.Resampling.NEAREST)

    return roof_out, heat_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--out_size", type=int, default=512)
    ap.add_argument("--white_thresh", type=int, default=245)
    ap.add_argument("--pad", type=int, default=16)
    args = ap.parse_args()

    hm_dir = os.path.join(args.data_root, "heatmap")
    rf_dir = os.path.join(args.data_root, "roof")
    if not os.path.isdir(hm_dir) or not os.path.isdir(rf_dir):
        raise SystemExit("Expected data_root/heatmap and data_root/roof")

    out_hm = os.path.join(args.out_root, "heatmap")
    out_rf = os.path.join(args.out_root, "roof")
    os.makedirs(out_hm, exist_ok=True)
    os.makedirs(out_rf, exist_ok=True)

    files = sorted([f for f in os.listdir(hm_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not files:
        raise SystemExit("No heatmap images found")

    missing = []
    for fn in files:
        if not os.path.exists(os.path.join(rf_dir, fn)):
            missing.append(fn)
    if missing:
        raise SystemExit(f"Missing roof images for {len(missing)} heatmaps. Example: {missing[:5]}")

    for i, fn in enumerate(files, 1):
        roof_path = os.path.join(rf_dir, fn)
        heat_path = os.path.join(hm_dir, fn)
        roof_out, heat_out = process_pair(roof_path, heat_path, args.out_size, args.white_thresh, args.pad)
        roof_out.save(os.path.join(out_rf, fn))
        heat_out.save(os.path.join(out_hm, fn))
        if i % 10 == 0:
            print(f"processed {i}/{len(files)}")

    print("done")


if __name__ == "__main__":
    main()
