"""Check paired dataset consistency for DATA_ROOT/{heatmap,roof}.

Reports:
- files existing only in heatmap
- files existing only in roof
- extension summary

Exit code:
- 0 if perfectly matched
- 1 if mismatch found

Usage:
python scripts/check_dataset_pairs.py --data_root /path/to/DATA_ROOT
"""

import argparse
import os
from typing import Set

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _stem_set(folder: str) -> Set[str]:
    stems = set()
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            continue
        stem, ext = os.path.splitext(fn)
        if ext.lower() in IMG_EXTS:
            stems.add(stem)
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()

    hm_dir = os.path.join(args.data_root, "heatmap")
    rf_dir = os.path.join(args.data_root, "roof")

    if not os.path.isdir(hm_dir) or not os.path.isdir(rf_dir):
        raise SystemExit("Expected data_root/heatmap and data_root/roof")

    hm_stems = _stem_set(hm_dir)
    rf_stems = _stem_set(rf_dir)

    only_hm = sorted(hm_stems - rf_stems)
    only_rf = sorted(rf_stems - hm_stems)
    inter = sorted(hm_stems & rf_stems)

    print(f"heatmap count: {len(hm_stems)}")
    print(f"roof count:    {len(rf_stems)}")
    print(f"paired count:  {len(inter)}")

    mismatch = False
    if only_hm:
        mismatch = True
        print(f"\n[ERROR] only in heatmap ({len(only_hm)}), examples: {only_hm[:10]}")
    if only_rf:
        mismatch = True
        print(f"\n[ERROR] only in roof ({len(only_rf)}), examples: {only_rf[:10]}")

    if mismatch:
        raise SystemExit(1)

    print("\n[OK] dataset pairs are fully matched by filename stem.")


if __name__ == "__main__":
    main()
