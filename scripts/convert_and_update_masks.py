#!/usr/bin/env python3
"""Convert PNG masks (e.g. NAME_mask.png) to compressed .npz (mask key)
and update train/val/test CSVs to reference the new mask .npz files.

Usage (on Amarel):
  conda activate nnunetv2   # or an env with numpy + pillow
  pip install --user pillow
  python scripts/convert_and_update_masks.py --data_root /scratch/$USER/brats

The script will create backups: train.csv.bak etc.
"""
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import csv


def convert_masks(data_root: Path):
    masks_dir = data_root / 'masks'
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks dir not found: {masks_dir}")
    pngs = sorted(masks_dir.glob('*.png'))
    converted = 0
    for p in pngs:
        name = p.stem  # e.g. BraTS20_Training_004_slice102_mask
        if name.endswith('_mask'):
            base = name[:-5]
        else:
            base = name
        out_npz = masks_dir / (base + '.npz')
        if out_npz.exists():
            continue
        # load png
        img = Image.open(p)
        arr = np.array(img)
        # if RGB, convert to single channel
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask_bin = (arr > 0).astype(np.uint8)
        np.savez_compressed(out_npz, mask=mask_bin)
        converted += 1
    return converted


def update_csvs(data_root: Path):
    csvs = ['train.csv', 'val.csv', 'test.csv']
    updated = {}
    for c in csvs:
        p = data_root / c
        if not p.exists():
            updated[c] = ('missing', 0)
            continue
        bak = p.with_suffix(p.suffix + '.bak')
        if not bak.exists():
            p.replace(bak)
            # restore original into variable orig_lines
            orig_lines = list(open(bak, newline=''))
            # now write back original to p to process reading normally
            with open(p, 'w', newline='') as fw:
                fw.writelines(orig_lines)
        # read and modify
        rows = []
        changed = 0
        total = 0
        with open(p, newline='') as fh:
            reader = csv.reader(fh)
            hdr = next(reader, None)
            for row in reader:
                total += 1
                if len(row) < 1:
                    rows.append(row); continue
                img_rel = row[0].strip()
                # compute expected mask npz path from image base name
                img_base = Path(img_rel).stem  # e.g. BraTS20_Training_004_slice102
                expected_mask_rel = os.path.join('masks', img_base + '.npz')
                expected_mask_abs = data_root / expected_mask_rel
                if expected_mask_abs.exists():
                    # ensure row has mask column and update it
                    if len(row) < 2 or row[1] != expected_mask_rel:
                        if len(row) < 2:
                            if len(row) == 1:
                                row.append(expected_mask_rel)
                            else:
                                row = [row[0], expected_mask_rel]
                        else:
                            row[1] = expected_mask_rel
                        changed += 1
                rows.append(row)
        # write back
        with open(p, 'w', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(['image_path','mask_path'])
            for r in rows:
                writer.writerow(r)
        updated[c] = (total, changed)
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        print('data_root not found:', data_root)
        return
    print('Converting PNG masks to .npz in', data_root / 'masks')
    n = convert_masks(data_root)
    print('Converted masks:', n)
    print('Updating CSVs to point to .npz masks')
    upd = update_csvs(data_root)
    for k,v in upd.items():
        print('CSV', k, '->', v)
    print('Done')


if __name__ == '__main__':
    main()
