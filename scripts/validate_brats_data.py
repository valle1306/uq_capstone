"""Validate prepared BRATS data by checking files and loading samples.

Usage:
  python scripts\validate_brats_data.py --data_root data\brats --n_samples 5
"""
import os
import argparse
import numpy as np
import csv
from pathlib import Path


def check_csv_and_files(csv_path, data_root):
    """Check that CSV exists and all referenced files exist."""
    if not os.path.exists(csv_path):
        return None, f"CSV not found: {csv_path}"
    
    missing_files = []
    valid_records = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            img_rel = row['image_path']
            mask_rel = row['mask_path']
            
            img_path = os.path.join(data_root, img_rel)
            mask_path = os.path.join(data_root, mask_rel)
            
            img_exists = os.path.exists(img_path)
            mask_exists = os.path.exists(mask_path)
            
            if not img_exists or not mask_exists:
                missing_files.append({
                    'row': i,
                    'image': img_rel,
                    'image_exists': img_exists,
                    'mask': mask_rel,
                    'mask_exists': mask_exists
                })
            else:
                valid_records.append((img_path, mask_path))
    
    return valid_records, missing_files


def load_and_check_npz(img_path, mask_path):
    """Load .npz files and check their contents."""
    try:
        img_data = np.load(img_path)
        mask_data = np.load(mask_path)
        
        if 'im' not in img_data:
            return False, f"Image missing 'im' key. Keys: {list(img_data.keys())}"
        if 'mask' not in mask_data:
            return False, f"Mask missing 'mask' key. Keys: {list(mask_data.keys())}"
        
        img = img_data['im']
        mask = mask_data['mask']
        
        # Check shapes
        if img.ndim not in [2, 3]:
            return False, f"Image has unexpected ndim: {img.ndim}, shape: {img.shape}"
        
        if mask.ndim != 2:
            return False, f"Mask has unexpected ndim: {mask.ndim}, shape: {mask.shape}"
        
        # Check value ranges
        if img.min() < 0 or img.max() > 1.1:
            return False, f"Image values outside [0,1]: min={img.min():.3f}, max={img.max():.3f}"
        
        mask_unique = np.unique(mask)
        if not np.all(np.isin(mask_unique, [0, 1])):
            return False, f"Mask has unexpected values: {mask_unique}"
        
        info = {
            'img_shape': img.shape,
            'img_dtype': str(img.dtype),
            'img_min': float(img.min()),
            'img_max': float(img.max()),
            'img_mean': float(img.mean()),
            'mask_shape': mask.shape,
            'mask_dtype': str(mask.dtype),
            'mask_unique': mask_unique.tolist(),
            'tumor_pixels': int(mask.sum()),
            'tumor_fraction': float(mask.mean())
        }
        
        return True, info
        
    except Exception as e:
        return False, f"Error loading files: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Validate prepared BRATS dataset')
    parser.add_argument('--data_root', required=True, help='Root directory with train.csv, val.csv, test.csv')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of samples to inspect per split')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    print(f"\n{'='*70}")
    print(f"BRATS Dataset Validation")
    print(f"{'='*70}")
    print(f"Data root: {data_root}")
    print(f"{'='*70}\n")
    
    # Check each split
    splits = ['train', 'val', 'test']
    all_valid = True
    
    for split in splits:
        csv_path = data_root / f'{split}.csv'
        print(f"\n{split.upper()} SET:")
        print(f"{'-'*70}")
        
        valid_records, missing = check_csv_and_files(csv_path, data_root)
        
        if valid_records is None:
            print(f"  ❌ {missing}")
            all_valid = False
            continue
        
        total_records = len(valid_records) + len(missing)
        print(f"  Total records: {total_records}")
        print(f"  Valid records: {len(valid_records)}")
        
        if missing:
            print(f"  ❌ Missing files: {len(missing)}")
            all_valid = False
            # Show first few missing
            for m in missing[:3]:
                print(f"     Row {m['row']}: img={m['image_exists']}, mask={m['mask_exists']}")
                print(f"       Image: {m['image']}")
                print(f"       Mask:  {m['mask']}")
        else:
            print(f"  ✓ All files exist")
        
        # Sample and check a few files
        if valid_records:
            n_check = min(args.n_samples, len(valid_records))
            import random
            random.seed(42)
            samples = random.sample(valid_records, n_check)
            
            print(f"\n  Checking {n_check} sample files:")
            for i, (img_path, mask_path) in enumerate(samples, 1):
                success, info = load_and_check_npz(img_path, mask_path)
                
                img_name = Path(img_path).name
                if success:
                    print(f"    [{i}] ✓ {img_name}")
                    print(f"        Image: {info['img_shape']} {info['img_dtype']}, "
                          f"range=[{info['img_min']:.3f}, {info['img_max']:.3f}]")
                    print(f"        Mask:  {info['mask_shape']} {info['mask_dtype']}, "
                          f"tumor={info['tumor_pixels']} pixels ({info['tumor_fraction']*100:.1f}%)")
                else:
                    print(f"    [{i}] ❌ {img_name}")
                    print(f"        Error: {info}")
                    all_valid = False
    
    # Final summary
    print(f"\n{'='*70}")
    if all_valid:
        print("✓ VALIDATION PASSED: Dataset is ready for training!")
    else:
        print("❌ VALIDATION FAILED: Please fix the issues above")
    print(f"{'='*70}\n")
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    exit(main())
