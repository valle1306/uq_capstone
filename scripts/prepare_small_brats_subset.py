"""Prepare a small subset of BRATS data for initial experiments.

This script:
1. Selects a small number of patients (default 25)
2. Converts NIfTI volumes to 2D slices in .npz format
3. Creates train/val/test splits
4. Validates the data

Usage (Windows):
  python scripts\prepare_small_brats_subset.py --brats_root "BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData" --out_dir data\brats --n_patients 25

Usage (Amarel/Linux):
  python scripts/prepare_small_brats_subset.py --brats_root /path/to/BraTS --out_dir /scratch/$USER/brats --n_patients 25
"""
import os
import argparse
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from glob import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--brats_root', required=True, 
                   help='Root folder of BRATS dataset (contains BraTS20_Training_XXX folders)')
    p.add_argument('--out_dir', default='data/brats', 
                   help='Output directory for .npz files and CSVs')
    p.add_argument('--n_patients', type=int, default=25, 
                   help='Number of patients to process (for small experiments)')
    p.add_argument('--modality', default='t1ce', 
                   help='Modality to extract: t1ce, t1, t2, flair')
    p.add_argument('--val_frac', type=float, default=0.15, 
                   help='Fraction for validation set')
    p.add_argument('--test_frac', type=float, default=0.15, 
                   help='Fraction for test set')
    p.add_argument('--slice_stride', type=int, default=3, 
                   help='Stride for selecting slices (higher = fewer slices)')
    p.add_argument('--min_tumor_pixels', type=int, default=50, 
                   help='Minimum tumor pixels per slice to include')
    p.add_argument('--seed', type=int, default=42, 
                   help='Random seed for reproducibility')
    return p.parse_args()


def find_modality_file(patient_dir, modality):
    """Find the NIfTI file for a specific modality."""
    pattern = os.path.join(patient_dir, f'*_{modality}.nii*')
    files = glob(pattern)
    return files[0] if files else None


def find_seg_file(patient_dir):
    """Find the segmentation file."""
    pattern = os.path.join(patient_dir, '*_seg.nii*')
    files = glob(pattern)
    return files[0] if files else None


def normalize_slice(img_slice):
    """Normalize image slice to [0, 1] range."""
    img_min = img_slice.min()
    img_max = img_slice.max()
    if img_max - img_min > 0:
        return (img_slice - img_min) / (img_max - img_min)
    return img_slice


def process_patient(patient_dir, patient_name, modality, out_images, out_masks, 
                   slice_stride, min_tumor_pixels):
    """Process one patient: extract slices and save as .npz files.
    
    Returns:
        List of tuples (image_rel_path, mask_rel_path) for valid slices
    """
    # Find files
    img_file = find_modality_file(patient_dir, modality)
    seg_file = find_seg_file(patient_dir)
    
    if img_file is None or seg_file is None:
        print(f"  WARNING: Missing files for {patient_name}")
        return []
    
    # Load NIfTI files
    try:
        img_nii = nib.load(img_file)
        seg_nii = nib.load(seg_file)
    except Exception as e:
        print(f"  ERROR loading files for {patient_name}: {e}")
        return []
    
    img_data = img_nii.get_fdata()
    seg_data = seg_nii.get_fdata()
    
    # Check dimensions match
    if img_data.shape != seg_data.shape:
        print(f"  WARNING: Shape mismatch for {patient_name}")
        return []
    
    # Process slices (z-axis is typically depth)
    depth = img_data.shape[2]
    slice_records = []
    
    for z in range(0, depth, slice_stride):
        img_slice = img_data[:, :, z]
        seg_slice = seg_data[:, :, z]
        
        # Convert segmentation to binary (any tumor = 1)
        mask_slice = (seg_slice > 0).astype(np.uint8)
        
        # Skip if not enough tumor pixels
        if mask_slice.sum() < min_tumor_pixels:
            continue
        
        # Normalize image
        img_norm = normalize_slice(img_slice).astype(np.float32)
        
        # Save as .npz
        slice_name = f'{patient_name}_slice{z:03d}.npz'
        img_path = os.path.join(out_images, slice_name)
        mask_path = os.path.join(out_masks, slice_name)
        
        # Save with 'im' and 'mask' keys to match expected format
        # Add channel dimension to image: (1, H, W)
        img_with_channel = img_norm[np.newaxis, :, :]  # Add channel dimension
        np.savez_compressed(img_path, im=img_with_channel)
        np.savez_compressed(mask_path, mask=mask_slice)
        
        # Store relative paths
        img_rel = os.path.join('images', slice_name)
        mask_rel = os.path.join('masks', slice_name)
        slice_records.append((img_rel, mask_rel))
    
    return slice_records


def write_csv(filepath, records):
    """Write CSV file with image_path,mask_path columns."""
    with open(filepath, 'w') as f:
        f.write('image_path,mask_path\n')
        for img_path, mask_path in records:
            f.write(f'{img_path},{mask_path}\n')
    print(f"  Wrote {filepath}: {len(records)} slices")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"\n{'='*60}")
    print(f"BRATS Small Subset Preparation")
    print(f"{'='*60}")
    print(f"Source: {args.brats_root}")
    print(f"Output: {args.out_dir}")
    print(f"Patients: {args.n_patients}")
    print(f"Modality: {args.modality}")
    print(f"Slice stride: {args.slice_stride}")
    print(f"Min tumor pixels: {args.min_tumor_pixels}")
    print(f"{'='*60}\n")
    
    # Find all patient directories
    brats_root = Path(args.brats_root)
    all_patients = [d for d in brats_root.iterdir() 
                   if d.is_dir() and d.name.startswith('BraTS20_Training')]
    all_patients.sort()
    
    print(f"Found {len(all_patients)} total patient folders")
    
    # Select random subset
    if args.n_patients < len(all_patients):
        selected_patients = random.sample(all_patients, args.n_patients)
        print(f"Randomly selected {args.n_patients} patients")
    else:
        selected_patients = all_patients
        print(f"Using all {len(all_patients)} patients")
    
    selected_patients.sort()
    
    # Create output directories
    out_dir = Path(args.out_dir)
    out_images = out_dir / 'images'
    out_masks = out_dir / 'masks'
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    
    # Process all patients
    all_records = []
    print(f"\nProcessing patients...")
    for i, patient_dir in enumerate(selected_patients, 1):
        patient_name = patient_dir.name
        print(f"[{i}/{len(selected_patients)}] {patient_name}...", end=' ')
        
        records = process_patient(
            str(patient_dir), patient_name, args.modality,
            str(out_images), str(out_masks),
            args.slice_stride, args.min_tumor_pixels
        )
        
        all_records.extend(records)
        print(f"{len(records)} slices")
    
    print(f"\nTotal slices extracted: {len(all_records)}")
    
    if len(all_records) == 0:
        print("ERROR: No valid slices extracted!")
        return
    
    # Split into train/val/test
    from sklearn.model_selection import train_test_split
    
    train_val_records, test_records = train_test_split(
        all_records, test_size=args.test_frac, random_state=args.seed
    )
    
    val_size = args.val_frac / (1 - args.test_frac)
    train_records, val_records = train_test_split(
        train_val_records, test_size=val_size, random_state=args.seed
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_records)} slices ({len(train_records)/len(all_records)*100:.1f}%)")
    print(f"  Val:   {len(val_records)} slices ({len(val_records)/len(all_records)*100:.1f}%)")
    print(f"  Test:  {len(test_records)} slices ({len(test_records)/len(all_records)*100:.1f}%)")
    
    # Write CSV files
    print(f"\nWriting CSV files to {out_dir}...")
    write_csv(out_dir / 'train.csv', train_records)
    write_csv(out_dir / 'val.csv', val_records)
    write_csv(out_dir / 'test.csv', test_records)
    
    # Write summary
    summary_path = out_dir / 'dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"BRATS Small Subset Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Source: {args.brats_root}\n")
        f.write(f"Generated: {Path.cwd()}\n")
        f.write(f"Patients: {args.n_patients}\n")
        f.write(f"Modality: {args.modality}\n")
        f.write(f"Slice stride: {args.slice_stride}\n")
        f.write(f"Min tumor pixels: {args.min_tumor_pixels}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"\nDataset splits:\n")
        f.write(f"  Train: {len(train_records)} slices\n")
        f.write(f"  Val:   {len(val_records)} slices\n")
        f.write(f"  Test:  {len(test_records)} slices\n")
        f.write(f"  Total: {len(all_records)} slices\n")
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Dataset prepared in: {out_dir}")
    print(f"Summary written to: {summary_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
