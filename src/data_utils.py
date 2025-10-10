import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class BraTSSegmentationDataset(Dataset):
    """
    Dataset for BraTS segmentation from NPZ files.
    Each NPZ file contains 'image' (H, W) and 'mask' (H, W) arrays.
    """
    def __init__(self, npz_dir, augment=False):
        """
        Args:
            npz_dir: Directory containing .npz files
            augment: Whether to apply data augmentation
        """
        self.npz_files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
        self.augment = augment
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found in {npz_dir}")
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        # Load NPZ file
        data = np.load(self.npz_files[idx])
        image = data['image']  # (H, W)
        mask = data['mask']    # (H, W)
        
        # Convert to torch tensors and add channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # (1, H, W)
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image / image.max()
        
        # Ensure mask is binary
        mask = (mask > 0).float()
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
        
        return {
            'image': image,
            'mask': mask
        }


class CXRFolderDataset(Dataset):
    """Simple dataset expecting a CSV with `image_path,label` and image files under a root."""
    def __init__(self, csv_path, root_dir, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['label'])
        return img, label
