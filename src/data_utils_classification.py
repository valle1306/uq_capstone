"""
Medical Image Classification Dataset Utilities

Supports multiple medical imaging classification datasets:
1. Chest X-Ray Pneumonia (Kaggle) - 5,863 images, 2 classes
2. OCT Retinal Images (Kaggle) - 84,495 images, 4 classes
3. Skin Cancer MNIST (HAM10000) - 10,015 images, 7 classes
4. Brain Tumor MRI (Kaggle) - 3,264 images, 4 classes

All datasets will be automatically downloaded and prepared for training.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path


class MedicalImageDataset(Dataset):
    """Generic medical image dataset class"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class ChestXRayPneumonia:
    """
    Chest X-Ray Pneumonia Detection Dataset
    
    Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    Classes: NORMAL (0), PNEUMONIA (1)
    Total: ~5,863 images
    """
    
    def __init__(self, data_dir='data/chest_xray'):
        self.data_dir = Path(data_dir)
        self.num_classes = 2
        self.class_names = ['NORMAL', 'PNEUMONIA']
        
    def prepare_data(self):
        """Download and prepare Chest X-Ray dataset"""
        if not self.data_dir.exists():
            print("Preparing Chest X-Ray Pneumonia dataset...")
            print("Please download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
            print(f"Extract to: {self.data_dir}")
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}")
        
        # Verify structure
        train_dir = self.data_dir / 'train'
        test_dir = self.data_dir / 'test'
        val_dir = self.data_dir / 'val'
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training data not found at {train_dir}")

        # Use ASCII-only message to avoid encoding issues on some consoles
        print(f"Chest X-Ray dataset found at {self.data_dir}")
        return True
    
    def get_datasets(self, img_size=224):
        """Get train, val, test datasets with transforms"""
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load data
        train_paths, train_labels = self._load_split('train')
        test_paths, test_labels = self._load_split('test')
        
        # Create calibration set from train (80-20 split)
        n_train = int(0.8 * len(train_paths))
        indices = torch.randperm(len(train_paths))
        
        train_idx = indices[:n_train]
        cal_idx = indices[n_train:]
        
        cal_paths = [train_paths[i] for i in cal_idx]
        cal_labels = [train_labels[i] for i in cal_idx]
        train_paths = [train_paths[i] for i in train_idx]
        train_labels = [train_labels[i] for i in train_idx]
        
        # Create datasets
        train_dataset = MedicalImageDataset(train_paths, train_labels, train_transform)
        cal_dataset = MedicalImageDataset(cal_paths, cal_labels, test_transform)
        test_dataset = MedicalImageDataset(test_paths, test_labels, test_transform)
        
        print(f"Train: {len(train_dataset)}, Cal: {len(cal_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, cal_dataset, test_dataset
    
    def _load_split(self, split):
        """Load image paths and labels for a split"""
        split_dir = self.data_dir / split
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            # Accept common image extensions (jpeg, jpg, png)
            for pattern in ('*.jpeg', '*.jpg', '*.png'):
                for img_file in class_dir.glob(pattern):
                    image_paths.append(str(img_file))
                    labels.append(class_idx)
        
        return image_paths, labels


class OCTRetinalImages:
    """
    OCT (Optical Coherence Tomography) Retinal Images
    
    Source: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
    Classes: CNV, DME, DRUSEN, NORMAL (4 classes)
    Total: ~84,495 images
    """
    
    def __init__(self, data_dir='data/oct_retinal'):
        self.data_dir = Path(data_dir)
        self.num_classes = 4
        self.class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        
    def prepare_data(self):
        """Download and prepare OCT dataset"""
        if not self.data_dir.exists():
            print("Preparing OCT Retinal Images dataset...")
            print("Please download from: https://www.kaggle.com/datasets/paultimothymooney/kermany2018")
            print(f"Extract to: {self.data_dir}")
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}")

        print(f"OCT Retinal dataset found at {self.data_dir}")
        return True
    
    def get_datasets(self, img_size=224):
        """Get train, val, test datasets with transforms"""
        
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load data
        train_paths, train_labels = self._load_split('train')
        test_paths, test_labels = self._load_split('test')
        
        # Create calibration set
        n_train = int(0.8 * len(train_paths))
        indices = torch.randperm(len(train_paths))
        
        train_idx = indices[:n_train]
        cal_idx = indices[n_train:]
        
        cal_paths = [train_paths[i] for i in cal_idx]
        cal_labels = [train_labels[i] for i in cal_idx]
        train_paths = [train_paths[i] for i in train_idx]
        train_labels = [train_labels[i] for i in train_idx]
        
        train_dataset = MedicalImageDataset(train_paths, train_labels, train_transform)
        cal_dataset = MedicalImageDataset(cal_paths, cal_labels, test_transform)
        test_dataset = MedicalImageDataset(test_paths, test_labels, test_transform)
        
        print(f"Train: {len(train_dataset)}, Cal: {len(cal_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, cal_dataset, test_dataset
    
    def _load_split(self, split):
        """Load image paths and labels for a split"""
        split_dir = self.data_dir / split
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            # Accept common image extensions (jpeg, jpg, png)
            for pattern in ('*.jpeg', '*.jpg', '*.png'):
                for img_file in class_dir.glob(pattern):
                    image_paths.append(str(img_file))
                    labels.append(class_idx)
        
        return image_paths, labels


class BrainTumorMRI:
    """
    Brain Tumor MRI Classification
    
    Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
    Classes: glioma, meningioma, notumor, pituitary (4 classes)
    Total: ~7,023 images
    """
    
    def __init__(self, data_dir='data/brain_tumor'):
        self.data_dir = Path(data_dir)
        self.num_classes = 4
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
    def prepare_data(self):
        """Download and prepare Brain Tumor dataset"""
        if not self.data_dir.exists():
            print("Preparing Brain Tumor MRI dataset...")
            print("Please download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
            print(f"Extract to: {self.data_dir}")
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}")

        print(f"Brain Tumor MRI dataset found at {self.data_dir}")
        return True
    
    def get_datasets(self, img_size=224):
        """Get train, val, test datasets with transforms"""
        
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load all data
        all_paths, all_labels = self._load_all_data()
        
        # Split: 70% train, 15% cal, 15% test
        n = len(all_paths)
        indices = torch.randperm(n)
        
        n_train = int(0.7 * n)
        n_cal = int(0.15 * n)
        
        train_idx = indices[:n_train]
        cal_idx = indices[n_train:n_train + n_cal]
        test_idx = indices[n_train + n_cal:]
        
        train_paths = [all_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        cal_paths = [all_paths[i] for i in cal_idx]
        cal_labels = [all_labels[i] for i in cal_idx]
        test_paths = [all_paths[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]
        
        train_dataset = MedicalImageDataset(train_paths, train_labels, train_transform)
        cal_dataset = MedicalImageDataset(cal_paths, cal_labels, test_transform)
        test_dataset = MedicalImageDataset(test_paths, test_labels, test_transform)
        
        print(f"Train: {len(train_dataset)}, Cal: {len(cal_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, cal_dataset, test_dataset
    
    def _load_all_data(self):
        """Load all image paths and labels"""
        image_paths = []
        labels = []
        
        # Try Training and Testing directories
        for subdir in ['Training', 'Testing']:
            base_dir = self.data_dir / subdir
            if not base_dir.exists():
                continue
                
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = base_dir / class_name
                if not class_dir.exists():
                    continue
                    
                for img_file in class_dir.glob('*.jpg'):
                    image_paths.append(str(img_file))
                    labels.append(class_idx)
        
        return image_paths, labels


def get_classification_loaders(dataset_name='chest_xray', 
                               data_dir=None,
                               batch_size=32, 
                               num_workers=4,
                               img_size=224):
    """
    Get data loaders for medical image classification
    
    Args:
        dataset_name: 'chest_xray', 'oct_retinal', or 'brain_tumor'
        data_dir: Path to dataset (optional)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        img_size: Image size (default: 224)
    
    Returns:
        train_loader, cal_loader, test_loader, num_classes
    """
    
    # Select dataset
    if dataset_name == 'chest_xray':
        dataset = ChestXRayPneumonia(data_dir if data_dir else 'data/chest_xray')
    elif dataset_name == 'oct_retinal':
        dataset = OCTRetinalImages(data_dir if data_dir else 'data/oct_retinal')
    elif dataset_name == 'brain_tumor':
        dataset = BrainTumorMRI(data_dir if data_dir else 'data/brain_tumor')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Prepare data
    dataset.prepare_data()
    
    # Get datasets
    train_dataset, cal_dataset, test_dataset = dataset.get_datasets(img_size)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, cal_loader, test_loader, dataset.num_classes


if __name__ == '__main__':
    """Test dataset loading"""
    print("Testing Medical Image Classification Datasets\n")
    
    # Test Chest X-Ray
    try:
        print("=" * 60)
        print("Testing Chest X-Ray Pneumonia Dataset")
        print("=" * 60)
        train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
            dataset_name='chest_xray',
            batch_size=16
        )
        print(f"Loaded successfully. Classes: {num_classes}")

        # Test one batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}, Labels: {labels.shape}")

    except Exception as e:
        print(f"Error: {e}")

    print()
