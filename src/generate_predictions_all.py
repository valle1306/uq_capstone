"""
Generate predictions.npz for all UQ methods on Amarel

Expects trained models in runs/classification/<method>/
Saves predictions to runs/classification/<method>/predictions.npz

Usage (on Amarel):
    cd /scratch/hpl14/uq_capstone
    python src/generate_predictions_all.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_classification_loaders(dataset_name='chest_xray', batch_size=32, num_workers=4):
    """Load train/cal/test dataloaders"""
    try:
        from data_utils_classification import get_classification_loaders as get_loaders
        return get_loaders(dataset_name, batch_size, num_workers)
    except ImportError as e:
        print(f"Error importing data_utils_classification: {e}")
        print("Make sure you're running from /scratch/hpl14/uq_capstone/")
        raise

def load_model(model_path, num_classes=2, device='cuda'):
    """Load trained ResNet-18 model"""
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def get_predictions(model, loader, device, num_classes=2):
    """Get softmax probabilities for all samples in loader"""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting'):
            # Handle different data formats
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, labels = batch
            elif isinstance(batch, dict):
                images = batch.get('image', batch.get('x'))
                labels = batch.get('label', batch.get('y'))
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_probs, all_labels

def main():
    methods = ['baseline', 'swag_sgd', 'swag_adam', 'ensemble', 'dropout']
    base_dir = Path('runs/classification')
    
    # Load data
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS FOR ALL UQ METHODS")
    print("="*70)
    
    print("\nLoading data loaders...")
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name='chest_xray',
        batch_size=64,
        num_workers=4
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Calibration samples: {len(cal_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    n_cal = len(cal_loader.dataset)
    n_test = len(test_loader.dataset)
    
    # Process each method
    successful = 0
    failed = 0
    
    for method in methods:
        model_dir = base_dir / method
        model_path = model_dir / 'best_model.pth'
        
        if not model_path.exists():
            print(f"\n⚠️  Model not found: {model_path}")
            failed += 1
            continue
        
        print(f"\n{'='*70}")
        print(f"Generating predictions for: {method.upper()}")
        print(f"{'='*70}")
        
        # Load model
        try:
            print(f"  Loading model from {model_path}...")
            model = load_model(str(model_path), num_classes, device)
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to load {method}: {e}")
            failed += 1
            continue
        
        # Get predictions on calibration set
        try:
            print(f"  Computing predictions on calibration set ({n_cal} samples)...")
            cal_probs, cal_labels = get_predictions(model, cal_loader, device, num_classes)
            print(f"    Shape: {cal_probs.shape}")
            
            # Get predictions on test set
            print(f"  Computing predictions on test set ({n_test} samples)...")
            test_probs, test_labels = get_predictions(model, test_loader, device, num_classes)
            print(f"    Shape: {test_probs.shape}")
            
            # Combine for conformal
            all_probs = np.concatenate([cal_probs, test_probs], axis=0)
            all_labels = np.concatenate([cal_labels, test_labels], axis=0)
            
            # Define split indices
            cal_idx = np.arange(0, n_cal)
            test_idx = np.arange(n_cal, n_cal + n_test)
            
            # Verify integrity
            assert all_probs.shape[0] == len(all_labels), "Shape mismatch!"
            assert all_probs.shape[1] == num_classes, f"Expected {num_classes} classes, got {all_probs.shape[1]}"
            
            # Save NPZ
            output_path = model_dir / 'predictions.npz'
            np.savez(
                output_path,
                probs=all_probs.astype(np.float32),
                y=all_labels.astype(np.int32),
                cal_idx=cal_idx,
                test_idx=test_idx
            )
            
            print(f"  ✓ Saved to {output_path}")
            print(f"    Total samples: {all_probs.shape[0]}")
            print(f"    Classes: {num_classes}")
            print(f"    Cal/Test split: {n_cal}/{n_test}")
            print(f"    File size: {output_path.stat().st_size / 1e6:.1f} MB")
            
            successful += 1
            
        except Exception as e:
            print(f"  ❌ Failed to generate predictions for {method}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully generated: {successful} methods")
    print(f"Failed: {failed} methods")
    
    if failed == 0:
        print(f"\n✓ All predictions generated successfully!")
        print(f"\nNext step:")
        print(f"  1. Download all predictions.npz files locally")
        print(f"  2. Run: python src/run_conformal_all.py --methods {' '.join(methods)} --cal_size {n_cal}")
    else:
        print(f"\n⚠️  Some methods failed. Check errors above.")

if __name__ == '__main__':
    main()
