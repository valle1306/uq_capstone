"""
Train U-Net with SWAG (Stochastic Weight Averaging-Gaussian)

SWAG training procedure:
1. Train normally for first N epochs (warmup)
2. Reduce learning rate (annealing phase)
3. Collect model snapshots every K epochs during annealing
4. SWAG posterior is built from these collected snapshots

Based on: Maddox et al. "A Simple Baseline for Bayesian Uncertainty Estimation in Deep Learning"
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import BraTSSegmentationDataset
from src.model_utils import UNet, DiceBCELoss, DiceLoss
from src.swag import SWAG, SWAGScheduler


def train_epoch(model, loader, criterion, optimizer, device, desc="Train"):
    """Train for one epoch"""
    model.train()
    losses = []
    
    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return np.mean(losses)


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            losses.append(loss.item())
    
    return np.mean(losses)


def train_swag(
    data_dir: str,
    output_dir: str,
    epochs: int = 30,
    swag_start_epoch: int = 15,  # Start collecting after this epoch
    swag_lr: float = 1e-4,  # Lower LR for SWAG collection phase
    batch_size: int = 8,
    lr: float = 1e-3,
    in_channels: int = 1,
    max_num_models: int = 20,  # K parameter in paper
    collect_freq: int = 1,  # Collect every N epochs
    device: str = 'cuda'
):
    """
    Train U-Net with SWAG
    
    Training schedule:
    - Epochs 0-14: Normal training with lr=1e-3
    - Epoch 15: Switch to SWAG mode, reduce lr to 1e-4
    - Epochs 15-29: Collect model snapshots every epoch
    """
    
    print("=" * 60)
    print("SWAG Training Configuration")
    print("=" * 60)
    print(f"Total epochs: {epochs}")
    print(f"SWAG start epoch: {swag_start_epoch}")
    print(f"Initial LR: {lr}")
    print(f"SWAG LR: {swag_lr}")
    print(f"Max models (K): {max_num_models}")
    print(f"Collection frequency: every {collect_freq} epoch(s)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("\n📁 Loading datasets...")
    train_dataset = BraTSSegmentationDataset(
        npz_dir=os.path.join(data_dir, 'train'),
        augment=True
    )
    val_dataset = BraTSSegmentationDataset(
        npz_dir=os.path.join(data_dir, 'val'),
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"✓ Train: {len(train_dataset)} slices")
    print(f"✓ Val: {len(val_dataset)} slices")
    
    # Create base model
    print(f"\n🏗️ Creating U-Net (in_channels={in_channels})...")
    base_model = UNet(in_channels=in_channels, num_classes=1, dropout_rate=0.0)
    
    # Create SWAG wrapper
    swag_model = SWAG(base_model, max_num_models=max_num_models)
    swag_model.to(device)
    
    # Create SWAG scheduler
    swag_scheduler = SWAGScheduler(
        swag_model=swag_model,
        collect_start_epoch=swag_start_epoch,
        collect_freq=collect_freq
    )
    
    # Loss and optimizer
    criterion = DiceBCELoss()
    optimizer = optim.Adam(base_model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'swag_epochs': []
    }
    
    best_val_loss = float('inf')
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Switch to SWAG collection phase
        if epoch == swag_start_epoch:
            print(f"\n{'='*60}")
            print(f"🔄 Switching to SWAG collection phase at epoch {epoch}")
            print(f"   Reducing learning rate: {lr} → {swag_lr}")
            print(f"   Will collect models every {collect_freq} epoch(s)")
            print(f"{'='*60}\n")
            
            # Reduce learning rate for annealing phase
            for param_group in optimizer.param_groups:
                param_group['lr'] = swag_lr
        
        # Train epoch
        train_loss = train_epoch(
            base_model, 
            train_loader, 
            criterion, 
            optimizer, 
            device,
            desc=f"Epoch {epoch+1}/{epochs}"
        )
        
        # Validate
        val_loss = validate(base_model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Collect model for SWAG (if in collection phase)
        swag_scheduler.step(epoch, base_model)
        if epoch >= swag_start_epoch:
            history['swag_epochs'].append(epoch)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model (base model, not SWAG)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_base_model.pth'))
            print(f"   ✓ Saved best base model (val_loss: {val_loss:.4f})")
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete! Collected {swag_model.n_models} SWAG snapshots")
    print("=" * 60)
    
    # Save SWAG model with all statistics
    print("\n💾 Saving SWAG model...")
    swag_checkpoint = {
        'n_models': swag_model.n_models,
        'mean': swag_model.mean,
        'sq_mean': swag_model.sq_mean,
        'cov_mat_sqrt': swag_model.cov_mat_sqrt,
        'max_num_models': swag_model.max_num_models,
        'config': {
            'in_channels': in_channels,
            'num_classes': 1,
            'swag_start_epoch': swag_start_epoch,
            'swag_lr': swag_lr,
            'max_num_models': max_num_models
        }
    }
    torch.save(swag_checkpoint, os.path.join(output_dir, 'swag_model.pth'))
    print(f"✓ Saved SWAG checkpoint to {output_dir}/swag_model.pth")
    
    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = {
        'epochs': epochs,
        'swag_start_epoch': swag_start_epoch,
        'swag_lr': swag_lr,
        'batch_size': batch_size,
        'lr': lr,
        'in_channels': in_channels,
        'max_num_models': max_num_models,
        'collect_freq': collect_freq,
        'final_train_loss': float(train_loss),
        'final_val_loss': float(val_loss),
        'best_val_loss': float(best_val_loss),
        'swag_models_collected': swag_model.n_models
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📊 Final Results:")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   SWAG models collected: {swag_model.n_models}")
    print(f"\n✨ All outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net with SWAG")
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                       default='/scratch/hpl14/uq_capstone/data/brats_subset_npz',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, 
                       default='runs/swag',
                       help='Output directory')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Total training epochs')
    parser.add_argument('--swag_start', type=int, default=15,
                       help='Epoch to start SWAG collection')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--swag_lr', type=float, default=1e-4,
                       help='Learning rate for SWAG collection phase')
    
    # SWAG params
    parser.add_argument('--max_models', type=int, default=20,
                       help='Max number of models to store (K parameter)')
    parser.add_argument('--collect_freq', type=int, default=1,
                       help='Collect model every N epochs')
    
    # Model
    parser.add_argument('--in_ch', type=int, default=1,
                       help='Input channels')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("SWAG (Stochastic Weight Averaging-Gaussian) Training")
    print("Maddox et al. 2019")
    print("=" * 60)
    
    train_swag(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        swag_start_epoch=args.swag_start,
        swag_lr=args.swag_lr,
        batch_size=args.batch,
        lr=args.lr,
        in_channels=args.in_ch,
        max_num_models=args.max_models,
        collect_freq=args.collect_freq,
        device=args.device
    )
