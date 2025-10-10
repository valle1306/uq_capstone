"""
Train baseline U-Net for brain tumor segmentation (no UQ)
Matches the working test_minimal training setup
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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import BraTSSegmentationDataset
from src.model_utils import UNet, DiceBCELoss


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


def train_baseline(
    data_dir: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    in_channels: int = 1,
    device: str = 'cuda'
):
    """Train baseline U-Net model"""
    
    print("=" * 60)
    print("BASELINE U-NET TRAINING")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Input channels: {in_channels}")
    print(f"Output: {output_dir}")
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
    
    # Create model
    print(f"\n🏗️ Creating U-Net (in_channels={in_channels})...")
    model = UNet(in_channels=in_channels, num_classes=1, dropout_rate=0.0)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device,
            desc=f"Epoch {epoch+1}/{epochs}"
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"   ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("=" * 60)
    
    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'in_channels': in_channels,
        'final_train_loss': float(train_loss),
        'final_val_loss': float(val_loss),
        'best_val_loss': float(best_val_loss)
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✨ All outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline U-Net")
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                       default='/scratch/hpl14/uq_capstone/data/brats_subset_npz',
                       help='Path to NPZ data directory')
    parser.add_argument('--output_dir', type=str, 
                       default='runs/baseline',
                       help='Output directory')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    # Model
    parser.add_argument('--in_ch', type=int, default=1,
                       help='Input channels')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    train_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        in_channels=args.in_ch,
        device=args.device
    )
