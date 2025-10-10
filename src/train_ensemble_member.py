"""
Train individual Deep Ensemble member - matches working baseline training
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


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    losses = []
    
    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            losses.append(loss.item())
    
    return np.mean(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--member_id', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print(f"ENSEMBLE MEMBER {args.member_id}")
    print(f"Seed: {args.seed}, Device: {device}")
    print("=" * 60)
    
    # Load data
    train_dataset = BraTSSegmentationDataset(
        npz_dir=os.path.join(args.data_dir, 'train'),
        augment=True
    )
    val_dataset = BraTSSegmentationDataset(
        npz_dir=os.path.join(args.data_dir, 'val'),
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = UNet(in_channels=args.in_ch, num_classes=1, dropout_rate=args.dropout).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
    
    # Save
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump({
            'member_id': args.member_id,
            'seed': args.seed,
            'best_val_loss': best_val_loss
        }, f)
    
    print(f"✅ Complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
