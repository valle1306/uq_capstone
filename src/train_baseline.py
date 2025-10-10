"""
Train baseline segmentation model (no UQ)
This serves as the baseline for comparison with UQ methods.
"""
import os
import argparse
import torch
import json
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

from src.segmentation_utils import UNetSmall, train_seg_epoch, eval_seg
from src.train_seg import BratsSliceDataset


def parse_args():
    p = argparse.ArgumentParser(description='Train baseline segmentation model')
    p.add_argument('--data_root', required=True, help='Path to data directory')
    p.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    p.add_argument('--batch', type=int, default=8, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--device', default='cuda', help='Device to use')
    p.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    p.add_argument('--in_ch', type=int, default=1, help='Number of input channels')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    p.add_argument('--save_dir', default='runs/baseline', help='Directory to save results')
    return p.parse_args()


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    
    print("="*60)
    print("BASELINE SEGMENTATION TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print(f"Save dir: {args.save_dir}")
    print("="*60)
    
    # Load data
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    multimodal = True
    
    train_ds = BratsSliceDataset(
        os.path.join(args.data_root, 'train.csv'), 
        args.data_root, 
        transform=transform, 
        multimodal=multimodal
    )
    val_ds = BratsSliceDataset(
        os.path.join(args.data_root, 'val.csv'), 
        args.data_root, 
        transform=transform, 
        multimodal=multimodal
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print()
    
    # Create model
    model = UNetSmall(in_ch=args.in_ch, out_ch=1, dropout_p=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_dice = 0.0
    history = {'train_loss': [], 'val_dice': []}
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_seg_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_dice = eval_seg(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)
        
        print(f'Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f} val_dice={val_dice:.4f}')
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch{epoch+1}.pth'))
    
    print()
    print("="*60)
    print(f"Training complete! Best val dice: {best_dice:.4f}")
    print("="*60)
    
    # Save training history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = vars(args)
    config['best_val_dice'] = best_dice
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
