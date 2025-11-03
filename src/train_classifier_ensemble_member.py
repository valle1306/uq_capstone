"""
Train Ensemble Member for Medical Image Classification

Train multiple independent models for Deep Ensemble uncertainty quantification.
Each model is trained with different random initialization.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import numpy as np
from pathlib import Path

from data_utils_classification import get_classification_loaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, best_acc, output_dir, filename='checkpoint.pth'):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    save_path = Path(output_dir) / filename
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train Ensemble Member')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='chest_xray',
                        choices=['chest_xray', 'oct_retinal', 'brain_tumor'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset directory')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    
    # Model arguments
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pretrained weights')
    
    # Ensemble arguments
    parser.add_argument('--member_id', type=int, required=True,
                        help='Ensemble member ID (for different seeds)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='runs/classification/ensemble',
                        help='Output directory')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    
    args = parser.parse_args()
    
    # Set random seed based on member ID
    seed = args.seed + args.member_id * 1000
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory for this member
    member_dir = Path(args.output_dir) / f'member_{args.member_id}'
    member_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args).copy()
    config['actual_seed'] = seed
    with open(member_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=" * 70)
    print(f"Training Ensemble Member {args.member_id}")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Architecture: {args.arch}")
    print(f"Seed: {seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {member_dir}")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    print("Data loaded")
    
    # Create model
    print("\nCreating model...")
    if args.arch == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = models.resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained)
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(args.device)
    
    print("Model created")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, args.device)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print summary
        print(f"  Train: {train_loss:.4f} / {train_acc:.2f}% | Val: {val_loss:.4f} / {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_acc, member_dir, 'best_model.pth')
            print(f"  New best: {best_acc:.2f}%")
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, best_acc, member_dir, f'checkpoint_epoch_{epoch}.pth')
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, best_acc, member_dir, 'final_model.pth')
    
    # Save training history
    with open(member_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Member {args.member_id} Training Complete!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
