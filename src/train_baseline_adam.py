"""
Adam Baseline Training for Chest X-Ray Classification
Following medical imaging literature (Mehta et al. 2021, Adams & Elhabian 2023)

Key details:
- Adam optimizer with lr=0.0001 (medical imaging standard)
- Cosine annealing learning rate schedule
- Same weight decay, batch size, epochs as SGD experiments
- Provides comparison for SWAG-Adam implementation

References:
- Mehta et al. (2021): Propagating uncertainty across cascaded medical imaging tasks
- Adams & Elhabian (2023): Benchmarking scalable epistemic UQ in organ segmentation
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import models

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils_classification import get_classification_loaders


def train_epoch(model, loader, criterion, optimizer, device):
    """Standard training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc='Train', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100. * correct / total


def eval_model(model, loader, criterion, device):
    """Evaluate model on dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Adam Baseline Training (Medical Imaging)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                        default='data/chest_xray',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained ImageNet weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate (medical imaging standard)')
    parser.add_argument('--lr_final', type=float, default=0.00005,
                        help='Final learning rate for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (matching SGD experiments)')
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='Adam beta parameters')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, 
                        default='results/baseline_adam',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading datasets...")
    train_loader, cal_loader, test_loader = get_classification_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Calibration samples: {len(cal_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model} model...")
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained)
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=args.pretrained)
    else:  # resnet50
        model = models.resnet50(pretrained=args.pretrained)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer (following medical imaging literature)
    print("Using Adam optimizer (medical imaging standard)")
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.adam_betas),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_final
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_loss, test_acc = eval_model(
            model, test_loader, criterion, device
        )
        
        # Step scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, save_dir / 'best_model.pt')
            print(f"✓ New best model saved! (Acc: {test_acc:.2f}%)")
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print("="*50)
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
    }, save_dir / 'final_model.pt')
    
    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()
