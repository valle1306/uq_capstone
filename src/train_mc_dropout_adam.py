"""
MC Dropout Training with Adam Optimizer for Chest X-Ray Classification
Following medical imaging literature

Key details:
- Adam optimizer with lr=0.0001
- Dropout rate=0.2 (lighter than 0.3 for medical imaging)
- Cosine annealing learning rate schedule
- Same training parameters as other Adam experiments

References:
- MC Dropout: Gal & Ghahramani (2016)
- Medical imaging adaptation: Mehta et al. (2021)
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


class ResNetWithDropout(nn.Module):
    """ResNet with dropout for MC Dropout"""
    
    def __init__(self, base_model, num_classes, dropout_rate=0.2):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        
        # Replace final layer with dropout
        num_features = base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)
    
    def enable_dropout(self):
        """Enable dropout during inference for MC sampling"""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Eval", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='MC Dropout Training with Adam')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/chest_xray')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate for MC Dropout')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--lr_final', type=float, default=0.00005,
                        help='Final learning rate for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.999])
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='results/mc_dropout_adam')
    parser.add_argument('--seed', type=int, default=42)
    
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
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model} with dropout={args.dropout_rate}...")
    if args.model == 'resnet18':
        base = models.resnet18(pretrained=args.pretrained)
    elif args.model == 'resnet34':
        base = models.resnet34(pretrained=args.pretrained)
    else:
        base = models.resnet50(pretrained=args.pretrained)
    
    model = ResNetWithDropout(base, num_classes=2, dropout_rate=args.dropout_rate)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    print("Using Adam optimizer")
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.adam_betas),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing scheduler
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
        
        # Validate
        test_loss, test_acc = validate(
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
