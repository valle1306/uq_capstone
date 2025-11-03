#!/usr/bin/env python3
"""
Retrain MC Dropout Classifier - CORRECTED VERSION
Start from baseline weights + fine-tune with proper dropout

This ensures MC Dropout has good performance (~90%) while maintaining stochasticity
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
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


def main():
    parser = argparse.ArgumentParser(description='Retrain MC Dropout Classifier (from baseline)')
    parser.add_argument('--baseline_path', type=str, required=True, help='Path to baseline checkpoint')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate (0.2 recommended)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='runs/classification/mc_dropout_retrain')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print(f"Dropout rate: {args.dropout_rate}")
    
    # Load data
    print("\nLoading data...")
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name='chest_xray',
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Load baseline checkpoint
    print(f"\nLoading baseline from {args.baseline_path}")
    base_model = models.resnet18(pretrained=False)
    baseline_checkpoint = torch.load(args.baseline_path, map_location='cpu')
    base_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    
    # Create MC Dropout model with baseline weights
    model = ResNetWithDropout(base_model, num_classes, dropout_rate=args.dropout_rate)
    model = model.to(device)
    
    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\nStarting training...")
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    best_model_path = output_dir / 'best_model.pth'
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        val_loss, val_acc = validate(model, cal_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✓ New best model! Saving to {best_model_path}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'dropout_rate': args.dropout_rate
            }, best_model_path)
        
        scheduler.step()
    
    # Save training history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'dropout_rate': args.dropout_rate,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'best_val_acc': best_val_acc,
            'initialized_from': args.baseline_path
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'='*70}")
    
    # Test performance
    print("\nTesting on test set...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
