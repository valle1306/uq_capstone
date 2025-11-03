#!/usr/bin/env python3
"""
Retrain SWAG - Initialize from baseline for better posterior approximation

Key insight: SWAG should start from a good model (baseline) then collect
snapshots during fine-tuning. This gives better uncertainty estimates.
"""

import os
import json
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import models

from data_utils_classification import get_classification_loaders
from swag import SWAG, SWAGScheduler


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for inputs, labels in pbar:
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

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validate', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Retrain SWAG from baseline')
    parser.add_argument('--baseline_path', type=str, required=True, help='Baseline model checkpoint')
    parser.add_argument('--dataset', type=str, default='chest_xray')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--swag_start', type=int, default=30, help='Start collecting SWAG snapshots at this epoch')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='runs/classification/swag_classification_retrain')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print(f"SWAG snapshots starting at epoch {args.swag_start}")
    
    # Load data
    print("\nLoading data...")
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Load baseline checkpoint
    print(f"\nLoading baseline from {args.baseline_path}")
    base_model = models.resnet18(pretrained=False)
    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, num_classes)
    
    baseline_checkpoint = torch.load(args.baseline_path, map_location='cpu')
    base_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    
    # Create SWAG wrapper
    swag = SWAG(base_model, max_num_models=20)
    swag = swag.to(device)
    
    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(swag.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print("\nStarting training...")
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'swag_snapshots': 0
    }
    
    best_val_acc = 0
    best_base_model_path = output_dir / 'best_base_model.pth'
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        train_loss, train_acc = train_epoch(swag, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(swag, cal_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Save best base model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✓ New best model! Saving to {best_base_model_path}")
            torch.save({
                'model_state_dict': swag.base_model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, best_base_model_path)
        
        # Collect SWAG snapshots starting at swag_start epoch
        if epoch >= args.swag_start:
            swag.collect_model(swag.base_model)
            history['swag_snapshots'] += 1
            print(f"✓ Collected SWAG snapshot {history['swag_snapshots']} (total: {swag.n_models})")
        
        scheduler.step()
    
    # Save SWAG model
    swag_path = output_dir / 'swag_model.pth'
    torch.save({
        'n_models': swag.n_models,
        'mean': swag.mean,
        'sq_mean': swag.sq_mean,
        'cov_mat_sqrt': swag.cov_mat_sqrt,
        'max_num_models': swag.max_num_models
    }, swag_path)
    print(f"\n✓ SWAG model saved to {swag_path}")
    
    # Save history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'epochs': args.epochs,
            'swag_start': args.swag_start,
            'swag_snapshots': history['swag_snapshots'],
            'batch_size': args.batch_size,
            'lr': args.lr,
            'best_val_acc': best_val_acc,
            'initialized_from': args.baseline_path
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"SWAG snapshots collected: {history['swag_snapshots']}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
