"""
Deep Ensemble Training with Adam Optimizer for Chest X-Ray Classification - 300 Epochs
Train 5 ensemble members independently with Adam optimizer
Extends to match SWAG paper training duration

Key details:
- Adam optimizer with lr=0.0001
- 300 epochs (matching original SWAG paper duration)
- 5 ensemble members with different random seeds
- Cosine annealing learning rate schedule
- Same training parameters as other Adam 300-epoch experiments

References:
- Deep Ensembles: Lakshminarayanan et al. (2017)
- Medical imaging application: Mehta et al. (2021)
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


def train_single_member(member_id, args, device, train_loader, test_loader):
    """Train a single ensemble member"""
    print(f"\n{'='*70}")
    print(f"Training Ensemble Member {member_id+1}/5 (300 Epochs)")
    print(f"{'='*70}")
    
    # Set seed for this member
    member_seed = args.seed + member_id * 1000
    torch.manual_seed(member_seed)
    np.random.seed(member_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(member_seed)
    
    # Create model
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained)
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=args.pretrained)
    else:
        model = models.resnet50(pretrained=args.pretrained)
    
    # Modify final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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
    
    # Training history for this member
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }
    
    best_acc = 0.0
    member_dir = Path(args.save_dir) / f'member_{member_id}'
    member_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nMember {member_id+1} | Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
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
        
        print(f"Train: {train_loss:.4f} / {train_acc:.2f}% | Test: {test_loss:.4f} / {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, member_dir / 'best_model.pt')
            print(f"✓ New best for member {member_id+1}! (Acc: {test_acc:.2f}%)")
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
    }, member_dir / 'final_model.pt')
    
    # Save history
    with open(member_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description='Deep Ensemble Training with Adam - 300 Epochs')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/chest_xray')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    
    # Training arguments (300 epochs)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_final', type=float, default=0.00005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.999])
    
    # Ensemble arguments
    parser.add_argument('--num_members', type=int, default=5,
                        help='Number of ensemble members')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='results/ensemble_adam_300')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*70)
    print("Deep Ensemble Training with Adam - 300 Epochs")
    print("="*70)
    print(f"Number of members: {args.num_members}")
    print(f"Epochs per member: {args.epochs}")
    print("="*70)
    
    # Load data
    print("\nLoading datasets...")
    train_loader, cal_loader, test_loader = get_classification_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Calibration samples: {len(cal_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Train ensemble members
    models = []
    member_accs = []
    
    for member_id in range(args.num_members):
        model, best_acc = train_single_member(
            member_id, args, device, train_loader, test_loader
        )
        models.append(model)
        member_accs.append(best_acc)
    
    print("\n" + "="*70)
    print("All ensemble members trained! Now evaluating ensemble...")
    print("="*70)
    
    # Ensemble evaluation
    print("\nEvaluating ensemble predictions...")
    all_preds = []
    all_labels = []
    
    for member_id, model in enumerate(models):
        model.eval()
        member_preds = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'Member {member_id+1}'):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                member_preds.append(probs.cpu())
                
                if len(all_labels) < len(test_loader.dataset):
                    all_labels.extend(labels.numpy())
        
        all_preds.append(torch.cat(member_preds, dim=0))
    
    # Ensemble predictions
    all_preds = torch.stack(all_preds)
    mean_probs = all_preds.mean(dim=0)
    pred_classes = mean_probs.argmax(dim=1).numpy()
    all_labels = np.array(all_labels)
    
    ensemble_acc = 100.0 * (pred_classes == all_labels).mean()
    
    # Compute uncertainty
    predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1).mean().item()
    
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    for i, acc in enumerate(member_accs):
        print(f"Member {i+1} best accuracy: {acc:.2f}%")
    print(f"\nMean member accuracy: {np.mean(member_accs):.2f}%")
    print(f"Ensemble accuracy: {ensemble_acc:.2f}%")
    print(f"Predictive entropy: {predictive_entropy:.4f}")
    print("="*70)
    
    # Save results
    results = {
        'member_accuracies': member_accs,
        'mean_member_acc': np.mean(member_accs),
        'ensemble_acc': ensemble_acc,
        'predictive_entropy': predictive_entropy,
        'num_members': args.num_members
    }
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()
