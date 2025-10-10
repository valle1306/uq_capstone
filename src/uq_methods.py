"""
Uncertainty Quantification Methods for Segmentation
Implements: Temperature Scaling, MC Dropout, Deep Ensembles, Conformal Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibrating neural network predictions.
    Learns a single temperature parameter to scale logits.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def fit(self, model, val_loader, device, max_iters=50):
        """
        Calibrate temperature on validation set using NLL loss.
        
        Args:
            model: Trained segmentation model
            val_loader: Validation data loader
            device: Device to run on
            max_iters: Maximum optimization iterations
        """
        nll_criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iters)
        
        logits_list = []
        labels_list = []
        
        model.eval()
        print("Collecting validation predictions for temperature scaling...")
        with torch.no_grad():
            for x, y in tqdm(val_loader, leave=False):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                logits_list.append(logits)
                labels_list.append(y)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        print(f"Fitting temperature scaling on {len(logits)} samples...")
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()


class MCDropoutSegmentation:
    """
    Monte Carlo Dropout for uncertainty estimation.
    Enables dropout at test time and samples multiple predictions.
    """
    def __init__(self, model, dropout_rate=0.2, num_samples=20):
        """
        Args:
            model: Segmentation model with dropout layers
            dropout_rate: Dropout probability
            num_samples: Number of MC samples to draw
        """
        self.model = model
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
    
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(self, x, device):
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            x: Input tensor [B, C, H, W]
            device: Device to run on
            
        Returns:
            mean_pred: Mean prediction [B, 1, H, W]
            std_pred: Predictive standard deviation [B, 1, H, W]
            samples: All MC samples [num_samples, B, 1, H, W]
        """
        self.model.eval()
        self.enable_dropout()
        
        samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(x.to(device))
                probs = torch.sigmoid(logits)
                samples.append(probs.cpu())
        
        samples = torch.stack(samples)  # [num_samples, B, 1, H, W]
        mean_pred = samples.mean(dim=0)
        std_pred = samples.std(dim=0)
        
        return mean_pred, std_pred, samples


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty estimation.
    Trains multiple models with different initializations.
    """
    def __init__(self, models: List[nn.Module]):
        """
        Args:
            models: List of trained models
        """
        self.models = models
        self.num_models = len(models)
    
    def predict_with_uncertainty(self, x, device):
        """
        Generate ensemble predictions with uncertainty.
        
        Args:
            x: Input tensor [B, C, H, W]
            device: Device to run on
            
        Returns:
            mean_pred: Mean prediction [B, 1, H, W]
            std_pred: Predictive standard deviation [B, 1, H, W]
            all_preds: All model predictions [num_models, B, 1, H, W]
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x.to(device))
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu())
        
        all_preds = torch.stack(predictions)  # [num_models, B, 1, H, W]
        mean_pred = all_preds.mean(dim=0)
        std_pred = all_preds.std(dim=0)
        
        return mean_pred, std_pred, all_preds


class ConformalPrediction:
    """
    Conformal Prediction for calibrated prediction sets.
    Provides finite-sample coverage guarantees.
    """
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: Miscoverage level (1-alpha is the target coverage)
        """
        self.alpha = alpha
        self.threshold = None
    
    def calibrate(self, model, cal_loader, device):
        """
        Calibrate conformity scores on calibration set.
        
        Args:
            model: Trained segmentation model
            cal_loader: Calibration data loader
            device: Device to run on
        """
        conformity_scores = []
        
        model.eval()
        print(f"Calibrating conformal prediction (target coverage: {1-self.alpha:.1%})...")
        with torch.no_grad():
            for x, y in tqdm(cal_loader, leave=False):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)
                
                # Conformity score: 1 - probability of true class
                # For pixels where y=1, score = 1 - prob; for y=0, score = prob
                scores = torch.where(y > 0.5, 1 - probs, probs)
                conformity_scores.append(scores.cpu().flatten())
        
        all_scores = torch.cat(conformity_scores)
        
        # Compute quantile threshold
        n = len(all_scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = torch.quantile(all_scores, q).item()
        
        print(f"Conformal threshold: {self.threshold:.4f}")
        return self.threshold
    
    def predict_sets(self, model, x, device):
        """
        Generate prediction sets with coverage guarantee.
        
        Args:
            model: Trained model
            x: Input tensor [B, C, H, W]
            device: Device to run on
            
        Returns:
            pred_sets: Binary prediction sets [B, 1, H, W]
            probs: Model probabilities [B, 1, H, W]
        """
        model.eval()
        with torch.no_grad():
            logits = model(x.to(device))
            probs = torch.sigmoid(logits)
            
            # Prediction set: include class if probability > (1 - threshold)
            pred_sets = (probs > (1 - self.threshold)).float()
            
        return pred_sets.cpu(), probs.cpu()


def compute_uncertainty_metrics(mean_pred, std_pred, y_true):
    """
    Compute uncertainty quantification metrics.
    
    Args:
        mean_pred: Mean predictions [B, 1, H, W]
        std_pred: Predictive uncertainty [B, 1, H, W]
        y_true: Ground truth [B, 1, H, W]
        
    Returns:
        dict: Uncertainty metrics
    """
    # Convert to binary predictions
    y_pred = (mean_pred > 0.5).float()
    
    # Compute errors
    errors = (y_pred != y_true).float()
    
    # Metrics
    metrics = {
        'mean_uncertainty': std_pred.mean().item(),
        'uncertainty_on_errors': std_pred[errors > 0].mean().item() if errors.sum() > 0 else 0,
        'uncertainty_on_correct': std_pred[errors == 0].mean().item() if (errors == 0).sum() > 0 else 0,
        'dice_score': compute_dice(y_pred, y_true),
        'error_rate': errors.mean().item(),
    }
    
    return metrics


def compute_dice(pred, target, smooth=1e-6):
    """Compute Dice coefficient."""
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice.item()


def compute_calibration_metrics(probs, targets, num_bins=10):
    """
    Compute calibration metrics: ECE (Expected Calibration Error) and reliability diagram data.
    
    Args:
        probs: Predicted probabilities [N]
        targets: Ground truth binary labels [N]
        num_bins: Number of bins for calibration
        
    Returns:
        dict: Calibration metrics including ECE and bin statistics
    """
    probs = probs.flatten().numpy()
    targets = targets.flatten().numpy()
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
    
    return {
        'ece': ece,
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs,
        'bin_counts': bin_counts
    }
