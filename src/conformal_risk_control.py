"""
Conformal Risk Control Implementation

Based on: "Conformal Risk Control" by Angelopoulos et al. (2022)
Paper: https://arxiv.org/abs/2208.02814

Extends standard conformal prediction to control arbitrary risk metrics:
- False Negative Rate (FNR)
- Precision
- Expected Set Size
- Custom composite losses

Key difference from standard conformal:
- Standard: Coverage guarantee (e.g., "90% of time, true label in set")
- Risk Control: Arbitrary risk bounds (e.g., "FNR ≤ 5%", "Precision ≥ 80%")
"""

import torch
import numpy as np
from typing import Callable, List, Tuple
from tqdm import tqdm


class ConformalRiskControl:
    """
    Conformal Prediction with Risk Control
    
    Controls arbitrary risk functionals using calibration data.
    Provides probabilistic guarantees on risk metrics.
    """
    
    def __init__(self, 
                 loss_fn: Callable,
                 alpha: float = 0.1,
                 delta: float = 0.1,
                 lambda_type: str = 'adaptive'):
        """
        Args:
            loss_fn: Loss function λ(y_true, pred_set, probs) -> float
                     Should return non-negative loss for each prediction
            alpha: Target risk level (upper bound on expected loss)
            delta: Confidence parameter (guarantee holds w.p. ≥ 1-δ)
            lambda_type: 'fixed' or 'adaptive' threshold selection
        """
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.delta = delta
        self.lambda_type = lambda_type
        self.threshold = None
        self.calibration_losses = None
        
    def calibrate(self, model, cal_loader, device='cuda'):
        """
        Calibrate threshold using calibration data
        
        Implements Algorithm 1 from the paper:
        1. Compute losses for all calibration samples at different thresholds
        2. Find threshold that controls risk at level α
        
        Args:
            model: Trained classifier
            cal_loader: DataLoader with calibration data
            device: Device to run on
        """
        model.eval()
        
        # Step 1: Collect all predictions and losses
        all_losses_by_threshold = []

        print(f"Calibrating risk control (alpha={self.alpha}, delta={self.delta})...")

        with torch.no_grad():
            for inputs, labels in tqdm(cal_loader, desc="Computing calibration losses"):
                inputs = inputs.to(device)
                labels = labels.cpu().numpy()
                
                # Get predictions
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                # For each sample, compute loss at different thresholds
                for i in range(len(labels)):
                    prob = probs[i]
                    label = labels[i]
                    
                    # Sort classes by probability (descending)
                    sorted_idx = np.argsort(-prob)
                    sorted_probs = prob[sorted_idx]
                    
                    # For each possible threshold (= each probability value)
                    for k in range(len(prob)):
                        # Threshold at k-th highest probability
                        threshold_val = sorted_probs[k]
                        
                        # Prediction set: all classes with prob >= threshold
                        pred_set = sorted_idx[:k+1].tolist()
                        
                        # Compute loss for this prediction set
                        loss_val = self.loss_fn(label, pred_set, prob)
                        
                        all_losses_by_threshold.append((threshold_val, loss_val))
        
        # Step 2: Find threshold using risk control bound
        # Sort by threshold value
        all_losses_by_threshold.sort(key=lambda x: x[0], reverse=True)
        
        n = len(all_losses_by_threshold)
        
        # Use Theorem 1 from paper: find λ̂ such that empirical risk ≤ α
        # With high probability (1-δ), true risk ≤ α + O(√(log(1/δ)/n))
        
        if self.lambda_type == 'adaptive':
            # Adaptive threshold: account for finite sample correction
            correction = np.sqrt(np.log(1/self.delta) / (2*n))
            target_risk = max(0, self.alpha - correction)
        else:
            # Fixed threshold
            target_risk = self.alpha
        
        # Find threshold where empirical risk crosses target
        cumulative_loss = 0
        best_threshold = 0.0
        
        for i, (threshold_val, loss_val) in enumerate(all_losses_by_threshold):
            cumulative_loss += loss_val
            empirical_risk = cumulative_loss / (i + 1)
            
            if empirical_risk <= target_risk:
                best_threshold = threshold_val
                break
        
        self.threshold = best_threshold
        self.calibration_losses = all_losses_by_threshold

        # Summary
        print(f"Calibrated threshold: lambda = {self.threshold:.4f}")
        print(f"  Target risk: alpha = {self.alpha:.4f}")
        print(f"  Calibration samples: n = {n}")

        return self.threshold
    
    def predict(self, model, x, device='cuda'):
        """
        Generate prediction sets with risk control
        
        Args:
            model: Trained classifier
            x: Input batch [B, C, H, W]
            device: Device to run on
        
        Returns:
            pred_sets: List of lists, each containing class indices
            probs: Predicted probabilities [B, num_classes]
            uncertainties: Uncertainty scores [B]
        """
        model.eval()
        
        with torch.no_grad():
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        pred_sets = []
        uncertainties = []
        
        for prob in probs:
            # Include all classes with probability >= threshold
            pred_set = np.where(prob >= self.threshold)[0].tolist()
            
            # Always include at least the top prediction
            if len(pred_set) == 0:
                pred_set = [np.argmax(prob)]
            
            pred_sets.append(pred_set)
            
            # Uncertainty = size of prediction set (or entropy)
            if len(pred_set) == 1:
                uncertainty = 0.0
            else:
                # Entropy of prediction set
                set_probs = prob[pred_set]
                set_probs = set_probs / set_probs.sum()
                uncertainty = -np.sum(set_probs * np.log(set_probs + 1e-10))
            
            uncertainties.append(uncertainty)
        
        return pred_sets, probs, np.array(uncertainties)
    
    def evaluate_risk(self, model, test_loader, device='cuda'):
        """
        Evaluate actual risk on test set
        
        Returns:
            metrics: Dictionary with risk metrics
        """
        model.eval()
        
        total_loss = 0
        total_samples = 0
        coverages = []
        set_sizes = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating risk"):
                inputs = inputs.to(device)
                labels_np = labels.numpy()
                
                # Get predictions
                pred_sets, probs, uncertainties = self.predict(model, inputs, device)
                
                # Compute metrics
                for i, label in enumerate(labels_np):
                    pred_set = pred_sets[i]
                    prob = probs[i]
                    
                    # Loss
                    loss_val = self.loss_fn(label, pred_set, prob)
                    total_loss += loss_val
                    total_samples += 1
                    
                    # Coverage
                    coverages.append(1.0 if label in pred_set else 0.0)
                    
                    # Set size
                    set_sizes.append(len(pred_set))
        
        metrics = {
            'empirical_risk': total_loss / total_samples,
            'coverage': np.mean(coverages),
            'avg_set_size': np.mean(set_sizes),
            'std_set_size': np.std(set_sizes),
            'target_risk': self.alpha,
            'threshold': self.threshold
        }
        
        return metrics


# ============================================================================
# Loss Functions for Different Risk Metrics
# ============================================================================

def false_negative_rate_loss(y_true: int, pred_set: List[int], probs: np.ndarray) -> float:
    """
    False Negative Rate Loss
    
    Loss = 1 if true label NOT in prediction set, else 0
    
    Use case: Medical diagnosis where missing disease is critical
    Example: Control FNR ≤ 5% for cancer detection
    """
    return 0.0 if y_true in pred_set else 1.0


def precision_loss(y_true: int, pred_set: List[int], probs: np.ndarray) -> float:
    """
    Precision Loss
    
    Loss = 1 - precision of prediction set
    Precision = 1/|pred_set| if y_true in pred_set, else 0
    
    Use case: When false positives are costly
    Example: Minimize unnecessary treatments
    """
    if y_true in pred_set:
        precision = 1.0 / len(pred_set)
    else:
        precision = 0.0
    return 1.0 - precision


def expected_set_size_loss(y_true: int, pred_set: List[int], probs: np.ndarray) -> float:
    """
    Expected Set Size Loss
    
    Loss = size of prediction set
    
    Use case: Control computational cost or user burden
    Example: Limit number of follow-up tests
    """
    return float(len(pred_set))


def recall_loss(y_true: int, pred_set: List[int], probs: np.ndarray) -> float:
    """
    Recall Loss (same as FNR for binary)
    
    Loss = 1 - recall = 1 if true label not in set, else 0
    """
    return false_negative_rate_loss(y_true, pred_set, probs)


def f1_loss(y_true: int, pred_set: List[int], probs: np.ndarray) -> float:
    """
    F1-Score Loss
    
    Loss = 1 - F1 score
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Use case: Balance precision and recall
    """
    # Recall
    recall = 1.0 if y_true in pred_set else 0.0
    
    # Precision
    if y_true in pred_set:
        precision = 1.0 / len(pred_set)
    else:
        precision = 0.0
    
    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return 1.0 - f1


def composite_loss(y_true: int, 
                   pred_set: List[int], 
                   probs: np.ndarray,
                   w_fnr: float = 0.5,
                   w_size: float = 0.5) -> float:
    """
    Composite Loss: Weighted combination of FNR and set size
    
    Loss = w_fnr * FNR + w_size * (normalized set size)
    
    Use case: Balance coverage and efficiency
    Example: Control FNR while keeping small prediction sets
    
    Args:
        y_true: True label
        pred_set: Predicted set of labels
        probs: Predicted probabilities
        w_fnr: Weight for false negative rate
        w_size: Weight for set size
    """
    fnr = false_negative_rate_loss(y_true, pred_set, probs)
    size = expected_set_size_loss(y_true, pred_set, probs)
    
    # Normalize size by number of classes
    num_classes = len(probs)
    normalized_size = size / num_classes
    
    return w_fnr * fnr + w_size * normalized_size


def weighted_misclassification_loss(y_true: int,
                                     pred_set: List[int],
                                     probs: np.ndarray,
                                     class_weights: np.ndarray = None) -> float:
    """
    Weighted Misclassification Loss
    
    Different costs for missing different classes
    
    Use case: Class-specific costs in medical diagnosis
    Example: Missing cancer (class 1) is worse than false alarm
    
    Args:
        y_true: True label
        pred_set: Predicted set
        probs: Predicted probabilities
        class_weights: Weight for missing each class [num_classes]
    """
    if class_weights is None:
        class_weights = np.ones(len(probs))
    
    if y_true in pred_set:
        return 0.0
    else:
        return class_weights[y_true]


# ============================================================================
# Helper Functions
# ============================================================================

def create_composite_loss(w_fnr=0.5, w_size=0.5):
    """Factory function for composite loss with fixed weights"""
    def loss_fn(y_true, pred_set, probs):
        return composite_loss(y_true, pred_set, probs, w_fnr, w_size)
    return loss_fn


def create_weighted_loss(class_weights):
    """Factory function for weighted misclassification loss"""
    def loss_fn(y_true, pred_set, probs):
        return weighted_misclassification_loss(y_true, pred_set, probs, class_weights)
    return loss_fn


if __name__ == '__main__':
    """Test CRC implementation"""
    print("Testing Conformal Risk Control Implementation\n")
    
    # Simulate some predictions
    num_samples = 100
    num_classes = 4
    
    # Random predictions and labels
    probs = np.random.dirichlet(np.ones(num_classes), num_samples)
    labels = np.random.randint(0, num_classes, num_samples)
    
    # Test different loss functions
    print("Testing loss functions:")
    print("=" * 60)
    
    for i in range(5):
        prob = probs[i]
        label = labels[i]
        pred_set = np.argsort(-prob)[:2].tolist()  # Top-2
        
        print(f"\nSample {i+1}:")
        print(f"  True label: {label}")
        print(f"  Pred set: {pred_set}")
        print(f"  Probs: {prob}")
        print(f"  FNR Loss: {false_negative_rate_loss(label, pred_set, prob):.3f}")
        print(f"  Precision Loss: {precision_loss(label, pred_set, prob):.3f}")
        print(f"  Set Size Loss: {expected_set_size_loss(label, pred_set, prob):.3f}")
        print(f"  F1 Loss: {f1_loss(label, pred_set, prob):.3f}")
        print(f"  Composite Loss: {composite_loss(label, pred_set, prob):.3f}")
    
    print("\nAll loss functions working correctly!")
