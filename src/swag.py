"""
SWAG (Stochastic Weight Averaging-Gaussian) for Uncertainty Quantification
Implementation based on Maddox et al. "A Simple Baseline for Bayesian Uncertainty Estimation in Deep Learning"

SWAG maintains a running average of model weights and their deviations to approximate
the posterior distribution over weights, enabling Bayesian uncertainty estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from collections import OrderedDict
import copy


class SWAG(nn.Module):
    """
    SWAG wrapper for any PyTorch model.
    
    Collects model snapshots during training and fits a Gaussian approximation
    to the posterior distribution over weights.
    
    Args:
        base_model: Base neural network model
        max_num_models: Maximum number of model snapshots to store (K parameter)
        var_clamp: Minimum variance for numerical stability
    """
    
    def __init__(self, base_model: nn.Module, max_num_models: int = 20, var_clamp: float = 1e-30, max_var: float = 100.0):
        super().__init__()
        self.base_model = base_model
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        self.max_var = max_var  # Cap maximum variance
        
        # Storage for SWAG statistics
        self.n_models = 0
        self.mean = None  # Running mean of weights
        self.sq_mean = None  # Running mean of squared weights (for variance)
        self.cov_mat_sqrt = []  # Low-rank covariance approximation
        
        # Initialize mean and sq_mean with current model weights
        self._init_swag()
    
    def _init_swag(self):
        """Initialize SWAG statistics with base model weights"""
        self.mean = self._flatten_params(self.base_model.state_dict())
        self.sq_mean = self.mean ** 2
    
    def _flatten_params(self, state_dict: OrderedDict) -> torch.Tensor:
        """Flatten model parameters into a single vector"""
        return torch.cat([p.flatten() for p in state_dict.values()])
    
    def _unflatten_params(self, flat_params: torch.Tensor) -> OrderedDict:
        """Unflatten parameter vector back to state dict format"""
        state_dict = self.base_model.state_dict()
        unflattened = OrderedDict()
        idx = 0
        for key, param in state_dict.items():
            numel = param.numel()
            unflattened[key] = flat_params[idx:idx+numel].view(param.shape)
            idx += numel
        return unflattened
    
    def collect_model(self, model: nn.Module):
        """
        Collect a model snapshot for SWAG.
        Call this periodically during training (e.g., every epoch after learning rate annealing).
        
        Args:
            model: Current model to collect
        """
        # Flatten current model parameters
        w = self._flatten_params(model.state_dict()).to(self.mean.device)
        
        # Update running mean: mean_new = (n*mean_old + w) / (n+1)
        self.mean = (self.mean * self.n_models + w) / (self.n_models + 1)
        
        # Update running mean of squares
        self.sq_mean = (self.sq_mean * self.n_models + w**2) / (self.n_models + 1)
        
        # Update low-rank covariance matrix
        # Store deviation from mean: D_i = w_i - mean
        dev = w - self.mean
        self.cov_mat_sqrt.append(dev.clone())
        
        # Keep only last K models for low-rank approximation
        if len(self.cov_mat_sqrt) > self.max_num_models:
            self.cov_mat_sqrt.pop(0)
        
        self.n_models += 1
    
    def sample(self, scale: float = 1.0, diag_noise: bool = True) -> nn.Module:
        """
        Sample a model from the SWAG posterior distribution.
        
        Args:
            scale: Scale factor for sampling (0.5 is common)
            diag_noise: Whether to include diagonal noise
        
        Returns:
            Sampled model with perturbed weights
        """
        if self.n_models == 0:
            raise ValueError("No models collected yet. Call collect_model() first.")
        
        # Compute diagonal variance: sigma^2 = E[w^2] - E[w]^2
        var = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp, self.max_var)
        
        # Sample from standard normal
        z1 = torch.randn_like(self.mean)
        
        # Diagonal component: mean + scale * sqrt(var) * z1
        w_sample = self.mean + scale * torch.sqrt(var) * z1
        
        # Low-rank component (if we have collected models)
        if len(self.cov_mat_sqrt) > 0 and diag_noise:
            # Stack deviations into matrix D: [num_models, num_params]
            D = torch.stack(self.cov_mat_sqrt, dim=0)
            
            # Sample from low-rank: (1/sqrt(2(K-1))) * D^T * z2
            z2 = torch.randn(len(self.cov_mat_sqrt), device=self.mean.device)
            K = len(self.cov_mat_sqrt)
            low_rank_sample = (1.0 / np.sqrt(2 * (K - 1))) * torch.matmul(D.T, z2)
            
            w_sample += scale * low_rank_sample
        
        # Create new model with sampled weights
        sampled_model = copy.deepcopy(self.base_model)
        sampled_state_dict = self._unflatten_params(w_sample)
        sampled_model.load_state_dict(sampled_state_dict)
        
        return sampled_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using mean weights"""
        return self.base_model(x)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 30,
        scale: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates by sampling from SWAG posterior.
        
        Args:
            x: Input tensor [B, C, H, W]
            n_samples: Number of posterior samples
            scale: Sampling scale (0.5 is recommended by paper)
        
        Returns:
            mean_pred: Mean prediction [B, num_classes, H, W]
            uncertainty: Predictive uncertainty [B, H, W]
        """
        self.base_model.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample model from posterior
                sampled_model = self.sample(scale=scale)
                sampled_model.eval()
                sampled_model.to(x.device)
                
                # Make prediction
                pred = torch.sigmoid(sampled_model(x))
                predictions.append(pred)
        
        # Stack predictions: [n_samples, B, num_classes, H, W]
        predictions = torch.stack(predictions, dim=0)
        
        # Compute mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Compute uncertainty (standard deviation)
        uncertainty = predictions.std(dim=0).mean(dim=1)  # Average over classes
        
        return mean_pred, uncertainty


class SWAGScheduler:
    """
    Helper class to schedule SWAG model collection during training.
    Typically collects models after learning rate annealing.
    
    Args:
        swag_model: SWAG wrapper
        collect_start_epoch: Start collecting after this epoch
        collect_freq: Collect every N epochs
    """
    
    def __init__(self, swag_model: SWAG, collect_start_epoch: int = 15, collect_freq: int = 1):
        self.swag_model = swag_model
        self.collect_start_epoch = collect_start_epoch
        self.collect_freq = collect_freq
    
    def step(self, epoch: int, model: nn.Module):
        """Call this at the end of each epoch"""
        if epoch >= self.collect_start_epoch and (epoch - self.collect_start_epoch) % self.collect_freq == 0:
            print(f"📸 SWAG: Collecting model snapshot at epoch {epoch}")
            self.swag_model.collect_model(model)


def load_swag_model(checkpoint_path: str, base_model: nn.Module) -> SWAG:
    """
    Load a trained SWAG model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        base_model: Base model architecture
    
    Returns:
        Loaded SWAG model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create SWAG wrapper
    swag = SWAG(
        base_model=base_model,
        max_num_models=checkpoint.get('max_num_models', 20)
    )
    
    # Load SWAG statistics
    swag.n_models = checkpoint['n_models']
    swag.mean = checkpoint['mean']
    swag.sq_mean = checkpoint['sq_mean']
    swag.cov_mat_sqrt = checkpoint['cov_mat_sqrt']
    
    # Load base model weights (use mean weights)
    base_state_dict = swag._unflatten_params(swag.mean)
    swag.base_model.load_state_dict(base_state_dict)
    
    return swag


if __name__ == "__main__":
    # Test SWAG with dummy model
    from model_utils import UNet
    
    print("Testing SWAG implementation...")
    
    # Create model
    model = UNet(in_channels=1, num_classes=1)
    
    # Create SWAG wrapper
    swag = SWAG(model, max_num_models=5)
    
    # Simulate collecting models
    print("Collecting 5 model snapshots...")
    for i in range(5):
        # Simulate training by adding noise to weights
        for param in model.parameters():
            param.data += torch.randn_like(param) * 0.01
        swag.collect_model(model)
    
    print(f"Collected {swag.n_models} models")
    
    # Test sampling
    print("Sampling from SWAG posterior...")
    sampled = swag.sample(scale=0.5)
    print("✓ Sampled model successfully")
    
    # Test prediction with uncertainty
    print("Testing prediction with uncertainty...")
    dummy_input = torch.randn(2, 1, 128, 128)
    mean_pred, uncertainty = swag.predict_with_uncertainty(dummy_input, n_samples=10)
    print(f"✓ Mean prediction shape: {mean_pred.shape}")
    print(f"✓ Uncertainty shape: {uncertainty.shape}")
    print(f"✓ Mean uncertainty: {uncertainty.mean().item():.4f}")
    
    print("\n✅ SWAG implementation test passed!")
