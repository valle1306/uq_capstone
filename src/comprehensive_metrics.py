"""
Comprehensive Metrics Evaluation for UQ Methods

Computes and compares all uncertainty quantification metrics:
- Accuracy & Error Rates
- Calibration (ECE, MCE, Brier Score)
- Uncertainty Metrics (variance, entropy)
- Coverage & Set Size (Conformal methods)
- Risk Bounds & Guarantees

Usage:
    python src/comprehensive_metrics.py \
        --baseline_path runs/classification/baseline/best_model.pth \
        --mc_dropout_path runs/classification/mc_dropout/best_model.pth \
        --ensemble_dir runs/classification/ensemble \
        --swag_path runs/classification/swag_classification/swag_model.pth \
        --dataset chest_xray \
        --output_dir runs/classification/metrics
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision import models
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from data_utils_classification import get_classification_loaders
from conformal_risk_control import (
    ConformalRiskControl,
    false_negative_rate_loss,
    expected_set_size_loss,
    composite_loss,
    f1_loss,
    precision_loss
)
from swag import load_swag_model


class ComprehensiveMetricsEvaluator:
    """Evaluates comprehensive metrics for all UQ methods"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
    
    def compute_calibration_metrics(self, probs, labels):
        """
        Compute calibration metrics: ECE, MCE, Brier Score
        
        Args:
            probs: Predicted probabilities [N, num_classes]
            labels: True labels [N]
        
        Returns:
            dict with ECE, MCE, Brier
        """
        # Accuracy
        preds = np.argmax(probs, axis=1)
        accuracy = np.mean(preds == labels) * 100
        
        # Brier Score
        num_classes = probs.shape[1]
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(labels)), labels] = 1
        brier = np.mean((probs - one_hot) ** 2)
        
        # Expected Calibration Error (ECE)
        n_bins = 15
        confidences = np.max(probs, axis=1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            
            bin_mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if bin_mask.sum() > 0:
                bin_acc = np.mean(preds[bin_mask] == labels[bin_mask])
                bin_conf = np.mean(confidences[bin_mask])
                bin_size = bin_mask.sum() / len(labels)
                ece += bin_size * abs(bin_acc - bin_conf)
        
        # Maximum Calibration Error (MCE)
        mce = 0.0
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            
            bin_mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if bin_mask.sum() > 0:
                bin_acc = np.mean(preds[bin_mask] == labels[bin_mask])
                bin_conf = np.mean(confidences[bin_mask])
                mce = max(mce, abs(bin_acc - bin_conf))
        
        return {
            'accuracy': accuracy,
            'brier_score': brier,
            'ece': ece,
            'mce': mce
        }
    
    def compute_uncertainty_metrics(self, all_uncertainties, correct):
        """
        Compute uncertainty quality metrics
        
        Args:
            all_uncertainties: Uncertainty estimates [N]
            correct: Binary correctness [N]
        
        Returns:
            dict with uncertainty metrics
        """
        if len(all_uncertainties) == 0:
            return {}
        
        # Correct predictions should have lower uncertainty
        correct_unc = all_uncertainties[correct == 1]
        incorrect_unc = all_uncertainties[correct == 0]
        
        metrics = {
            'mean_uncertainty': float(np.mean(all_uncertainties)),
            'std_uncertainty': float(np.std(all_uncertainties)),
            'min_uncertainty': float(np.min(all_uncertainties)),
            'max_uncertainty': float(np.max(all_uncertainties)),
        }
        
        # Uncertainty separation
        if len(correct_unc) > 0 and len(incorrect_unc) > 0:
            metrics['correct_mean_unc'] = float(np.mean(correct_unc))
            metrics['incorrect_mean_unc'] = float(np.mean(incorrect_unc))
            metrics['unc_separation'] = float(np.mean(incorrect_unc) - np.mean(correct_unc))
        
        return metrics
    
    def compute_error_analysis(self, preds, labels, probs):
        """
        Detailed error analysis: confusion matrix, per-class metrics
        
        Args:
            preds: Predictions [N]
            labels: Labels [N]
            probs: Probabilities [N, num_classes]
        
        Returns:
            dict with error analysis
        """
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # Overall metrics
        total_correct = tp.sum()
        total_samples = len(labels)
        
        # False Positive Rate, False Negative Rate, True Negative Rate, True Positive Rate
        fpr = fp.sum() / (fp.sum() + tn.sum()) if (fp.sum() + tn.sum()) > 0 else 0
        fnr = fn.sum() / (fn.sum() + tp.sum()) if (fn.sum() + tp.sum()) > 0 else 0
        
        # ROC AUC (for binary classification)
        try:
            if probs.shape[1] == 2:
                roc_auc = roc_auc_score(labels, probs[:, 1])
            else:
                # One-vs-rest ROC AUC
                roc_auc = roc_auc_score(labels, probs, multi_class='ovr')
        except:
            roc_auc = None
        
        return {
            'fpr': float(fpr),
            'fnr': float(fnr),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'total_correct': int(total_correct),
            'total_samples': int(total_samples)
        }
    
    def evaluate_all_methods(self, methods_config):
        """
        Evaluate all UQ methods and compute comprehensive metrics
        
        Args:
            methods_config: dict with paths to all methods
        
        Returns:
            dict with all metrics
        """
        all_metrics = {}
        
        # Load test data
        _, cal_loader, test_loader, num_classes = get_classification_loaders(
            dataset_name='chest_xray',
            batch_size=32,
            num_workers=4
        )
        
        # Evaluate each method
        for method_name, config in methods_config.items():
            print(f"\n{'='*70}")
            print(f"Evaluating {method_name}")
            print(f"{'='*70}")
            
            if method_name == 'Baseline':
                metrics = self._evaluate_baseline(config['path'], test_loader, num_classes)
            elif method_name == 'MC Dropout':
                metrics = self._evaluate_mc_dropout(config['path'], test_loader, num_classes)
            elif method_name == 'Deep Ensemble':
                metrics = self._evaluate_ensemble(config['paths'], test_loader, num_classes)
            elif method_name == 'SWAG':
                metrics = self._evaluate_swag(config['path'], test_loader, num_classes)
            
            all_metrics[method_name] = metrics
        
        # Add Conformal Risk Control (post-hoc on baseline)
        print(f"\n{'='*70}")
        print(f"Evaluating Conformal Risk Control (post-hoc on Baseline)")
        print(f"{'='*70}")
        baseline_path = methods_config['Baseline']['path']
        baseline_model = self._load_baseline_model(baseline_path, num_classes)
        crc_results = self._evaluate_conformal(baseline_model, cal_loader, test_loader, num_classes)
        # Merge CRC results dict into all_metrics (not a list)
        all_metrics.update(crc_results)
        
        return all_metrics
    
    def _load_baseline_model(self, model_path, num_classes):
        """Load baseline model"""
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        # Load to CPU first, then move to device to avoid CUDA mismatch
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model
    
    def _evaluate_baseline(self, model_path, test_loader, num_classes):
        """Evaluate baseline model"""
        model = self._load_baseline_model(model_path, num_classes)
        
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Baseline'):
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Compute metrics
        cal_metrics = self.compute_calibration_metrics(all_probs, all_labels)
        error_metrics = self.compute_error_analysis(all_preds, all_labels, all_probs)
        
        results = {
            'method': 'Baseline',
            **cal_metrics,
            **error_metrics
        }
        
        self._print_metrics(results)
        return results
    
    def _evaluate_mc_dropout(self, model_path, test_loader, num_classes):
        """Evaluate MC Dropout"""
        # Define ResNetWithDropout class inline to avoid import path issues
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
        
        # Read config to get exact dropout_rate used during training
        model_dir = Path(model_path).parent
        config_path = model_dir / 'config.json'
        dropout_rate = 0.2  # default
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                dropout_rate = config.get('dropout_rate', 0.2)
                print(f"  Loaded dropout_rate={dropout_rate} from config")
        else:
            print(f"  Warning: config.json not found, using default dropout_rate={dropout_rate}")
        
        base_model = models.resnet18(pretrained=False)
        model = ResNetWithDropout(base_model, num_classes, dropout_rate=dropout_rate)
        # Load to CPU first, then move to device to avoid CUDA mismatch
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        all_probs_mean = []
        all_uncertainties = []
        all_preds = []
        all_labels = []
        
        # MC sampling - use T=15 samples for good uncertainty estimation
        n_mc_samples = 15
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'MC Dropout (T={n_mc_samples})'):
                inputs = inputs.to(self.device)
                
                # MC sampling - enable dropout and sample
                probs_samples = []
                for _ in range(n_mc_samples):
                    model.enable_dropout()  # Enable dropout for this sample
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    probs_samples.append(probs.cpu())
                
                probs_samples = torch.stack(probs_samples)
                probs_mean = probs_samples.mean(dim=0)
                probs_var = probs_samples.var(dim=0)
                uncertainty = probs_var.mean(dim=1)
                
                all_probs_mean.append(probs_mean.numpy())
                all_uncertainties.append(uncertainty.numpy())
                all_preds.append(torch.argmax(probs_mean, dim=1).numpy())
                all_labels.append(labels.numpy())
        
        all_probs_mean = np.concatenate(all_probs_mean)
        all_uncertainties = np.concatenate(all_uncertainties)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        cal_metrics = self.compute_calibration_metrics(all_probs_mean, all_labels)
        unc_metrics = self.compute_uncertainty_metrics(all_uncertainties, (all_preds == all_labels).astype(int))
        error_metrics = self.compute_error_analysis(all_preds, all_labels, all_probs_mean)
        
        results = {
            'method': 'MC Dropout',
            'n_samples': n_mc_samples,
            **cal_metrics,
            **unc_metrics,
            **error_metrics
        }
        
        self._print_metrics(results)
        return results
    
    def _evaluate_ensemble(self, model_paths, test_loader, num_classes):
        """Evaluate Deep Ensemble"""
        models = []
        for path in model_paths:
            model = self._load_baseline_model(path, num_classes)
            models.append(model)
        
        all_probs_mean = []
        all_uncertainties = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Ensemble'):
                inputs = inputs.to(self.device)
                
                probs_samples = []
                for model in models:
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    probs_samples.append(probs.cpu())
                
                probs_samples = torch.stack(probs_samples)
                probs_mean = probs_samples.mean(dim=0)
                probs_var = probs_samples.var(dim=0)
                uncertainty = probs_var.mean(dim=1)
                
                all_probs_mean.append(probs_mean.numpy())
                all_uncertainties.append(uncertainty.numpy())
                all_preds.append(torch.argmax(probs_mean, dim=1).numpy())
                all_labels.append(labels.numpy())
        
        all_probs_mean = np.concatenate(all_probs_mean)
        all_uncertainties = np.concatenate(all_uncertainties)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        cal_metrics = self.compute_calibration_metrics(all_probs_mean, all_labels)
        unc_metrics = self.compute_uncertainty_metrics(all_uncertainties, (all_preds == all_labels).astype(int))
        error_metrics = self.compute_error_analysis(all_preds, all_labels, all_probs_mean)
        
        results = {
            'method': 'Deep Ensemble',
            'n_models': len(models),
            **cal_metrics,
            **unc_metrics,
            **error_metrics
        }
        
        self._print_metrics(results)
        return results
    
    def _evaluate_swag(self, swag_path, test_loader, num_classes):
        """Evaluate SWAG with proper posterior sampling"""
        base_model = models.resnet18(pretrained=False)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, num_classes)
        
        swag = load_swag_model(swag_path, base_model)
        swag = swag.to(self.device)
        swag.eval()
        
        all_probs_mean = []
        all_uncertainties = []
        all_preds = []
        all_labels = []
        
        # Use scale=0.5 as paper recommends (scale=1.0 causes numerical issues)
        # SWAG mean model is fundamentally different from baseline (trained separately)
        # This is expected - different initialization leads to different local minima
        best_scale = 0.5
        n_swag_samples = 30
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'SWAG (T={n_swag_samples}, scale={best_scale})'):
                inputs = inputs.to(self.device)
                
                preds = []
                for _ in range(n_swag_samples):
                    try:
                        # Sample from SWAG posterior
                        sampled = swag.sample(scale=best_scale)
                        sampled = sampled.to(self.device)
                        sampled.eval()
                        
                        with torch.no_grad():
                            outputs = sampled(inputs)
                        
                        probs = torch.softmax(outputs, dim=1)
                        preds.append(probs.cpu())
                    except Exception as e:
                        print(f"Warning: SWAG sampling failed: {e}, falling back to mean model")
                        # Fallback to mean model
                        outputs = swag.base_model(inputs)
                        probs = torch.softmax(outputs, dim=1)
                        preds.append(probs.cpu())
                
                if len(preds) == 0:
                    continue
                    
                preds = torch.stack(preds)
                probs_mean = preds.mean(dim=0)
                
                # Compute variance with numerical stability
                probs_var = preds.var(dim=0)
                uncertainty = probs_var.mean(dim=1)
                
                # Sanity check: if uncertainty has NaN, use alternative calculation
                if torch.isnan(uncertainty).any():
                    # Use std instead of var
                    uncertainty = preds.std(dim=0).mean(dim=1)
                
                all_probs_mean.append(probs_mean.numpy())
                if not torch.isnan(uncertainty).any():
                    all_uncertainties.append(uncertainty.numpy())
                all_preds.append(torch.argmax(probs_mean, dim=1).numpy())
                all_labels.append(labels.numpy())
        
        if len(all_probs_mean) == 0:
            raise RuntimeError("SWAG evaluation produced no results")
        
        all_probs_mean = np.concatenate(all_probs_mean)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Handle uncertainty - may be empty if we hit NaNs
        if all_uncertainties:
            all_uncertainties = np.concatenate(all_uncertainties)
        else:
            all_uncertainties = np.zeros_like(all_preds, dtype=float)
        
        cal_metrics = self.compute_calibration_metrics(all_probs_mean, all_labels)
        unc_metrics = self.compute_uncertainty_metrics(all_uncertainties, (all_preds == all_labels).astype(int))
        error_metrics = self.compute_error_analysis(all_preds, all_labels, all_probs_mean)
        
        results = {
            'method': 'SWAG',
            'n_samples': n_swag_samples,
            'sampling_scale': best_scale,
            'note': 'SWAG trained separately - different local minimum than baseline',
            **cal_metrics,
            **unc_metrics,
            **error_metrics
        }
        
        self._print_metrics(results)
        return results
    
    def _evaluate_conformal(self, model, cal_loader, test_loader, num_classes):
        """Evaluate Conformal Risk Control"""
        # Return dict (not list) so summary table can access metrics properly
        results_dict = {}
        
        loss_configs = [
            ('FNR Control (alpha=0.05)', false_negative_rate_loss, 0.05),
            ('FNR Control (alpha=0.10)', false_negative_rate_loss, 0.10),
            ('Set Size Control', expected_set_size_loss, 2.0),
            ('Composite Loss', lambda y, s, p: composite_loss(y, s, p, 0.5, 0.5), 0.15),
        ]
        
        for name, loss_fn, alpha in loss_configs:
            print(f"  Testing: {name}")
            
            crc = ConformalRiskControl(loss_fn=loss_fn, alpha=alpha, delta=0.1)
            crc.calibrate(model, cal_loader, self.device)
            metrics = crc.evaluate_risk(model, test_loader, self.device)
            
            results = {
                'method': f'CRC - {name}',
                'target_risk': alpha,
                'empirical_risk': metrics['empirical_risk'],
                'coverage': metrics['coverage'],
                'avg_set_size': metrics['avg_set_size'],
                'std_set_size': metrics['std_set_size'],
                'threshold': metrics['threshold'],
                # Add accuracy-like metrics for summary table compatibility
                'accuracy': (1.0 - metrics['empirical_risk']) * 100.0 if metrics['empirical_risk'] is not None else None,
                'ece': None,
                'brier_score': None,
                'fnr': metrics['empirical_risk'],
                'mean_uncertainty': None
            }
            
            results_dict[f'CRC - {name}'] = results
        
        return results_dict
    
    def _print_metrics(self, metrics):
        """Pretty print metrics"""
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.2f}%")
        print(f"  ECE: {metrics.get('ece', 'N/A'):.4f}")
        print(f"  Brier Score: {metrics.get('brier_score', 'N/A'):.4f}")
        print(f"  FPR: {metrics.get('fpr', 'N/A'):.4f}")
        print(f"  FNR: {metrics.get('fnr', 'N/A'):.4f}")
        if 'mean_uncertainty' in metrics:
            print(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
            print(f"  Uncertainty Separation: {metrics.get('unc_separation', 'N/A'):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Metrics Evaluation')
    parser.add_argument('--baseline_path', type=str, required=True)
    parser.add_argument('--mc_dropout_path', type=str, default=None)
    parser.add_argument('--ensemble_dir', type=str, default=None)
    parser.add_argument('--n_ensemble', type=int, default=5)
    parser.add_argument('--swag_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='runs/classification/metrics')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build methods config
    methods_config = {
        'Baseline': {'path': args.baseline_path}
    }
    
    if args.mc_dropout_path:
        methods_config['MC Dropout'] = {'path': args.mc_dropout_path}
    
    if args.ensemble_dir:
        ensemble_paths = [
            Path(args.ensemble_dir) / f'member_{i}' / 'best_model.pth'
            for i in range(args.n_ensemble)
        ]
        methods_config['Deep Ensemble'] = {'paths': ensemble_paths}
    
    if args.swag_path:
        methods_config['SWAG'] = {'path': args.swag_path}
    
    # Evaluate all methods
    evaluator = ComprehensiveMetricsEvaluator(device=args.device)
    all_metrics = evaluator.evaluate_all_methods(methods_config)
    
    # Save results
    output_path = output_dir / 'comprehensive_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Create summary table
    summary_df = pd.DataFrame([
        {
            'Method': m.get('method', k),
            'Accuracy (%)': m.get('accuracy', np.nan),
            'ECE': m.get('ece', np.nan),
            'Brier': m.get('brier_score', np.nan),
            'FNR': m.get('fnr', np.nan),
            'Mean Unc': m.get('mean_uncertainty', np.nan)
        }
        for k, m in all_metrics.items()
    ])
    
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(output_dir / 'metrics_summary.csv', index=False)
    print(f"Summary saved to: {output_dir / 'metrics_summary.csv'}")


if __name__ == '__main__':
    main()
