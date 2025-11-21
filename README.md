# Uncertainty Quantification for Medical Image Classification

A comprehensive empirical study comparing four uncertainty quantification methods on chest X-ray pneumonia detection: baseline maximum likelihood, Monte Carlo Dropout, Deep Ensembles, and Stochastic Weight Averaging-Gaussian (SWAG), with conformal prediction for provable coverage guarantees.

**Author:** Phan Nguyen Huong Le  
**Advisor:** Gemma Moran  
**Institution:** Rutgers University  
**Date:** November 2025

## Key Findings

### SGD 50-Epoch Results (ResNet-18, 4,172 training samples)

| Method | Validation Accuracy | Test Accuracy | ECE | Status |
|--------|-------------------|---------------|-----|--------|
| MC Dropout | 91.35% | 84.13% | TBD | Best validation |
| Deep Ensemble | 90.06% | 85.90% | 0.027 | Best calibration |
| Baseline | 89.90% | 84.62% | TBD | Reference |
| SWAG | 89.42% | 84.62% | TBD | Underperforms |

### Critical Discovery

SWAG underperforms on small medical datasets (4,172 samples) because rapid overfitting occurs before SWAG's collection window (epochs 28-50). Best validation accuracy occurs at epoch 1, and by epoch 27 the model has already overfit. This violates Maddox et al.'s (2019) assumption that SGD explores a broad posterior mode—small datasets cause memorization rather than exploration.

**Hypothesis:** SWAG requires at least 10-15K samples per class to prevent early overfitting and enable meaningful weight averaging.

### Conformal Prediction

Achieves stable 90% coverage (± 1%) with 1,044 calibration samples across all base models, providing distribution-free guarantees for clinical deployment.

## Repository Structure

```
uq_capstone/
├── src/                     # Training scripts for all methods
├── scripts/amarel/          # SLURM job scripts for HPC cluster
├── analysis/                # Visualization and analysis tools
├── results/                 # Experimental results and figures
├── thesis_draft.md          # Complete thesis document
└── requirements.txt         # Python dependencies
```

## Experimental Design

### Experiment 1-2 (Complete)
- **Configuration:** SGD 50-epoch, ResNet-18, momentum=0.9, cosine annealing
- **Status:** Complete (November 20, 2025)
- **Outcome:** Fair apples-to-apples comparison, reveals SWAG's small-dataset limitations

### Experiment 4 (In Progress)
- **Configuration:** SGD 300-epoch, ResNet-50, momentum=0.9, cosine annealing
- **Purpose:** Rigorous replication of original SWAG paper methodology
- **Status:** MC Dropout complete, Ensemble at epoch 148/300 (50% complete)
- **Tests:** Whether extended training + deeper architecture mitigates rapid overfitting

## Dataset

**Chest X-Ray Images (Pneumonia)** - Kermany et al. (2018)
- Total: 5,840 images (1,583 Normal, 4,273 Pneumonia)
- Split: 4,172 train / 1,044 calibration / 624 test
- Class distribution: 73% Pneumonia (imbalanced, reflects real-world prevalence)

## Methods

1. **Baseline** - Standard ResNet training via maximum likelihood
2. **MC Dropout** - Bayesian approximation via dropout at test time (T=15 passes, p=0.2)
3. **Deep Ensemble** - 5 independent models from different random initializations
4. **SWAG** - Gaussian posterior approximation from SGD weight snapshots (K=20 snapshots, T=30 samples)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- See `requirements.txt` for complete dependencies

## Key References

- **SWAG:** Maddox et al. "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (NeurIPS 2019)
- **MC Dropout:** Gal & Ghahramani "Dropout as a Bayesian Approximation" (ICML 2016)
- **Deep Ensembles:** Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty" (NIPS 2017)
- **Conformal Prediction:** Romano et al. "Conformalized Quantile Regression" (NeurIPS 2019)
- **Dataset:** Kermany et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" (Cell 2018)

## Documentation

Primary documentation is in `thesis_draft.md`, which includes:
- Complete methodology
- Experimental design and results
- Analysis of SWAG's small-dataset limitations
- Implementation details and pitfalls
- Conformal prediction calibration


This project is for academic research purposes.
