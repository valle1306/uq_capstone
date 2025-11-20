# Experiment #3: Adam Optimizer Experiments

## Overview
Following medical imaging literature, we implement all 4 UQ methods with Adam optimizer instead of SGD. This provides a fair comparison for medical imaging applications where Adam is the standard optimizer.

## Scripts Created

### Training Scripts (src/)
1. **`train_baseline_adam.py`** - Baseline model with Adam
2. **`train_swag_adam.py`** - SWAG with Adam (medical imaging adaptation)
3. **`train_mc_dropout_adam.py`** - MC Dropout with Adam
4. **`train_ensemble_adam.py`** - Deep Ensemble with Adam

### SLURM Scripts (scripts/amarel/)
1. **`train_baseline_adam.slurm`** - Submit baseline training
2. **`train_swag_adam.slurm`** - Submit SWAG training
3. **`train_mc_dropout_adam.slurm`** - Submit MC Dropout training
4. **`train_ensemble_adam.slurm`** - Submit ensemble training (job array for 5 members)

## Hyperparameters

All methods use identical hyperparameters for fair comparison:

```python
# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,              # Standard for medical imaging (Mehta et al. 2021)
    betas=(0.9, 0.999),     # Default PyTorch values
    eps=1e-8,
    weight_decay=5e-4        # Matching SGD experiments
)

# Learning rate schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,                # Total epochs
    eta_min=0.00005          # Final learning rate (lr_final)
)

# Training
epochs = 50
batch_size = 32
```

### SWAG-Specific Parameters
```python
swa_start = 27              # Epoch to start collection (last 46% of training)
max_num_models = 20         # Maximum snapshots to collect
swag_rank = 20              # Low-rank covariance approximation
scale = 0.5                 # Sampling scale (0.0=mean, 1.0=full std)
```

### MC Dropout Parameters
```python
dropout_rate = 0.2          # Lighter than 0.3 for medical imaging
```

### Ensemble Parameters
```python
num_members = 5             # Standard ensemble size
member_seeds = [42, 1042, 2042, 3042, 4042]  # Different for each member
```

## Submission Commands

On Amarel, run:

```bash
# Navigate to project directory
cd ~/uq_capstone

# Submit all Adam experiments
sbatch scripts/amarel/train_baseline_adam.slurm
sbatch scripts/amarel/train_swag_adam.slurm
sbatch scripts/amarel/train_mc_dropout_adam.slurm
sbatch scripts/amarel/train_ensemble_adam.slurm  # This submits 5 jobs via array

# Check job status
squeue -u $USER

# Monitor logs (example for baseline)
tail -f logs/baseline_adam_*.log
```

## Expected Runtime

| Method | Runtime per Job | Total GPU Hours |
|--------|----------------|-----------------|
| Baseline | ~1 hour | 1 |
| SWAG | ~1 hour | 1 |
| MC Dropout | ~1 hour | 1 |
| Ensemble | ~1 hour × 5 | 5 |
| **TOTAL** | | **8 GPU hours** |

Note: Ensemble uses job array, so all 5 members run in parallel if resources available.

## Results Location

Results will be saved to:
- `results/baseline_adam/`
- `results/swag_adam/`
- `results/mc_dropout_adam/`
- `results/ensemble_adam/`

Each directory contains:
- `config.json` - Training configuration
- `best_model.pt` - Best model checkpoint
- `final_model.pt` - Final epoch checkpoint
- `history.json` - Training/validation metrics per epoch

## Literature References

1. **Mehta et al. (2021):** "Propagating uncertainty across cascaded medical imaging tasks"
   - Used Adam with lr=0.0002 for medical imaging SWAG
   - Demonstrated successful uncertainty propagation

2. **Adams & Elhabian (2023):** "Benchmarking scalable epistemic UQ in organ segmentation"
   - Applied SWAG with Adam for 3D medical image segmentation
   - Showed robust uncertainty estimates in medical context

3. **Original SWAG:** Maddox et al. (2019) - Used SGD for CIFAR-10
   - This experiment adapts SWAG for medical imaging with Adam

## Key Differences from Original SWAG Paper

| Aspect | Original (Maddox 2019) | Our Adaptation |
|--------|------------------------|----------------|
| Optimizer | SGD (momentum=0.9) | Adam (betas=0.9, 0.999) |
| Learning Rate | 0.05 → 0.01 | 0.0001 → 0.00005 |
| Dataset | CIFAR-10 (50K) | Chest X-ray (4.1K) |
| Epochs | 300 | 50 |
| Collection | 161-300 (46%) | 27-50 (46%) |
| Initialization | Random | Pretrained ImageNet |

## Expected Outcomes

Based on previous Adam experiments (Experiment #0):
- **Baseline-Adam:** 90-92% accuracy (previously achieved 91.67%)
- **SWAG-Adam:** 88-91% accuracy with uncertainty quantification
- **MC Dropout-Adam:** 89-91% accuracy
- **Ensemble-Adam:** 90-92% accuracy (best uncertainty estimates)

**Hypothesis:** Adam experiments will outperform SGD experiments (Experiment #1/#2) due to:
1. Adam better suited for fine-tuning pretrained models
2. Adaptive learning rates handle class imbalance better
3. Medical imaging literature precedent

## Next Steps After Completion

1. Download results from Amarel
2. Generate uncertainty metrics (ECE, NLL, Brier score)
3. Apply conformal prediction for calibration
4. Compare with SGD experiments (Experiment #1/#2)
5. Update thesis with findings

---
**Status:** Scripts ready, awaiting submission to Amarel  
**Created:** 2025-01-XX  
**Updated:** 2025-01-XX
