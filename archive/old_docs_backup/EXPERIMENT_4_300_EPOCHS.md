# Experiment #4: 300-Epoch Experiments (Original Paper Duration)

## Overview
Following the original SWAG paper (Maddox et al. 2019), we train all 4 UQ methods for 300 epochs to match the CIFAR-10 training duration. This provides:
1. Direct comparison to the original paper methodology
2. More SGD iterates for better posterior approximation
3. Sufficient exploration of the loss landscape

## Scripts Created

### Training Scripts (src/)
1. **`train_baseline_sgd_300.py`** - Baseline with SGD, 300 epochs
2. **`train_swag_proper_300.py`** - SWAG with SGD, 300 epochs, collection 162-300
3. **`train_mc_dropout_sgd_300.py`** - MC Dropout with SGD, 300 epochs
4. **`train_ensemble_sgd_300.py`** - Deep Ensemble with SGD, 300 epochs

### SLURM Scripts (scripts/amarel/)
1. **`train_baseline_sgd_300.slurm`** - Submit baseline training
2. **`train_swag_sgd_300.slurm`** - Submit SWAG training
3. **`train_mc_dropout_sgd_300.slurm`** - Submit MC Dropout training
4. **`train_ensemble_sgd_300.slurm`** - Submit ensemble training (job array for 5 members)

## Hyperparameters

All methods use identical hyperparameters matching the original SWAG paper:

```python
# Optimizer (SGD as in original paper)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,                # Initial learning rate
    momentum=0.9,           # Matching original paper
    weight_decay=1e-4       # L2 regularization
)

# Learning rate schedule (cosine annealing)
# From lr_init=0.01 to swa_lr=0.005 over swa_start epochs
# Then constant at 0.005 for collection period

# Training
epochs = 300                # Matching CIFAR-10 experiments
batch_size = 32
```

### SWAG-Specific Parameters
```python
swa_start = 162             # Epoch to start collection (54% through training)
                            # Matches original: 161/300 = 53.7%
swa_lr = 0.005              # Collection learning rate
max_num_models = 20         # Maximum snapshots to collect
swa_c_epochs = 1            # Collect every epoch
```

### Key Difference from 50-Epoch Experiments

| Aspect | 50 Epochs (Exp #1-3) | 300 Epochs (Exp #4) |
|--------|---------------------|---------------------|
| Total Training | 50 epochs | 300 epochs |
| SWAG Collection | Epochs 27-50 (24 epochs) | Epochs 162-300 (138 epochs) |
| Collection % | 46% of training | 46% of training |
| LR Schedule Reference | swa_start=27 | swa_start=162 |
| Runtime | ~1 hour | ~6 hours |

## Submission Commands

On Amarel, run:

```bash
# Navigate to project directory
cd ~/uq_capstone

# Submit all 300-epoch experiments
sbatch scripts/amarel/train_baseline_sgd_300.slurm
sbatch scripts/amarel/train_swag_sgd_300.slurm
sbatch scripts/amarel/train_mc_dropout_sgd_300.slurm
sbatch scripts/amarel/train_ensemble_sgd_300.slurm  # This submits 5 jobs via array

# Check job status
squeue -u $USER

# Monitor logs (example for baseline)
tail -f logs/baseline_sgd_300_*.log
```

## Expected Runtime

| Method | Runtime per Job | Total GPU Hours |
|--------|----------------|-----------------|
| Baseline | ~6 hours | 6 |
| SWAG | ~6 hours | 6 |
| MC Dropout | ~6 hours | 6 |
| Ensemble | ~6 hours × 5 | 30 |
| **TOTAL** | | **48 GPU hours** |

Note: Ensemble uses job array, so all 5 members run in parallel if resources available.

## Results Location

Results will be saved to:
- `results/baseline_sgd_300/`
- `results/swag_sgd_300/`
- `results/mc_dropout_sgd_300/`
- `results/ensemble_sgd_300/`

## Rationale for 300 Epochs

### From Original SWAG Paper (CIFAR-10)
- **Training:** 300 epochs total
- **Collection:** Epochs 161-300 (last 46%)
- **Result:** Collected 138 snapshots of SGD iterates
- **Benefit:** More samples = better posterior approximation

### Why This Matters for Our Work

1. **More SGD Iterates:** 138 snapshots (vs 24 in 50-epoch version)
   - Better approximation of weight space geometry
   - More diverse samples from the SGD trajectory
   - Improved covariance estimation

2. **Better Loss Landscape Exploration:**
   - Longer training explores more of the loss surface
   - Reduces risk of underfitting
   - Captures more modes in posterior distribution

3. **Direct Comparison to Original Paper:**
   - Same training duration as CIFAR-10 experiments
   - Validates SWAG on medical imaging with proper setup
   - Addresses reviewer concerns about implementation fidelity

### Potential Concerns

**Q: Will 300 epochs cause overfitting on our smaller dataset?**

A: Possibly, but we have mitigations:
- Pretrained ImageNet weights (transfer learning)
- Weight decay (L2 regularization)
- Data augmentation (random flips, rotations, color jitter)
- Validation monitoring (can do early stopping post-hoc)

**Q: Is 300 epochs necessary for pretrained models?**

A: The original paper uses random initialization. However:
- Testing with 300 epochs validates the methodology fully
- Provides comparison point for 50-epoch experiments
- Medical imaging literature varies (some use 50, some 200+)
- This experiment answers: "Do we need long training for SWAG?"

## Expected Outcomes

Based on original paper and medical imaging literature:

| Metric | 50 Epochs (Exp #1) | 300 Epochs (Exp #4) | Expected Change |
|--------|-------------------|---------------------|-----------------|
| Accuracy | 79-82% | 82-85% | +3% (more training) |
| ECE | 0.05-0.08 | 0.04-0.07 | -0.01 (better calibration) |
| SWAG Snapshots | 24 | 138 | +114 (5.75× more) |
| Training Time | 1 hour | 6 hours | 6× longer |

**Hypothesis:** 
- **Accuracy:** Modest improvement (2-3%) from longer training
- **Uncertainty:** Significantly better (SWAG benefits from more snapshots)
- **Calibration:** Improved ECE and Brier scores
- **Trade-off:** Diminishing returns after ~100-150 epochs for pretrained models

## Key Research Questions

1. **Is 300 epochs necessary for medical imaging SWAG?**
   - Compare Exp #1 (50 epochs) vs Exp #4 (300 epochs)
   - Measure: Accuracy, ECE, NLL, Brier score

2. **How many SWAG snapshots are sufficient?**
   - 24 snapshots (50 epochs) vs 138 snapshots (300 epochs)
   - Does more snapshots = better uncertainty?

3. **Does pretrained transfer learning change optimal epoch count?**
   - Original paper: random init, 300 epochs
   - Our work: pretrained ImageNet, 50 vs 300 epochs
   - Expected: Pretrained models may saturate earlier

## Next Steps After Completion

1. **Download results from Amarel**
   ```bash
   rsync -avz amarel:~/uq_capstone/results/ results/
   ```

2. **Generate comparative analysis:**
   - Plot accuracy/loss curves (50 vs 300 epochs)
   - Compare calibration metrics (ECE, NLL, Brier)
   - Analyze SWAG posterior quality (eigenvalue spectra)

3. **Update thesis with findings:**
   - Add 300-epoch results to experimental section
   - Discuss epoch count sensitivity
   - Provide recommendations for medical imaging SWAG

4. **Create visualization:**
   - Training curves: 50 vs 300 epochs
   - Uncertainty quality: SWAG snapshot count impact
   - Computational cost-benefit analysis

---
**Status:** Scripts ready, awaiting submission to Amarel  
**Created:** 2025-01-XX  
**Updated:** 2025-01-XX  
**Estimated Completion:** ~6 hours after submission
