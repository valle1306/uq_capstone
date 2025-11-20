# SWAG Implementation: Corrected Approach

## Problem Statement

Initial SWAG implementation achieved only 78% accuracy compared to 90%+ baseline, indicating a critical implementation error.

## Root Cause Analysis

### Professor's Questions
1. **What are the SWAG updates if Adam is used instead of SGD?**
   - The paper explicitly uses SGD with momentum=0.9
   - Adam is mentioned in the appendix as "future work" but not implemented
   - Mixing optimizers creates an unfair comparison

2. **Compare apples to apples**
   - Original baseline used Adam optimizer
   - SWAG used SGD (as per paper)
   - Different optimizers → different convergence → unfair comparison

3. **Check conformal calibration set size**
   - Current: 80-20 split from training data
   - Status: ✅ Correct per conformal prediction literature

## Solution

### 1. Exact SWAG Implementation

File: `src/train_swag_proper.py`

Key components from Maddox et al. (2019):
- **Optimizer**: `SGD(lr=0.01, momentum=0.9, wd=1e-4)`
- **Learning Rate Schedule**: Cosine annealing from 0.01 → 0.005
- **Collection**: Epochs 27-50 (last 46% of training, scaled from paper's 161-300)
- **SWAG Parameters**: max_models=20, collect every epoch
- **Batch Normalization**: Update statistics after sampling using training set
- **Evaluation**: Sample with scale=0.0 (mean) or scale=0.5 (ensemble)

### 2. Fair Baseline Comparison

File: `src/train_baseline_sgd.py`

**NEW**: SGD baseline with identical setup:
- Same optimizer: `SGD(lr=0.01, momentum=0.9, wd=1e-4)`
- Same learning rate schedule: Cosine annealing 0.01 → 0.005
- Same architecture, batch size, epochs
- Only difference: No SWAG collection

This ensures any performance differences are due to SWAG's Bayesian model averaging, not the optimizer.

### 3. Conformal Calibration

Current setup (verified correct):
- Split training data 80-20
- 80% for model training
- 20% for calibration quantile computation
- Independent test set for final evaluation

## Training Commands

### 1. SGD Baseline
```bash
# On Amarel
sbatch scripts/amarel/train_baseline_sgd.slurm

# Local
python src/train_baseline_sgd.py \
    --dataset chest_xray \
    --data_dir data/chest_xray \
    --arch resnet18 \
    --pretrained \
    --epochs 50 \
    --batch_size 32 \
    --lr_init 0.01 \
    --swa_lr 0.005 \
    --swa_start 27 \
    --momentum 0.9 \
    --wd 1e-4 \
    --output_dir runs/classification/baseline_sgd
```

### 2. SWAG Training
```bash
# On Amarel
sbatch scripts/amarel/train_swag_proper.slurm

# Local
python src/train_swag_proper.py \
    --dataset chest_xray \
    --data_dir data/chest_xray \
    --arch resnet18 \
    --pretrained \
    --epochs 50 \
    --batch_size 32 \
    --lr_init 0.01 \
    --swa_lr 0.005 \
    --swa_start 27 \
    --swa_c_epochs 1 \
    --max_num_models 20 \
    --momentum 0.9 \
    --wd 1e-4 \
    --output_dir runs/classification/swag_proper
```

## Expected Results

Based on Maddox et al. (2019) findings:

1. **Accuracy**: SWAG ≈ Baseline (within 1-2%)
2. **Calibration**: SWAG << Baseline (much better calibrated)
3. **OOD Detection**: SWAG > Baseline (better uncertainty on out-of-distribution)
4. **Ensemble**: SWAG ensemble (30 samples) > SWAG mean > Baseline

The key benefit of SWAG is **not** improved accuracy, but:
- Better calibrated probabilities
- Reliable uncertainty estimates
- Efficient Bayesian model averaging (1 training run → 30 models)

## References

Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019). 
*A Simple Baseline for Bayesian Uncertainty in Deep Learning*. 
Neural Information Processing Systems (NeurIPS).

Official implementation: https://github.com/wjmaddox/swa_gaussian
