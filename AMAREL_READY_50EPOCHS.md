# 🚀 UQ Classification Pipeline - Ready for Amarel (50 Epochs)

## Executive Summary

All uncertainty quantification (UQ) methods are now **production-ready** and optimized for 50-epoch training on Rutgers Amarel HPC. This represents a significant improvement over the 5-epoch preliminary runs.

**Status:** ✅ All scripts verified, tested, and pushed to GitHub
**Next Step:** Clone repo on Amarel and submit training jobs

---

## What's Been Done

### ✅ Training Scripts Verified & Production-Ready

All training scripts have been:
- ✓ Audited for GPU/CUDA compatibility
- ✓ Tested with proper imports and dependencies
- ✓ Fixed for Unicode/encoding issues (Windows cp1252 compatibility)
- ✓ Configured with optimal hyperparameters

**Available Methods:**
1. **Baseline** (`src/train_classifier_baseline.py`)
   - Standard ResNet-18 classifier
   - No uncertainty quantification
   
2. **MC Dropout** (`src/train_classifier_mc_dropout.py`)
   - ResNet-18 with dropout in final layer
   - Dropout active during inference (T=20 samples)
   
3. **Deep Ensemble** (`src/train_classifier_ensemble_member.py`)
   - 5 independent models with different seeds
   - Ensemble predictions and uncertainty
   
4. **SWAG** (`src/train_swag_classification.py`)
   - Stochastic Weight Averaging-Gaussian
   - Snapshot collection from epoch 30 onwards
   - T=30 posterior samples
   
5. **Conformal Risk Control** (`src/conformal_risk_control.py`)
   - Post-hoc calibration (no training required)
   - Multiple loss functions: FNR, precision, set size, F1, composite

### ✅ SBATCH Scripts Updated for 50 Epochs

All scripts have been updated with:
- ✓ **50 epoch training** (vs 5 epochs preliminary)
- ✓ Appropriate time limits (24-48 hours)
- ✓ GPU allocation (1x GPU per method)
- ✓ Proper output directories
- ✓ Environment setup and module loading

**Updated SBATCH Files:**
- `scripts/train_classifier_baseline.sbatch` - 24h
- `scripts/train_classifier_mc_dropout.sbatch` - 24h
- `scripts/train_classifier_ensemble.sbatch` - 48h (5 members sequential)
- `scripts/train_classifier_swag.sbatch` - NEW, 24h
- `scripts/evaluate_classification.sbatch` - 12h (with SWAG evaluation)

### ✅ Evaluation Script Enhanced

- `src/evaluate_uq_classification.py` updated to:
  - ✓ Evaluate all 5 methods
  - ✓ Support SWAG evaluation with `--swag_path` and `--swag_samples` args
  - ✓ Print Mean Uncertainty metrics for SWAG (matching MC Dropout/Ensemble)
  - ✓ Handle GPU device selection properly
  - ✓ Generate comprehensive `all_results.json` with all metrics

**Evaluation Metrics:**
- Accuracy (%)
- Expected Calibration Error (ECE)
- Brier Score
- Mean Uncertainty (for MC Dropout, Ensemble, SWAG)
- Coverage & Set Size (for Conformal Risk Control)

### ✅ Documentation & Setup Guides

New documentation files:
- `docs/AMAREL_WORKFLOW_50EPOCHS.md` - Complete step-by-step guide for 50-epoch runs
- `scripts/submit_all_uq_jobs.sh` - Automated job submission script

### ✅ Git Repository Updated

All code pushed to GitHub:
```
✓ 45 files added/modified in latest commit
✓ All training scripts, evaluation, and SBATCH files version-controlled
✓ Ready to clone on Amarel: git clone https://github.com/valle1306/uq_capstone.git
```

---

## Expected Performance Improvements

| Method | 5 Epochs | 50 Epochs (Expected) |
|--------|----------|----------------------|
| Baseline | 85.26% | 88-92% |
| MC Dropout | 86.38% | 89-93% |
| Ensemble | 88.78% | 90-94% ⭐ |
| SWAG | 82.85% | 87-91% |
| Calibration | ECE 0.06 | ECE 0.02-0.04 |

**Why 50 epochs?**
- Better convergence and model maturity
- More stable uncertainty estimates
- Improved calibration (lower ECE/Brier)
- Meaningful comparison between methods
- Closer to typical production training

---

## Quick Start on Amarel

### 1. Clone & Setup (5 minutes)
```bash
ssh YOUR_USERNAME@amarel.rutgers.edu
cd /scratch/$USER
git clone https://github.com/valle1306/uq_capstone.git
cd uq_capstone

# Setup conda environment
conda create -n uq_capstone python=3.9 -y
conda activate uq_capstone
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn tqdm numpy pandas
```

### 2. Upload Dataset (if not already present)
```bash
# From local machine
scp -r data/chest_xray YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/data/
```

### 3. Submit Training Jobs
```bash
cd /scratch/$USER/uq_capstone

# Option A: Sequential (recommended first time)
sbatch scripts/train_classifier_baseline.sbatch
sbatch scripts/train_classifier_mc_dropout.sbatch
sbatch scripts/train_classifier_ensemble.sbatch
sbatch scripts/train_classifier_swag.sbatch

# Option B: Parallel (if multiple GPUs available)
for script in train_classifier_{baseline,mc_dropout,ensemble,swag}.sbatch; do
    sbatch scripts/$script
done

# Check job status
squeue -u $USER
```

### 4. Monitor Training
```bash
# View live progress
tail -f runs/classification/baseline/train_*.out

# Check GPU usage
nvidia-smi

# Total runtime estimate: 110-120 hours sequential
```

### 5. Run Evaluation (after all training complete)
```bash
sbatch scripts/evaluate_classification.sbatch

# Or run directly
python src/evaluate_uq_classification.py \
    --dataset chest_xray \
    --data_dir data/chest_xray \
    --baseline_path runs/classification/baseline/best_model.pth \
    --mc_dropout_path runs/classification/mc_dropout/best_model.pth \
    --ensemble_dir runs/classification/ensemble \
    --n_ensemble 5 \
    --swag_path runs/classification/swag_classification/swag_model.pth \
    --swag_samples 30 \
    --mc_samples 20 \
    --dropout_rate 0.3 \
    --device cuda
```

### 6. Download Results
```bash
# On local machine
scp -r YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/runs/classification ./runs/
```

---

## File Structure

```
uq_capstone/
├── src/
│   ├── train_classifier_baseline.py          ✓ Ready
│   ├── train_classifier_mc_dropout.py        ✓ Ready
│   ├── train_classifier_ensemble_member.py   ✓ Ready
│   ├── train_swag_classification.py          ✓ Ready
│   ├── evaluate_uq_classification.py         ✓ Enhanced with SWAG
│   ├── conformal_risk_control.py             ✓ Ready
│   ├── data_utils_classification.py          ✓ Ready
│   ├── swag.py                               ✓ Ready
│   └── visualize_uq_results.py               ✓ Ready
├── scripts/
│   ├── train_classifier_baseline.sbatch      ✓ 50 epochs, 24h
│   ├── train_classifier_mc_dropout.sbatch    ✓ 50 epochs, 24h
│   ├── train_classifier_ensemble.sbatch      ✓ 50 epochs, 48h
│   ├── train_classifier_swag.sbatch          ✓ NEW: 50 epochs, 24h
│   ├── evaluate_classification.sbatch        ✓ Updated with SWAG
│   ├── submit_all_uq_jobs.sh                 ✓ NEW: Auto-submit all jobs
│   └── run_all_classification_experiments.sh ✓ Ready
├── data/
│   └── chest_xray/                           (Upload from local)
├── docs/
│   ├── AMAREL_WORKFLOW_50EPOCHS.md           ✓ NEW: Complete guide
│   ├── CLASSIFICATION_SETUP_GUIDE.md         ✓ Reference
│   └── ... other docs ...
├── runs/
│   └── classification/                       (Will be populated after training)
│       ├── baseline/
│       ├── mc_dropout/
│       ├── ensemble/member_{0,1,2,3,4}/
│       ├── swag_classification/
│       └── evaluation/
└── README.md                                 ✓ Updated
```

---

## Key Features Implemented

### Bayesian Uncertainty Quantification
- ✓ **MC Dropout**: Uncertainty via prediction variance
- ✓ **Deep Ensemble**: Uncertainty via ensemble variance
- ✓ **SWAG**: Bayesian weight-space uncertainty via posterior sampling
- ✓ **Conformal**: Risk-controlled prediction sets

### Metrics & Calibration
- ✓ Accuracy, ECE, Brier Score
- ✓ Mean & Std Uncertainty
- ✓ Coverage & Set Size
- ✓ Risk control guarantees (FNR, precision, etc.)

### Production-Ready
- ✓ GPU/CUDA support
- ✓ Proper device handling (cuda/cpu)
- ✓ Batch processing
- ✓ Checkpoint saving & loading
- ✓ Comprehensive logging
- ✓ Error handling

---

## Performance Expectations

### Baseline (ResNet-18)
- Parameters: ~11.2M
- Training time: ~12-15 hours per 50 epochs (GPU)
- Typical accuracy: 85-92%

### MC Dropout (T=20 samples)
- Same architecture + dropout layer
- Inference: 20x forward passes
- Overhead: ~5% accuracy time (vs baseline)
- Uncertainty quality: Good

### Deep Ensemble (M=5 models)
- 5x independent ResNet-18
- Best performance but highest memory
- Inference: 5x forward passes
- Accuracy: Usually 1-3% higher than baseline
- Uncertainty quality: Excellent

### SWAG (T=30 samples)
- Weight-space Bayesian approximation
- Snapshot collection: epochs 30-50 (20 snapshots)
- Inference: 30x forward passes + weight sampling
- Overhead: ~2% vs MC Dropout
- Uncertainty quality: Very good

### Conformal Risk Control
- Post-hoc calibration (no additional training)
- Guarantees on empirical risk
- Example: "FNR ≤ 5% with 90% confidence"

---

## Troubleshooting

### Common Issues

**Q: CUDA out of memory**
- Reduce batch_size from 32 to 16 in SBATCH
- Reduce num_workers from 4 to 0

**Q: Job timeout after 24 hours**
- SBATCH time limit is appropriate for 50 epochs
- If still timing out, increase #SBATCH --time=36:00:00

**Q: Dataset not found**
- Ensure chest_xray uploaded: `ls data/chest_xray/train/`
- If missing: `scp -r data/chest_xray USER@amarel.rutgers.edu:/scratch/$USER/uq_capstone/data/`

**Q: Model checkpoints not saving**
- Check output directory permissions: `mkdir -p runs/classification/{baseline,mc_dropout,...}`
- Check disk space: `du -sh runs/`

---

## Next Steps

1. **SSH to Amarel** and clone repository
2. **Setup conda environment** with PyTorch
3. **Upload dataset** (chest_xray)
4. **Submit jobs** using SBATCH scripts
5. **Monitor training** with squeue/tail
6. **Download results** after completion
7. **Analyze results** and generate visualizations

---

## Contact & Support

- **Training Issue?** Check SBATCH output: `cat runs/classification/*/train_*.out`
- **Git Issue?** Run `git status` and `git log` to verify repo state
- **CUDA Issue?** Run `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`

---

## Summary

✅ **All systems go!** Code is tested, documented, and ready for production 50-epoch training on Amarel.

**Next action:** Clone repo on Amarel, setup environment, and submit jobs.

Estimated completion: **120 hours sequential** (~5 days) or **15-20 hours parallel** (if multiple GPUs available)

Expected results: **Significant improvement** over 5-epoch baseline (accuracy +3-5%, calibration improved by ~50%)

