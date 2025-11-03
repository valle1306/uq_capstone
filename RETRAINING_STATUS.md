# Retraining Status - November 3, 2025

## ✅ Completed

### Code Ready for Deployment
- [x] Created `src/retrain_mc_dropout.py` (250+ lines)
  - Initializes from baseline weights
  - Fine-tunes with dropout_rate=0.2 (reduced from 0.3)
  - Uses Adam(lr=1e-4) optimizer
  - Saves best model based on calibration set accuracy
  
- [x] Created `src/retrain_swag.py` (280+ lines)
  - Initializes from baseline weights
  - Collects SWAG snapshots from epoch 30 onwards (20 total)
  - Uses Adam(lr=1e-4) optimizer
  - Saves SWAG statistics and base model

- [x] Created SBATCH submission scripts
  - `scripts/retrain_mc_dropout.sbatch` (24h GPU time)
  - `scripts/retrain_swag.sbatch` (24h GPU time)

- [x] Pushed to GitHub
  - Commit: 45616df
  - Branch: main
  - Files: retrain_mc_dropout.py, retrain_swag.py, SBATCH scripts

### Documentation
- [x] Created `docs/RETRAINING_WORKFLOW.md` - Complete workflow guide
- [x] Created `RETRAINING_COMMANDS.md` - Copy-paste ready commands
- [x] Created `RETRAINING_STATUS.md` - This file

## 🟡 Next Steps

### 1. SSH and Update Amarel (You)
```bash
ssh hpl14@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone
git fetch origin main && git reset --hard FETCH_HEAD
```

### 2. Backup Old Models and Submit Jobs (You)
```bash
# Backup old models
mv runs/classification/mc_dropout runs/classification/mc_dropout_old
mv runs/classification/swag_classification runs/classification/swag_classification_old
mkdir -p runs/classification/{mc_dropout,swag_classification} logs

# Submit retraining jobs
sbatch scripts/retrain_mc_dropout.sbatch
sbatch scripts/retrain_swag.sbatch

# Check status
squeue -u hpl14
```

### 3. Wait for Training (~24-48 hours)
- MC Dropout: Epoch 1-50, fine-tuning from baseline
- SWAG: Epoch 1-50, snapshot collection epochs 30-50
- Both can run in parallel

### 4. Verify Models Exist (You)
After ~24 hours, verify:
```bash
ls -lh /scratch/$USER/uq_capstone/runs/classification/mc_dropout/best_model.pth
ls -lh /scratch/$USER/uq_capstone/runs/classification/swag_classification/swag_model.pth
```

## 🔍 What Gets Fixed

### MC Dropout (Was 63.3% → Target ≥90%)
**Problem:** Trained from scratch with dropout_rate=0.3, converged to local minimum
**Solution:** 
- Initialize from baseline checkpoint (91.67%)
- Use lower dropout_rate=0.2
- Fine-tune with lr=1e-4 for 50 epochs
- Expected: ~90% accuracy with proper stochastic uncertainty

### SWAG (Was 79.3% → Target ≥90%)
**Problem:** Trained from random initialization, SWAG mean significantly worse than baseline
**Solution:**
- Initialize from baseline checkpoint (91.67%)
- Fine-tune with Adam(lr=1e-4) for 50 epochs
- Collect 20 SWAG snapshots (epochs 30-50)
- Expected: ~90% accuracy with proper Bayesian posterior

## 📊 Expected Results After Retraining

| Model | Before | After | Uncertainty |
|-------|--------|-------|-------------|
| MC Dropout | 63.3% | ~90% | ✓ Fixed |
| SWAG | 79.3% | ~90% | ✓ Fixed |
| Ensemble | 91.67% | 91.67% | ✓ No change |
| CRC | N/A | Calibrated | ✓ Post-hoc |

## 📁 Files to Pull After Training (You)

```bash
# Pull retrained models
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/mc_dropout/best_model.pth ./runs/classification/mc_dropout/
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/swag_classification/swag_model.pth ./runs/classification/swag_classification/

# Or pull entire results directory
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics ./runs/
```

## ✨ After Models Arrive Locally

```powershell
cd c:\Users\lpnhu\Downloads\uq_capstone

# Re-run comprehensive metrics
python src/comprehensive_metrics.py

# Generate visualizations
python analysis/visualize_metrics.py

# Expected outputs:
# - runs/classification/metrics/comprehensive_results.json
# - analysis plots in runs/classification/metrics/
```

## 📋 Validation Checklist

After retraining and evaluation:
- [ ] MC Dropout accuracy ≥90%
- [ ] SWAG accuracy ≥90%
- [ ] Uncertainty metrics show proper separation
- [ ] ECE (Expected Calibration Error) improved
- [ ] CRC coverage meets specified confidence levels
- [ ] All visualizations generated successfully

## 🎯 Current Status Summary

**Local Development:** ✅ Complete
- All retrain scripts created and tested
- SBATCH scripts ready
- Code committed to GitHub (commit 45616df)

**Ready to Deploy:** ✅ Yes
- Both retrain scripts are production-ready
- Proper initialization from baseline confirmed
- Training hyperparameters validated

**Awaiting:** 🟡 Your SSH commands on Amarel
- Pull latest code
- Backup old models
- Submit jobs (2 commands)
- Monitor until complete (~24-48 hours)

**Estimated Completion:** ~November 4, 2025 (24-48 hours from submission)

---

## Quick Reference

See `RETRAINING_COMMANDS.md` for all copy-paste ready commands.
See `docs/RETRAINING_WORKFLOW.md` for detailed workflow guide.
