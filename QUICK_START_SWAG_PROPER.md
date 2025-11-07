# SWAG Proper Retraining - Quick Start Guide
**Date:** November 7, 2025

## What Changed

We discovered our SWAG implementation did NOT follow the Maddox et al. 2019 paper exactly:

### Previous Issues ❌
- Used Adam optimizer (paper uses SGD)
- **NO weight decay** (paper uses 1e-4 for L2 regularization)
- Simple cosine annealing (paper uses SWALR for SWA phase)
- No batch norm update (paper requires this)
- Initialized from baseline (paper trains from scratch)

### Fixed Implementation ✅
- **SGD with momentum=0.9** (as in paper)
- **Weight decay=1e-4** (L2 regularization - prevents overfitting!)
- **SWALR scheduler** for proper SWA phase
- **Batch norm update** after collecting snapshots
- **Train from scratch** (random initialization)

**Expected Result:** SWAG accuracy should improve from 83% → ~90%+

---

## Commands to Run

### 1. On Your Local Machine (Windows)

```powershell
# Commit and push the new scripts
cd C:\Users\lpnhu\Downloads\uq_capstone

git add src/retrain_swag_proper.py scripts/retrain_swag_proper.sbatch QUICK_START_SWAG_PROPER.md SWAG_RETRAIN_DECISION.md
git commit -m "Fix SWAG implementation to follow Maddox et al. 2019 paper exactly (SGD, weight decay, SWALR, batch norm update)"
git push origin main
```

### 2. On Amarel (SSH)

```bash
# SSH to Amarel
ssh hpl14@amarel.rutgers.edu

# Navigate to project
cd /scratch/$USER/uq_capstone

# Pull latest code
git pull origin main

# Backup old SWAG model (optional)
mv runs/classification/swag_classification runs/classification/swag_classification_old_adam
mkdir -p runs/classification/swag_classification

# Submit proper SWAG retraining job
sbatch scripts/retrain_swag_proper.sbatch

# Check job status
squeue -u hpl14

# Monitor progress (replace JOB_ID with actual job ID)
tail -f logs/swag_proper_JOBID.out
```

### 3. After Training Completes (~24 hours)

```bash
# On Amarel - Check results
cd /scratch/$USER/uq_capstone

# View training summary
cat runs/classification/swag_classification/config.json

# Check final accuracy
python -c "
import json
with open('runs/classification/swag_classification/training_history.json') as f:
    hist = json.load(f)
print(f'Final validation accuracy: {hist[\"val_acc\"][-1]:.2f}%')
print(f'Final test accuracy: {hist[\"test_acc\"][-1]:.2f}%')
"

# Re-run comprehensive evaluation
sbatch scripts/eval_and_visualize_on_amarel.sbatch
```

### 4. Download New Results

```powershell
# On Windows - download updated results
cd C:\Users\lpnhu\Downloads\uq_capstone

# Run download script
.\download_results.ps1

# Or manually download specific files
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics/metrics_summary.csv runs\classification\metrics\
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics/*.png runs\classification\metrics\
```

---

## Expected Results

### Before (Adam, no weight decay)
```
SWAG: 83.17% accuracy, ECE=0.1519
Issue: Overfitting (99.62% val, 85.58% test)
```

### After (SGD + weight decay + SWALR)
```
SWAG: ~90% accuracy, ECE=~0.05-0.10
Expected: Similar to baseline/ensemble
```

---

## Timeline

| Time | Action | Who |
|------|--------|-----|
| Now | Commit & push scripts | You |
| +5 min | SSH to Amarel, pull, submit job | You |
| +1-2 hours | Check job started successfully | You |
| +24 hours | Training completes | Amarel |
| +24h 15min | Rerun evaluation | You |
| +24h 30min | Download results | You |
| +24h 45min | Update presentation | You |

---

## Quick Commands Summary

```bash
# On Windows
git add src/retrain_swag_proper.py scripts/retrain_swag_proper.sbatch QUICK_START_SWAG_PROPER.md SWAG_RETRAIN_DECISION.md
git commit -m "Fix SWAG: SGD + weight decay + SWALR + batch norm update"
git push origin main

# On Amarel
ssh hpl14@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone
git pull origin main
mv runs/classification/swag_classification runs/classification/swag_classification_old
mkdir -p runs/classification/swag_classification
sbatch scripts/retrain_swag_proper.sbatch
squeue -u hpl14

# Check job (replace JOBID)
tail -f logs/swag_proper_JOBID.out

# After 24 hours
sbatch scripts/eval_and_visualize_on_amarel.sbatch

# Download results (on Windows)
.\download_results.ps1
```
