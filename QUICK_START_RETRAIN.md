# 🚀 RETRAINING QUICK START

## What Changed?

Originally trained 50 epochs:
- MC Dropout: 63.3% ❌ (should be ~90%)
- SWAG: 79.3% ❌ (should be ~90%)

Root cause: Both models trained from scratch, converged to suboptimal local minima.

## The Fix

✅ **Retrain from baseline** - Use proven 91.67% baseline checkpoint as initialization
✅ **Fine-tune properly** - Transfer learning with low learning rate (1e-4)
✅ **Correct hyperparameters**:
- MC Dropout: dropout_rate=0.2 (was 0.3)
- SWAG: Collect snapshots from epoch 30 onwards (20 total)

## Your Action Items

### On Amarel (SSH Terminal)

```bash
# 1. Connect
ssh hpl14@amarel.rutgers.edu

# 2. Update code
cd /scratch/$USER/uq_capstone
git fetch origin main && git reset --hard FETCH_HEAD

# 3. Backup old models
mv runs/classification/mc_dropout runs/classification/mc_dropout_old
mv runs/classification/swag_classification runs/classification/swag_classification_old
mkdir -p runs/classification/{mc_dropout,swag_classification} logs

# 4. Submit jobs
sbatch scripts/retrain_mc_dropout.sbatch
sbatch scripts/retrain_swag.sbatch

# 5. Check status (repeat every few minutes)
squeue -u hpl14
```

### Wait ~24-48 hours

Jobs will run in parallel. Monitor with:
```bash
squeue -u hpl14
```

### After Training Completes

**On Amarel:**
```bash
# Verify models exist
ls -lh runs/classification/mc_dropout/best_model.pth
ls -lh runs/classification/swag_classification/swag_model.pth
```

**On Windows (PowerShell):**
```powershell
# Pull new models
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/mc_dropout/best_model.pth ./runs/classification/mc_dropout/
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/swag_classification/swag_model.pth ./runs/classification/swag_classification/

# Re-run metrics evaluation
cd c:\Users\lpnhu\Downloads\uq_capstone
python src/comprehensive_metrics.py

# Generate plots
python analysis/visualize_metrics.py
```

## Expected Results

| Model | Before | After |
|-------|--------|-------|
| MC Dropout | 63.3% ❌ | ~90% ✅ |
| SWAG | 79.3% ❌ | ~90% ✅ |
| Ensemble | 91.67% ✓ | 91.67% ✓ |

## Key Files

- `src/retrain_mc_dropout.py` - MC Dropout retraining script
- `src/retrain_swag.py` - SWAG retraining script
- `scripts/retrain_mc_dropout.sbatch` - SLURM job for MC Dropout
- `scripts/retrain_swag.sbatch` - SLURM job for SWAG
- `RETRAINING_COMMANDS.md` - Full command reference
- `RETRAINING_STATUS.md` - Detailed status
- `docs/RETRAINING_WORKFLOW.md` - Complete workflow guide

## Git Status

✅ All code committed and pushed to GitHub (commit f892928)
- Branch: main
- Repo: https://github.com/valle1306/uq_capstone

## Timeline

1. Now: Retrain scripts ready (✅ Done)
2. You: SSH + Submit jobs (⏳ Your turn)
3. Wait: ~24-48 hours for training
4. You: Pull results locally
5. Me: Re-analyze metrics and generate final report

Ready? SSH to Amarel and run the commands above!
