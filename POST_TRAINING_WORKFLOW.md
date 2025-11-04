# Post-Training Workflow - Run on Amarel

**Status:** Training jobs complete ✅

## Step 1: Verify Models Exist

```bash
ssh hpl14@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone

# Check all models
ls -lh runs/classification/mc_dropout/best_model.pth
ls -lh runs/classification/swag_classification/swag_model.pth
ls -lh runs/classification/baseline/best_model.pth
ls -lh runs/classification/ensemble/member_*/best_model.pth
```

## Step 2: Run Metrics + Visualizations on Amarel (ONE COMMAND)

```bash
# Submit job to run comprehensive metrics AND visualizations
sbatch scripts/eval_and_visualize_on_amarel.sbatch

# Get job ID from output, then monitor:
squeue -u hpl14

# Watch progress in real-time:
tail -f logs/eval_visualize_comprehensive_<job_id>.out
```

**What This Does:**
- ✅ Evaluates all 5 UQ methods (Baseline, MC Dropout, Ensemble, SWAG, CRC)
- ✅ Computes calibration metrics (ECE, MCE, Brier)
- ✅ Computes uncertainty metrics
- ✅ Generates all visualizations (plots, ROC curves, etc.)
- ✅ Creates summary report
- ⏱ Takes ~30-60 minutes

## Step 3: Review Results on Amarel

```bash
# View summary report
cat runs/classification/metrics/EVALUATION_REPORT.txt

# Check accuracies (should show ~90% for MC Dropout and SWAG)
cat runs/classification/metrics/comprehensive_metrics.json | python -m json.tool | head -100

# Quick accuracy check
python -c "import json; r = json.load(open('runs/classification/metrics/comprehensive_metrics.json')); print(f'MC Dropout: {r[\"mc_dropout\"][\"accuracy\"]:.2%}'); print(f'SWAG: {r[\"swag\"][\"accuracy\"]:.2%}')"

# List all output files
ls -lh runs/classification/metrics/
```

## Step 4: Decision Point - Do Results Look Good?

### ✅ YES - Results are ~90% accuracy:
```bash
# Pull results to local machine (on Windows PowerShell)
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics ./runs/
```

### ❌ NO - Something is wrong:
1. Check logs for errors:
   ```bash
   tail -100 logs/eval_visualize_comprehensive_<job_id>.err
   ```
2. See troubleshooting section below
3. Rerun evaluation:
   ```bash
   sbatch scripts/eval_and_visualize_on_amarel.sbatch
   ```

## Step 5: Pull Results (If Good)

```powershell
# On Windows PowerShell

# Pull all results
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics ./runs/

# Verify locally
Get-ChildItem runs/classification/metrics/ -Filter *.json, *.csv, *.png, *.txt
```

## Troubleshooting

### "Model not found" Error
```bash
# Check if retraining actually completed
ls -lh runs/classification/mc_dropout/best_model.pth
ls -lh runs/classification/swag_classification/swag_model.pth

# If missing, check retrain logs
tail logs/retrain_mc_dropout_*.err
tail logs/retrain_swag_*.err

# Resubmit if needed
sbatch scripts/retrain_mc_dropout.sbatch
sbatch scripts/retrain_swag.sbatch
```

### "CUDA out of memory" Error
- Normal for comprehensive evaluation with CRC
- Already set to 32GB, should be sufficient
- Check scratch space: `du -sh /scratch/$USER/uq_capstone`

### "Evaluation FAILED" Error
```bash
# Check error log
cat logs/eval_visualize_comprehensive_<job_id>.err

# Verify comprehensive_metrics.py exists
ls -lh src/comprehensive_metrics.py

# Try running locally for debugging
python src/comprehensive_metrics.py --baseline_path runs/classification/baseline/best_model.pth --device cpu
```

### Results Show Low Accuracy (<85%)
- MC Dropout still low: Retrain may not have completed properly
- SWAG still low: Check if baseline checkpoint was used
- Verify by checking training logs:
  ```bash
  tail logs/retrain_mc_dropout_*.out
  tail logs/retrain_swag_*.out
  ```

## Key Differences from Original Workflow

| Step | Original | **NEW** |
|------|----------|--------|
| Metrics | Pull to Windows | **Run on Amarel** ✨ |
| Visualizations | Generate locally | **Generate on Amarel** ✨ |
| Decision | Pull everything, then review | **Review on Amarel first, then pull** ✨ |
| Time | ~45 min (local compute) | **~30-60 min (GPU compute)** |

## Commands Quick Reference

```bash
# SSH to Amarel and navigate
ssh hpl14@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone

# Check models exist
ls -lh runs/classification/{baseline,mc_dropout,swag_classification}/best_model.pth

# Submit evaluation + visualization job
sbatch scripts/eval_and_visualize_on_amarel.sbatch

# Monitor (repeat every minute)
squeue -u hpl14

# View progress
tail -f logs/eval_visualize_comprehensive_<job_id>.out

# Check accuracies when done
cat runs/classification/metrics/comprehensive_metrics.json | python -m json.tool | head -50

# View summary report
cat runs/classification/metrics/EVALUATION_REPORT.txt

# If good, pull to Windows (from Windows PowerShell)
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics ./runs/
```

---

**Next:** Run Step 1 above! 🚀
