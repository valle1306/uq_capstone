# Retraining Workflow on Amarel

## 🚀 Quick Commands

### Step 1: SSH to Amarel
```bash
ssh hpl14@amarel.rutgers.edu
```

### Step 2: Update Local Repository
```bash
cd /scratch/$USER/uq_capstone
git fetch origin main
git reset --hard FETCH_HEAD
```

### Step 3: Backup Old Models
```bash
# Save old models in case we need to compare
mv runs/classification/mc_dropout runs/classification/mc_dropout_old
mv runs/classification/swag_classification runs/classification/swag_classification_old
mkdir -p runs/classification/mc_dropout runs/classification/swag_classification
```

### Step 4: Create Logs Directory
```bash
mkdir -p logs
```

### Step 5: Submit Retraining Jobs
```bash
# Submit MC Dropout retrain (24h GPU)
sbatch scripts/retrain_mc_dropout.sbatch

# Submit SWAG retrain (24h GPU)
sbatch scripts/retrain_swag.sbatch
```

### Step 6: Monitor Jobs
```bash
# Check job status
squeue -u hpl14

# View specific job output (optional - while running)
tail -f logs/retrain_mc_dropout_<job_id>.out
tail -f logs/retrain_swag_<job_id>.out
```

## ⏰ Expected Timeline
- **MC Dropout retrain**: ~24 hours on GPU
- **SWAG retrain**: ~24 hours on GPU
- **Total**: Can run in parallel (~24 hours wall clock)

## 📊 After Retraining Completes

### Step 7: Re-evaluate Metrics
```bash
# Submit comprehensive evaluation job
sbatch scripts/evaluate_classification_comprehensive.sbatch

# Or run locally after pulling results
```

### Step 8: Pull Results Locally (Windows PowerShell)
```powershell
# From Windows, pull all results
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics ./runs/

# Or pull only new models
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/mc_dropout/best_model.pth ./runs/classification/mc_dropout/
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/swag_classification/swag_model.pth ./runs/classification/swag_classification/
```

## ✅ Verification Checklist

After retraining completes:
- [ ] MC Dropout achieves ≥90% accuracy
- [ ] SWAG achieves ≥90% accuracy
- [ ] Models saved to correct directories
- [ ] History files generated
- [ ] Comprehensive metrics run successfully
- [ ] Visualizations generated

## 🔧 Troubleshooting

### Check Job Failed
```bash
# View error log
cat logs/retrain_mc_dropout_<job_id>.err
cat logs/retrain_swag_<job_id>.err
```

### Re-run Failed Job
```bash
# If MC Dropout failed:
sbatch scripts/retrain_mc_dropout.sbatch

# If SWAG failed:
sbatch scripts/retrain_swag.sbatch
```

### Cancel Job
```bash
scancel <job_id>
```

## 📝 What Gets Saved

### MC Dropout Outputs
- `runs/classification/mc_dropout/best_model.pth` - Best model weights
- `runs/classification/mc_dropout/history.json` - Training history
- `runs/classification/mc_dropout/config.json` - Training config

### SWAG Outputs
- `runs/classification/swag_classification/swag_model.pth` - SWAG model with snapshots
- `runs/classification/swag_classification/best_base_model.pth` - Best base model before SWAG
- `runs/classification/swag_classification/history.json` - Training history
- `runs/classification/swag_classification/config.json` - Training config

## 🎯 Next Steps After Verification

1. Run comprehensive metrics evaluation
2. Compare results with old models
3. Generate visualizations
4. Analyze Conformal Risk Control
5. Prepare presentation materials
