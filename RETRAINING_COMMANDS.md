# Amarel Retraining - Full Command Sequence

## Copy-Paste Ready Commands

### 1️⃣ SSH + Update + Backup (Run All at Once)
```bash
ssh hpl14@amarel.rutgers.edu << 'EOF'
cd /scratch/$USER/uq_capstone
git fetch origin main && git reset --hard FETCH_HEAD
mv runs/classification/mc_dropout runs/classification/mc_dropout_old
mv runs/classification/swag_classification runs/classification/swag_classification_old
mkdir -p runs/classification/mc_dropout runs/classification/swag_classification logs
sbatch scripts/retrain_mc_dropout.sbatch
sbatch scripts/retrain_swag.sbatch
squeue -u hpl14
EOF
```

### 2️⃣ Or Execute Line by Line

**SSH to Amarel:**
```bash
ssh hpl14@amarel.rutgers.edu
```

**Once on Amarel, run these commands sequentially:**

```bash
# Navigate to project
cd /scratch/$USER/uq_capstone

# Pull latest code
git fetch origin main
git reset --hard FETCH_HEAD

# Backup old models
mv runs/classification/mc_dropout runs/classification/mc_dropout_old
mv runs/classification/swag_classification runs/classification/swag_classification_old

# Create directories
mkdir -p runs/classification/mc_dropout runs/classification/swag_classification logs

# Submit retraining jobs
sbatch scripts/retrain_mc_dropout.sbatch
sbatch scripts/retrain_swag.sbatch

# Check status
squeue -u hpl14
```

### 3️⃣ Monitor Jobs (While Running)

```bash
# Check all your jobs
squeue -u hpl14

# Watch updates (every 2 seconds)
watch -n 2 'squeue -u hpl14'

# View output while running (replace <job_id> with actual ID)
tail -f logs/retrain_mc_dropout_<job_id>.out
tail -f logs/retrain_swag_<job_id>.out
```

### 4️⃣ After Training Completes (Estimate 24-48 hours)

**On Amarel:**
```bash
# Check that models exist
ls -lh runs/classification/mc_dropout/best_model.pth
ls -lh runs/classification/swag_classification/swag_model.pth

# Run comprehensive evaluation + visualizations ON AMAREL
cd /scratch/$USER/uq_capstone
sbatch scripts/eval_and_visualize_on_amarel.sbatch

# Monitor evaluation job
squeue -u hpl14

# Check progress (while running)
tail -f logs/eval_visualize_comprehensive_<job_id>.out

# View results when done
cat runs/classification/metrics/comprehensive_metrics.json | python -m json.tool | head -100
```

### 5️⃣ Review Results on Amarel (Before Pulling)

```bash
# View detailed results
cat runs/classification/metrics/EVALUATION_REPORT.txt

# Check accuracies (should be ~90% for MC Dropout and SWAG)
python -c "import json; r = json.load(open('runs/classification/metrics/comprehensive_metrics.json')); print(f'MC Dropout Acc: {r[\"mc_dropout\"][\"accuracy\"]:.2%}'); print(f'SWAG Acc: {r[\"swag\"][\"accuracy\"]:.2%}')"

# List all generated files
ls -lh runs/classification/metrics/
```

### 6️⃣ If Results Look Good, Pull to Windows (PowerShell)

```powershell
# Pull all results and visualizations
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics ./runs/

# Or pull specific files
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics/comprehensive_metrics.json ./runs/classification/metrics/
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics/*.png ./runs/classification/metrics/
```

### 7️⃣ Local Analysis (Windows PowerShell - Optional)

```powershell
cd c:\Users\lpnhu\Downloads\uq_capstone

# View results locally
python -c "import json; r = json.load(open('runs/classification/metrics/comprehensive_metrics.json')); print(json.dumps(r, indent=2))" | head -100

# Generate additional analysis
python analysis/generate_uq_report.py
```

## Key Parameters

| Parameter | MC Dropout | SWAG |
|-----------|-----------|------|
| Initialization | Baseline weights | Baseline weights |
| Epochs | 50 | 50 |
| Learning Rate | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Batch Size | 32 | 32 |
| Dropout Rate | 0.2 | N/A |
| Snapshots | N/A | Epochs 30-50 (20 total) |
| Expected Accuracy | ≥90% | ≥90% |
| GPU Time | ~24h | ~24h |

## File Structure After Retraining

```
runs/classification/
├── baseline/
│   └── best_model.pth         ✓ Reference (91.67%)
├── mc_dropout/
│   ├── best_model.pth         ← NEW (retrained)
│   ├── history.json
│   └── config.json
├── mc_dropout_old/            ← OLD backup
│   └── best_model.pth         (63.3%)
├── ensemble/
│   ├── member_0/
│   ├── member_1/
│   ├── member_2/
│   ├── member_3/
│   └── member_4/              ✓ No change (91.67%)
├── swag_classification/
│   ├── swag_model.pth         ← NEW (retrained)
│   ├── best_base_model.pth
│   ├── history.json
│   └── config.json
├── swag_classification_old/   ← OLD backup
│   └── swag_model.pth         (79.3%)
└── metrics/
    └── comprehensive_results.json (← generated after eval)
```

## Expected Results After Retraining

✅ **MC Dropout**
- Accuracy: ~90% (was 63.3%)
- Stochastic uncertainty: Proper calibration
- MC sampling: 20 forward passes with dropout enabled

✅ **SWAG**
- Accuracy: ~90% (was 79.3%)
- Posterior approximation: Proper Bayesian treatment
- Scale: 0.5 (recommended value)

✅ **Other Methods (No Change)**
- Baseline: 91.67% (reference)
- Ensemble: 91.67% (5 members)
- CRC: Computed post-hoc from calibration set

## Troubleshooting

### Jobs Not Starting?
```bash
# Check node availability
sinfo
# Check if conda env exists
conda env list
```

### Out of Memory?
- Already set to 32GB, should be sufficient
- Check /scratch space: `du -sh /scratch/$USER/uq_capstone`

### Training Too Slow?
- Normal pace: ~8-12 minutes per epoch on V100 GPU
- Total: ~400-600 minutes (~7-10 hours) per model

### Job Killed After 24h?
- Time limit is set to 24h; jobs might need more time
- If approaching timeout, increase `--time=24:00:00` to `--time=30:00:00` in SBATCH files

## Status Indicators

🟢 **Ready to Go:**
- Retrain scripts committed and pushed to GitHub
- SBATCH scripts created on Amarel
- Baseline checkpoint available at runs/classification/baseline/best_model.pth
- Conda environment uq_capstone configured

🟡 **Waiting For:**
- Your command to SSH and submit jobs (Step 2️⃣)
- ~24-48 hours for training to complete
- Pull results back to local machine (Step 4️⃣)
- Re-run metrics evaluation (Step 5️⃣)

🔴 **Not Yet Started:**
- Retraining jobs
- Metrics evaluation
- Visualization generation

---

**Ready to proceed? Run the commands in section 2️⃣ on Amarel!**
