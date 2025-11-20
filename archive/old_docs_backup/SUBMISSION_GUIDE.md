# Quick Submission Guide: Experiments #3 and #4

## Current Status Check

Before submitting new experiments, check the status of currently running jobs:

```bash
# SSH to Amarel
ssh your_netid@amarel.rutgers.edu

# Check job status
squeue -u $USER

# Expected running jobs from Exp #1-2:
# - Baseline-SGD (Job 48316995)
# - SWAG-SGD (Job 48316996)
# - MC Dropout-SGD (Job 48317107)
# - Ensemble-SGD (Job 48317108)
```

## Experiment #3: Adam Optimizer (50 epochs)

### Submit All Methods

```bash
cd ~/uq_capstone

# Submit Adam experiments (total: 8 GPU hours)
sbatch scripts/amarel/train_baseline_adam.slurm      # ~1 hour
sbatch scripts/amarel/train_swag_adam.slurm          # ~1 hour
sbatch scripts/amarel/train_mc_dropout_adam.slurm    # ~1 hour
sbatch scripts/amarel/train_ensemble_adam.slurm      # ~1 hour × 5 members (parallel)

# Check submission
squeue -u $USER | grep adam
```

### Monitor Progress

```bash
# Watch logs in real-time
tail -f logs/baseline_adam_*.log
tail -f logs/swag_adam_*.log
tail -f logs/dropout_adam_*.log
tail -f logs/ensemble_adam_*.log

# Check for errors
grep -i error logs/*adam*.err

# Check accuracy in logs
grep "Test Acc" logs/*adam*.log
```

## Experiment #4: 300-Epoch Training

### Submit All Methods

```bash
cd ~/uq_capstone

# Submit 300-epoch experiments (total: 48 GPU hours)
sbatch scripts/amarel/train_baseline_sgd_300.slurm      # ~6 hours
sbatch scripts/amarel/train_swag_sgd_300.slurm          # ~6 hours
sbatch scripts/amarel/train_mc_dropout_sgd_300.slurm    # ~6 hours
sbatch scripts/amarel/train_ensemble_sgd_300.slurm      # ~6 hours × 5 members (parallel)

# Check submission
squeue -u $USER | grep 300
```

### Monitor Progress

```bash
# Watch logs
tail -f logs/baseline_sgd_300_*.log
tail -f logs/swag_sgd_300_*.log
tail -f logs/dropout_sgd_300_*.log
tail -f logs/ensemble_sgd_300_*.log

# Check current epoch
grep "Epoch" logs/*300*.log | tail -n 4

# Estimate time remaining (e.g., if at epoch 50/300)
# Remaining = (300-50) / 50 * elapsed_time
```

## Job Management

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View detailed job info
scontrol show job JOB_ID

# Check job efficiency
seff JOB_ID
```

### Cancel Jobs (if needed)

```bash
# Cancel specific job
scancel JOB_ID

# Cancel all your jobs (use with caution!)
scancel -u $USER

# Cancel all jobs matching pattern
scancel -u $USER --name=ensemble_adam
```

### Check Available Resources

```bash
# Check GPU availability
sinfo -p gpu

# Check your job priority
sprio -u $USER
```

## Expected Output Locations

After completion, results will be in:

```
results/
├── baseline_adam/           # Exp #3: Baseline with Adam
│   ├── config.json
│   ├── best_model.pt
│   ├── final_model.pt
│   └── history.json
├── swag_adam/              # Exp #3: SWAG with Adam
│   ├── config.json
│   ├── swag_model.pt
│   └── history.json
├── mc_dropout_adam/        # Exp #3: MC Dropout with Adam
│   └── ...
├── ensemble_adam/          # Exp #3: Ensemble with Adam
│   ├── member_0/
│   ├── member_1/
│   ├── ...
│   └── ensemble_summary.json
├── baseline_sgd_300/       # Exp #4: Baseline 300 epochs
│   └── ...
├── swag_sgd_300/           # Exp #4: SWAG 300 epochs
│   └── ...
├── mc_dropout_sgd_300/     # Exp #4: MC Dropout 300 epochs
│   └── ...
└── ensemble_sgd_300/       # Exp #4: Ensemble 300 epochs
    └── ...
```

## Download Results

### From Amarel to Local Machine

```bash
# On your local machine (PowerShell)
# Download specific experiment
rsync -avz your_netid@amarel.rutgers.edu:~/uq_capstone/results/baseline_adam/ results/baseline_adam/

# Download all results
rsync -avz your_netid@amarel.rutgers.edu:~/uq_capstone/results/ results/

# Download logs
rsync -avz your_netid@amarel.rutgers.edu:~/uq_capstone/logs/ logs/
```

### Check Downloaded Results

```bash
# Check result structure
ls -lah results/*/

# Check if models exist
find results/ -name "*.pt" -type f

# Check configuration files
find results/ -name "config.json" -type f -exec jq '.epochs, .lr' {} \;

# Quick accuracy check
grep "test_acc" results/*/history.json
```

## Troubleshooting

### Job Failed to Start

```bash
# Check job status
scontrol show job JOB_ID

# Common issues:
# 1. No GPU available -> Wait or reduce --gres=gpu:1
# 2. Memory limit -> Reduce batch size
# 3. Time limit -> Increase --time
```

### Job Running But No Output

```bash
# Check if job is actually running
squeue -j JOB_ID -O JobID,State,NodeList,Reason

# Check log file size (should be growing)
ls -lh logs/*.log

# Tail the log file
tail -f logs/JOB_NAME_JOB_ID.log
```

### Out of Memory Error

```bash
# Check error log
cat logs/JOB_NAME_JOB_ID.err

# If OOM, reduce batch size in script:
# --batch_size 32 -> --batch_size 16
```

### CUDA Error

```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check CUDA version match
python -c "import torch; print(torch.version.cuda)"
nvidia-smi
```

## Estimated Timeline

| Action | Duration | Notes |
|--------|----------|-------|
| Monitor Exp #1-2 | 0-12 hours | Already running |
| Submit Exp #3 | 5 minutes | 4 sbatch commands |
| Run Exp #3 | 1-6 hours | Parallel execution |
| Submit Exp #4 | 5 minutes | 4 sbatch commands |
| Run Exp #4 | 6-12 hours | Parallel execution |
| Download results | 10-30 minutes | Depends on network |
| **Total Time** | **~24 hours** | **Including wait time** |

## Quick Commands Reference

```bash
# Check all jobs
squeue -u $USER

# Monitor specific experiment
watch -n 10 'squeue -u $USER | grep adam'

# Check latest log output
tail -n 50 logs/*adam*.log | grep -E "Epoch|Acc|Loss"

# Count completed models
find results/ -name "best_model.pt" | wc -l

# Check disk usage
du -sh results/
```

## Next Steps After Completion

1. **Verify all results:**
   ```bash
   python -c "
   import json
   from pathlib import Path
   for exp in ['baseline_adam', 'swag_adam', 'mc_dropout_adam', 'ensemble_adam',
               'baseline_sgd_300', 'swag_sgd_300', 'mc_dropout_sgd_300', 'ensemble_sgd_300']:
       hist = Path(f'results/{exp}/history.json')
       if hist.exists():
           data = json.load(open(hist))
           print(f'{exp}: {data.get(\"test_acc\", [])[-1]:.2f}%')
   "
   ```

2. **Generate comparative analysis:**
   ```bash
   python analysis/compare_experiments.py
   python analysis/generate_uq_report.py
   ```

3. **Update thesis with results**

4. **Create visualizations for presentation**

---

**Quick Start (Copy-Paste):**

```bash
# Submit Experiment #3 (Adam)
cd ~/uq_capstone
sbatch scripts/amarel/train_baseline_adam.slurm && \
sbatch scripts/amarel/train_swag_adam.slurm && \
sbatch scripts/amarel/train_mc_dropout_adam.slurm && \
sbatch scripts/amarel/train_ensemble_adam.slurm && \
echo "Experiment #3 submitted!"

# Submit Experiment #4 (300 epochs)
sbatch scripts/amarel/train_baseline_sgd_300.slurm && \
sbatch scripts/amarel/train_swag_sgd_300.slurm && \
sbatch scripts/amarel/train_mc_dropout_sgd_300.slurm && \
sbatch scripts/amarel/train_ensemble_sgd_300.slurm && \
echo "Experiment #4 submitted!"

# Monitor
watch -n 30 'squeue -u $USER; echo "---"; tail -n 5 logs/*adam*.log logs/*300*.log 2>/dev/null | grep -E "Epoch|Test Acc"'
```
