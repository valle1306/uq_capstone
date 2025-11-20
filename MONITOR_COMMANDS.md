# SGD Training Monitoring Commands

## Check job queue
```powershell
ssh hpl14@amarel.rutgers.edu "squeue -u hpl14"
```

## Watch jobs in real-time (updates every 30 seconds)
```powershell
ssh hpl14@amarel.rutgers.edu "watch -n 30 'squeue -u hpl14'"
```

## Check latest logs (all 4 jobs)
```powershell
ssh hpl14@amarel.rutgers.edu "ls -lt /scratch/hpl14/uq_capstone/logs/*_sgd_*.out | head -8 && echo '' && tail -30 /scratch/hpl14/uq_capstone/logs/baseline_sgd_48316995.out && echo '' && tail -30 /scratch/hpl14/uq_capstone/logs/swag_proper_48316996.out"
```

## Check specific job progress
```powershell
# Baseline
ssh hpl14@amarel.rutgers.edu "tail -f /scratch/hpl14/uq_capstone/logs/baseline_sgd_48316995.out"

# SWAG
ssh hpl14@amarel.rutgers.edu "tail -f /scratch/hpl14/uq_capstone/logs/swag_proper_48316996.out"

# MC Dropout
ssh hpl14@amarel.rutgers.edu "tail -f /scratch/hpl14/uq_capstone/logs/mc_dropout_sgd_48316997.out"

# Ensemble
ssh hpl14@amarel.rutgers.edu "tail -f /scratch/hpl14/uq_capstone/logs/ensemble_sgd_48316998.out"
```

## Check for errors
```powershell
ssh hpl14@amarel.rutgers.edu "cat /scratch/hpl14/uq_capstone/logs/*_sgd_*.err | grep -i error"
```

## Download results when complete
```powershell
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/*_sgd ./runs/classification/
```

## Quick status check (recommended - run every 15 mins)
```powershell
ssh hpl14@amarel.rutgers.edu "squeue -u hpl14 && echo '' && ls -lth /scratch/hpl14/uq_capstone/runs/classification/*/results.json 2>/dev/null | head -4"
```

## Current Jobs (Updated)
- 48316995: Baseline (SGD) - Running, ~1 hour remaining
- 48316996: SWAG (SGD) - Running, ~1 hour remaining
- 48317107: MC Dropout (SGD) - Queued, ~2-3 hours
- 48317108: Deep Ensemble (SGD, 5 members sequentially) - Queued, ~10-12 hours

Expected completion: Baseline/SWAG by ~3:45pm, MC_Dropout by ~6-7pm, Ensemble by ~4-6am
