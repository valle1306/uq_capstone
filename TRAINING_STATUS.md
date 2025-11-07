# Training Status

**Job IDs:**
- MC Dropout: 47964089
- SWAG: 47964090

**Status:** Running on gpu028 (started Nov 3, 20:51)

**ETA:** ~24 hours (should complete Nov 4, 20:51)

## When Complete

```bash
ssh hpl14@amarel.rutgers.edu
cd /scratch/hpl14/uq_capstone

# Verify models
ls -lh runs/classification/mc_dropout/best_model.pth
ls -lh runs/classification/swag_classification/swag_model.pth

# Run metrics + visualizations
sbatch scripts/eval_and_visualize_on_amarel.sbatch

# Wait ~30-60 min, then review results
cat runs/classification/metrics/EVALUATION_REPORT.txt

# If good, pull to Windows
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics ./runs/
```

## Monitor Progress

```bash
squeue -u hpl14
tail -f logs/retrain_mc_dropout_47964089.out
tail -f logs/retrain_swag_47964090.out
```
