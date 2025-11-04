# ✅ Training Complete - Next Steps

**Status:** MC Dropout and SWAG retraining jobs finished ✅

## What Happens Now

Instead of pulling results to Windows and evaluating locally, we run **comprehensive metrics + visualizations directly on Amarel GPU** for faster results.

## Your Next Commands (on Amarel)

### 1. Verify Models
```bash
ssh hpl14@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone
ls -lh runs/classification/mc_dropout/best_model.pth
ls -lh runs/classification/swag_classification/swag_model.pth
```

### 2. Run Metrics + Visualizations (ONE JOB)
```bash
sbatch scripts/eval_and_visualize_on_amarel.sbatch
squeue -u hpl14  # Check status
```

**What This Does (on GPU):**
- Evaluates all 5 UQ methods
- Computes calibration metrics (ECE, MCE, Brier)
- Generates all plots/visualizations
- Creates summary report
- Takes ~30-60 minutes

### 3. Review Results on Amarel
```bash
# View summary
cat runs/classification/metrics/EVALUATION_REPORT.txt

# Check accuracies (should be ~90%)
python -c "import json; r = json.load(open('runs/classification/metrics/comprehensive_metrics.json')); print(f'MC Dropout: {r[\"mc_dropout\"][\"accuracy\"]:.2%}'); print(f'SWAG: {r[\"swag\"][\"accuracy\"]:.2%}')"
```

### 4. If Results Look Good, Pull to Windows
```powershell
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics ./runs/
```

## Advantages of This Approach

✅ **Faster** - GPU compute for metrics evaluation
✅ **No Data Transfer Overhead** - Results stay on Amarel until validated
✅ **Safer** - Review first, pull only if good
✅ **Complete** - All visualizations ready when you pull

## File Reference

- **`POST_TRAINING_WORKFLOW.md`** - Detailed guide (NEW)
- **`RETRAINING_COMMANDS.md`** - Updated with new workflow
- **`scripts/eval_and_visualize_on_amarel.sbatch`** - Combined evaluation job (NEW)

## Expected Results After Evaluation

```
MC Dropout:     ~90% accuracy (was 63.3%)  ← 26.7% improvement ✨
SWAG:           ~90% accuracy (was 79.3%)  ← 10.7% improvement ✨
Baseline:       91.67% accuracy (reference)
Ensemble:       91.67% accuracy (no change)
```

## Timeline

- Metrics + Visualizations: ~30-60 minutes on Amarel GPU
- Pull to Windows: ~5 minutes
- Total before final analysis: ~1 hour

---

**Ready?** Run the commands in the "Your Next Commands" section on Amarel! 🚀

See `POST_TRAINING_WORKFLOW.md` for full details and troubleshooting.
