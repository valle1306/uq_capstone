# Uncertainty Quantification Experiments - Complete Guide

This guide walks through running all uncertainty quantification (UQ) methods for brain tumor segmentation.

## Methods Implemented

1. **Baseline** - Standard U-Net (no uncertainty)
2. **Temperature Scaling** - Post-hoc calibration method
3. **MC Dropout** - Monte Carlo sampling with dropout at test time
4. **Deep Ensemble** - Train 5 models with different initializations
5. **Conformal Prediction** - Provides prediction sets with coverage guarantees

## Quick Start (Run Everything)

### On Amarel:

```bash
# SSH to Amarel
ssh hpl14@amarel.rutgers.edu

# Navigate to project
cd /scratch/hpl14/uq_capstone

# Run all experiments (submits all jobs with dependencies)
bash scripts/run_all_experiments.sh

# Monitor progress
squeue -u hpl14

# Check results when done
cat runs/evaluation/results.json
```

This will:
- Train baseline model (~2-3 hours)
- Train MC dropout model (~2-3 hours)
- Train 5 ensemble members in parallel (~2-3 hours)
- Evaluate all methods (~30 min)
- Generate comparison plots and metrics

**Total time: ~6-8 hours**

## Step-by-Step Manual Execution

If you prefer to run jobs individually:

### 1. Train Baseline Model

```bash
sbatch scripts/train_baseline.sbatch
```

- **Purpose**: Standard segmentation model without uncertainty
- **Output**: `runs/baseline/best_model.pth`
- **Time**: ~2-3 hours
- **Monitor**: `tail -f runs/baseline/train_*.out`

### 2. Train MC Dropout Model

```bash
sbatch scripts/train_mc_dropout.sbatch
```

- **Purpose**: Model with dropout layers for MC sampling
- **Output**: `runs/mc_dropout/best_model.pth`
- **Time**: ~2-3 hours
- **Note**: Uses dropout rate of 0.2

### 3. Train Deep Ensemble

```bash
sbatch scripts/train_ensemble.sbatch
```

- **Purpose**: Train 5 models with different seeds
- **Output**: `runs/ensemble/member_*/best_model.pth` (5 files)
- **Time**: ~2-3 hours (parallel array job)
- **Note**: Array job trains all 5 members simultaneously

### 4. Evaluate All Methods

```bash
sbatch scripts/evaluate_uq.sbatch
```

- **Purpose**: Compare all UQ methods
- **Requires**: All models from steps 1-3 must be trained
- **Output**: 
  - `runs/evaluation/results.json` (metrics)
  - `runs/evaluation/comparison.png` (plots)
- **Time**: ~30 minutes

## Understanding the Results

### Key Metrics

**Dice Score**
- Measures segmentation accuracy (0-1, higher better)
- Same across all methods for same model quality

**ECE (Expected Calibration Error)**
- Measures calibration quality (0-1, lower better)
- Temperature scaling should reduce ECE
- Good calibration means predicted probabilities match actual frequencies

**Uncertainty Metrics** (MC Dropout & Ensemble only)
- `mean_uncertainty`: Average predictive uncertainty
- `uncertainty_on_errors`: Uncertainty when model makes mistakes (should be HIGH)
- `uncertainty_on_correct`: Uncertainty when model is correct (should be LOW)
- Good UQ: High uncertainty on errors, low uncertainty on correct predictions

**Coverage** (Conformal Prediction only)
- Target: 90% (if alpha=0.1)
- Should achieve close to target coverage on test set
- Provides finite-sample guarantees

### Example Results Interpretation

```json
{
  "method": "MC Dropout",
  "dice": 0.85,
  "ece": 0.08,
  "mean_uncertainty": 0.12,
  "uncertainty_on_errors": 0.25,
  "uncertainty_on_correct": 0.08
}
```

This shows:
- Good segmentation (85% Dice)
- Reasonable calibration (8% ECE)
- Model is more uncertain (0.25) when wrong vs correct (0.08) ✓

## File Structure

```
uq_capstone/
├── src/
│   ├── uq_methods.py              # UQ implementations
│   ├── train_baseline.py          # Baseline training
│   ├── train_mc_dropout.py        # MC Dropout training
│   ├── train_ensemble_member.py   # Ensemble member training
│   ├── evaluate_uq.py             # Evaluation & comparison
│   └── segmentation_utils.py      # U-Net model
│
├── scripts/
│   ├── run_all_experiments.sh     # Master script
│   ├── train_baseline.sbatch      # Baseline job
│   ├── train_mc_dropout.sbatch    # MC Dropout job
│   ├── train_ensemble.sbatch      # Ensemble array job
│   └── evaluate_uq.sbatch         # Evaluation job
│
└── runs/
    ├── baseline/                  # Baseline results
    ├── mc_dropout/                # MC Dropout results
    ├── ensemble/                  # Ensemble results
    │   ├── member_0/
    │   ├── member_1/
    │   └── ...
    └── evaluation/                # Final comparison
        ├── results.json           # All metrics
        └── comparison.png         # Comparison plots
```

## Troubleshooting

### Job not starting?
```bash
squeue -u hpl14  # Check queue
scontrol show job JOBID  # Check why pending
```

### Job failed?
```bash
# Check error logs
cat runs/baseline/train_*.err
cat runs/mc_dropout/train_*.err
cat runs/ensemble/train_member*_*.err
```

### Missing dependencies?
```bash
# Activate environment and install
conda activate uq_capstone
pip install matplotlib  # For plots in evaluation
```

### Want to run with different hyperparameters?

Edit the .sbatch files or run directly:

```bash
python src/train_baseline.py \
    --data_root data/brats \
    --epochs 50 \
    --batch 16 \
    --lr 1e-4 \
    --save_dir runs/baseline_custom
```

## Experiment Variations

### More Epochs
Change `--epochs 30` to `--epochs 50` in sbatch files for better convergence.

### Larger Ensemble
Change `#SBATCH --array=0-4` to `#SBATCH --array=0-9` for 10 members.

### More MC Samples
In `evaluate_uq.sbatch`, change `--mc_samples 20` to `--mc_samples 50` for better uncertainty estimates.

### Different Calibration Level
In `evaluate_uq.py`, change `alpha=0.1` to `alpha=0.05` for 95% coverage.

## Next Steps

1. **Analyze Results**: Compare methods in `results.json`
2. **Visualize**: Look at uncertainty maps (implement custom visualization)
3. **Scale Up**: Use more data or full BraTS dataset
4. **Tune Hyperparameters**: Optimize for each method
5. **Write Report**: Document findings for capstone

## References

- Temperature Scaling: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
- MC Dropout: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
- Deep Ensembles: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty" (NIPS 2017)
- Conformal Prediction: Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2021)

## Questions?

Contact Dr. Gemma Moran or check the papers linked in `papers/` directory.
