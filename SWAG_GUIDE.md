# 🧠 SWAG Implementation Guide

## What is SWAG?

**SWAG (Stochastic Weight Averaging-Gaussian)** is an uncertainty quantification method from the paper:
> "A Simple Baseline for Bayesian Uncertainty Estimation in Deep Learning" by Maddox et al. (2019)

SWAG provides **Bayesian uncertainty** estimation by:
1. Training a model normally (warmup phase)
2. Collecting weight snapshots during learning rate annealing
3. Fitting a Gaussian distribution over weights
4. Sampling from this distribution at test time

---

## Implementation Details

### Training Procedure

```
Epochs 0-14:  Normal training (lr=1e-3)
              ├── Standard SGD optimization
              └── No snapshot collection

Epoch 15:     Switch to SWAG mode
              └── Reduce lr to 1e-4 (annealing)

Epochs 15-29: SWAG collection phase
              ├── Continue training with low lr
              ├── Collect weight snapshot every epoch
              └── Build Gaussian posterior
```

### Key Parameters

- **`swag_start_epoch`**: When to start collecting (default: 15)
- **`swag_lr`**: Learning rate for collection phase (default: 1e-4)
- **`max_num_models`**: Max snapshots to store (K parameter, default: 20)
- **`collect_freq`**: How often to collect (default: every 1 epoch)
- **`n_samples`**: Posterior samples at test time (default: 30)
- **`scale`**: Sampling scale factor (default: 0.5)

---

## How SWAG Works

### 1. Collection Phase (Training)

For each collected weight snapshot w_t:

```
mean = (n*mean + w_t) / (n+1)        # Running mean
sq_mean = (n*sq_mean + w_t²) / (n+1) # Running mean of squares
deviation = w_t - mean                # Store deviation
```

### 2. Posterior Approximation

SWAG fits: **p(w) ~ N(θ_SWA, Σ_SWA)**

Where:
- θ_SWA: Mean of collected weights
- Σ_SWA: Covariance (diagonal + low-rank)

```
Diagonal variance:  σ² = E[w²] - E[w]²
Low-rank component: (1/√(2(K-1))) * D^T * D
                    where D = stacked deviations
```

### 3. Prediction with Uncertainty

At test time:
1. Sample N models from posterior: w_i ~ N(θ_SWA, Σ_SWA)
2. Run each sampled model on input
3. Average predictions: mean_pred = (1/N) Σ pred_i
4. Compute uncertainty: uncertainty = std(pred_i)

---

## Files Created

### Core Implementation
- **`src/swag.py`**: SWAG class implementation
  - `SWAG`: Main wrapper class
  - `collect_model()`: Collect weight snapshots
  - `sample()`: Sample from posterior
  - `predict_with_uncertainty()`: Make predictions with UQ
  - `SWAGScheduler`: Helper for collection scheduling

### Training
- **`src/train_swag.py`**: Training script
  - Implements 2-phase training (warmup + collection)
  - Automatically switches LR at `swag_start_epoch`
  - Saves SWAG statistics to checkpoint

### SLURM Job
- **`scripts/train_swag.sbatch`**: Cluster job
  - 1 GPU, 16GB RAM, 4 hours
  - Runs 30 epochs total (15 warmup + 15 collection)

### Evaluation
- **`src/evaluate_uq.py`**: Updated to include SWAG
  - `evaluate_swag()`: Computes Dice, ECE, uncertainty metrics
  - Samples 30 models from posterior for predictions

---

## Running SWAG

### Option 1: Run All Experiments (Recommended)

```bash
cd /scratch/hpl14/uq_capstone
bash scripts/run_all_experiments.sh
```

This runs: Baseline, MC Dropout, Ensemble, **SWAG**, then Evaluation

### Option 2: Run SWAG Only

```bash
sbatch scripts/train_swag.sbatch

# Monitor
squeue -u hpl14
tail -f runs/swag/train_*.out
```

### Option 3: Manual Training

```bash
python src/train_swag.py \
    --data_dir /scratch/hpl14/uq_capstone/data/brats_subset_npz \
    --output_dir runs/swag \
    --epochs 30 \
    --swag_start 15 \
    --swag_lr 1e-4 \
    --max_models 20
```

---

## Expected Outputs

### Training Outputs

```
runs/swag/
├── swag_model.pth          # SWAG statistics (mean, variance, deviations)
├── best_base_model.pth     # Best checkpoint during training
├── history.json            # Training/val loss history
├── config.json             # Training configuration
├── train_*.out             # SLURM stdout
└── train_*.err             # SLURM stderr
```

### Checkpoint Contents

```python
swag_model.pth:
{
    'n_models': 15,                    # Number of snapshots collected
    'mean': Tensor([...]),             # Mean weights (flat vector)
    'sq_mean': Tensor([...]),          # Mean of squared weights
    'cov_mat_sqrt': [Tensor, ...],     # Deviation vectors (list)
    'max_num_models': 20,              # K parameter
    'config': {...}                    # Training config
}
```

---

## Evaluation Metrics

SWAG is evaluated on:

1. **Dice Score**: Segmentation accuracy
2. **ECE (Expected Calibration Error)**: How well-calibrated predictions are
3. **Mean Uncertainty**: Average prediction uncertainty
4. **Uncertainty on Errors**: Higher uncertainty on incorrect predictions (good!)
5. **Uncertainty on Correct**: Lower uncertainty on correct predictions

---

## Comparison to Other Methods

| Method | Uncertainty | Cost | Quality |
|--------|-------------|------|---------|
| **Baseline** | ❌ None | 1× | N/A |
| **Temperature Scaling** | ⚠️ Calibrated probabilities | 1× | Low |
| **MC Dropout** | ✅ Epistemic | 1× train, N× test | Medium |
| **Deep Ensemble** | ✅ Epistemic | 5× train, 5× test | **High** |
| **SWAG** | ✅ Bayesian | 1× train, N× test | **High** |
| **Conformal** | ✅ Coverage guarantees | 1× + calibration | Medium |

**SWAG Advantage**: Similar uncertainty quality to Deep Ensemble, but only requires training **one model**!

---

## Key Paper Results

From Maddox et al. (2019):

- SWAG matches or exceeds ensemble performance
- Works well with 15-20 collected snapshots
- Scale=0.5 is optimal for most tasks
- Start collecting after learning rate annealing
- Collect every 1-2 epochs during annealing

---

## Troubleshooting

### "Not enough models collected"
- Ensure `swag_start_epoch < epochs`
- Check that collection phase has enough epochs
- Need at least 10-15 snapshots for good results

### "Memory error"
- Reduce `max_num_models` (try 10-15)
- Reduce `batch_size`
- Use smaller model

### "Poor uncertainty estimates"
- Try different `scale` (0.3-0.7)
- Collect more snapshots
- Ensure proper learning rate annealing

### "SWAG worse than baseline"
- Check that collection phase learns (loss should decrease)
- Ensure `swag_lr` is not too low (try 5e-5 to 1e-4)
- Verify snapshots are diverse (check training logs)

---

## References

```
@article{maddox2019simple,
  title={A simple baseline for Bayesian uncertainty estimation in deep learning},
  author={Maddox, Wesley J and Izmailov, Pavel and Garipov, Timur and Vetrov, Dmitry P and Wilson, Andrew Gordon},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

**Paper**: https://arxiv.org/abs/1902.02476

---

## Next Steps

1. ✅ Upload code: `scripts\upload_uq_code.bat`
2. ✅ Run experiments: `bash scripts/run_all_experiments.sh`
3. ⏳ Wait 6-8 hours
4. 📊 Check results: `cat runs/evaluation/results.json`
5. 📈 View plots: Download `runs/evaluation/comparison.png`

SWAG will appear in the comparison alongside all other UQ methods!
