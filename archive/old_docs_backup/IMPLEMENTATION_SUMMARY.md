# UQ Capstone Project - Implementation Summary

## ✅ COMPLETED: Full UQ Implementation

**Date**: October 10, 2025  
**Status**: Ready to run all experiments on Amarel

---

## 🎯 What Was Implemented

### 1. **Core UQ Methods** (`src/uq_methods.py`)
✅ **Temperature Scaling** - Post-hoc calibration with learned temperature parameter  
✅ **MC Dropout** - Monte Carlo sampling with dropout at test time  
✅ **Deep Ensemble** - Multiple models with different initializations  
✅ **Conformal Prediction** - Prediction sets with coverage guarantees  
✅ **Uncertainty Metrics** - ECE, calibration, uncertainty quantification metrics

### 2. **Training Scripts**
✅ `train_baseline.py` - Standard U-Net (no UQ)  
✅ `train_mc_dropout.py` - Model with dropout for MC sampling  
✅ `train_ensemble_member.py` - Individual ensemble member training  
✅ `evaluate_uq.py` - Comprehensive evaluation and comparison

### 3. **SLURM Job Scripts** (Amarel)
✅ `train_baseline.sbatch` - Baseline training job  
✅ `train_mc_dropout.sbatch` - MC Dropout training job  
✅ `train_ensemble.sbatch` - Ensemble array job (5 members parallel)  
✅ `evaluate_uq.sbatch` - Evaluation and comparison job  
✅ `run_all_experiments.sh` - Master script to run everything

### 4. **Documentation**
✅ `UQ_EXPERIMENTS_GUIDE.md` - Complete usage guide  
✅ `TEST_RUN_SUCCESS.md` - Working test run documentation  
✅ Upload script with instructions

---

## 📊 Experiment Pipeline

```
1. Baseline Training (2-3 hrs)
   ├─> Standard U-Net model
   └─> runs/baseline/best_model.pth

2. MC Dropout Training (2-3 hrs)
   ├─> U-Net with dropout
   └─> runs/mc_dropout/best_model.pth

3. Deep Ensemble Training (2-3 hrs, parallel)
   ├─> 5 models with different seeds
   └─> runs/ensemble/member_*/best_model.pth

4. Evaluation (30 min)
   ├─> Compare all methods
   ├─> Temperature Scaling (uses baseline)
   ├─> Conformal Prediction (uses baseline)
   ├─> Generate metrics & plots
   └─> runs/evaluation/results.json + comparison.png
```

**Total Time**: ~6-8 hours (with parallel execution)

---

## 🚀 How to Run

### Method 1: Run Everything Automatically

```bash
# On your local machine:
powershell -ExecutionPolicy Bypass -File .\scripts\upload_uq_code.ps1

# Then SSH to Amarel:
ssh hpl14@amarel.rutgers.edu
cd /scratch/hpl14/uq_capstone
bash scripts/run_all_experiments.sh

# Monitor:
squeue -u hpl14
```

### Method 2: Run Jobs Individually

```bash
# On Amarel:
cd /scratch/hpl14/uq_capstone

# Submit jobs:
sbatch scripts/train_baseline.sbatch
sbatch scripts/train_mc_dropout.sbatch
sbatch scripts/train_ensemble.sbatch

# After all complete:
sbatch scripts/evaluate_uq.sbatch
```

---

## 📈 Expected Results

The evaluation will produce:

### `runs/evaluation/results.json`
```json
[
  {
    "method": "Baseline",
    "dice": 0.XX,
    "ece": 0.XX
  },
  {
    "method": "Temperature Scaling",
    "dice": 0.XX,
    "ece": 0.XX (should be lower than baseline)
  },
  {
    "method": "MC Dropout",
    "dice": 0.XX,
    "ece": 0.XX,
    "mean_uncertainty": 0.XX,
    "uncertainty_on_errors": 0.XX (should be high),
    "uncertainty_on_correct": 0.XX (should be low)
  },
  {
    "method": "Deep Ensemble",
    "dice": 0.XX,
    "ece": 0.XX,
    "mean_uncertainty": 0.XX,
    "uncertainty_on_errors": 0.XX (should be high),
    "uncertainty_on_correct": 0.XX (should be low)
  },
  {
    "method": "Conformal Prediction",
    "target_coverage": 0.90,
    "actual_coverage": 0.XX (should be ~0.90),
    "threshold": 0.XX
  }
]
```

### `runs/evaluation/comparison.png`
- Bar plots comparing Dice scores and ECE across methods
- Visual comparison of performance vs calibration

---

## 🔬 What Each Method Tests

### Baseline
- Raw model performance without UQ
- Benchmark for comparison

### Temperature Scaling
- Tests: Can we improve calibration post-hoc?
- Expected: Lower ECE than baseline
- Use case: Quick calibration fix

### MC Dropout
- Tests: Does test-time dropout provide useful uncertainty?
- Expected: Higher uncertainty on errors
- Use case: Single-model uncertainty estimation

### Deep Ensemble
- Tests: Does model diversity improve uncertainty?
- Expected: Best uncertainty quality, higher compute cost
- Use case: When you can afford multiple models

### Conformal Prediction
- Tests: Can we get coverage guarantees?
- Expected: Actual coverage ≈ target coverage
- Use case: When you need statistical guarantees

---

## 📁 File Structure After Experiments

```
uq_capstone/
├── runs/
│   ├── baseline/
│   │   ├── best_model.pth
│   │   ├── config.json
│   │   └── history.json
│   ├── mc_dropout/
│   │   ├── best_model.pth
│   │   ├── config.json
│   │   └── history.json
│   ├── ensemble/
│   │   ├── member_0/
│   │   │   ├── best_model.pth
│   │   │   ├── config.json
│   │   │   └── history.json
│   │   ├── member_1/ ...
│   │   └── member_4/
│   └── evaluation/
│       ├── results.json          ← Main results
│       └── comparison.png        ← Plots
```

---

## ✅ Checklist

- [x] UQ methods implemented
- [x] Training scripts created
- [x] SLURM jobs configured
- [x] Documentation written
- [x] Upload script ready
- [ ] **Upload code to Amarel** ← NEXT STEP
- [ ] **Run experiments** ← AFTER UPLOAD
- [ ] **Analyze results**
- [ ] **Write capstone report**

---

## 🎓 For Your Capstone Report

### Key Comparisons to Make:

1. **Performance**: Which method has best Dice score?
2. **Calibration**: Which has lowest ECE?
3. **Uncertainty Quality**: Which best distinguishes errors vs correct predictions?
4. **Computational Cost**: Time/memory tradeoffs
5. **Practical Use**: When to use each method?

### Possible Findings:

- Temperature scaling improves calibration with minimal cost
- MC Dropout provides reasonable uncertainty from single model
- Ensembles likely provide best uncertainty but 5x compute cost
- Conformal prediction provides guarantees but less interpretable

---

## 📚 References in Project

- `papers/baseline_for_uncertainty_DL.pdf` - Background on UQ in DL
- `papers/gentle_intro_conformal_dfuq.pdf` - Conformal prediction intro
- Dr. Moran's email - Original guidance and method list

---

## 🆘 Need Help?

**For technical issues:**
- Check `UQ_EXPERIMENTS_GUIDE.md` for detailed instructions
- Check error logs: `runs/*/train_*.err`
- Contact: Dr. Gemma Moran

**For cluster issues:**
- Amarel documentation: https://sites.google.com/view/cluster-user-guide
- OARC help: help@oarc.rutgers.edu

---

## 🎉 Ready to Go!

Everything is implemented and tested. You just need to:

1. Run the upload script
2. Submit the experiments
3. Wait 6-8 hours
4. Analyze the results!

**Good luck with your capstone project!** 🚀
