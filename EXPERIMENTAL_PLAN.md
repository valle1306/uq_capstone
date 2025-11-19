# Experimental Plan: Addressing Instructor Questions
**Date:** November 18, 2025  
**Purpose:** Fair comparison and proper conformal calibration

---

## 🎯 **Three Key Questions to Address**

### Question 1: What are the SWAG updates if Adam is used instead of SGD?
**Paper says:** "Future work" but uses Adam in appendix

### Question 2: Compare apples to apples
**Current Problem:** Mixed optimizers (Adam for Baseline/MC/Ensemble, SGD for SWAG)

### Question 3: Check conformal calibration set size
**Issue:** Need to verify calibration set is appropriate size

---

## 📊 **Current State - UNFAIR Comparison**

| Method | Optimizer | LR | Accuracy | Status |
|--------|-----------|-----|----------|---------|
| Baseline | **Adam** | 0.001 | 91.67% | ✓ Done |
| MC Dropout | **Adam** | 0.001 | 85.26% | ✓ Done |
| Deep Ensemble | **Adam** | 0.001 | 91.67% | ✓ Done |
| SWAG (old) | **Adam** | 0.001 | 83.17% | ✓ Done |
| SWAG (SGD aggressive) | **SGD** | 0.05 | 79.65% | ✗ Failed |
| SWAG (SGD conservative) | **SGD** | 0.01 | ??? | ⏳ Running |

**Problem:** We're comparing Adam-trained models (91%+) with SGD-trained SWAG (79-83%)!

---

## 🔬 **Experimental Plan**

### Experiment Set 1: **SGD for Everything** (Paper-Correct)
Retrain ALL models with SGD + momentum + weight_decay to match paper

**Benefits:**
- ✓ Scientifically rigorous (follows paper exactly)
- ✓ Fair comparison (same optimizer for all)
- ✓ Tests if SGD helps all methods or just SWAG

**Drawbacks:**
- Takes time (4-5 models × 24 hours each)
- Might not improve Baseline/Ensemble (they're already at 91%)

**Models to retrain:**
1. Baseline (SGD, lr=0.01, momentum=0.9, weight_decay=1e-4)
2. MC Dropout (SGD, lr=0.01, momentum=0.9, weight_decay=1e-4)
3. Deep Ensemble (5 models, SGD, lr=0.01, momentum=0.9, weight_decay=1e-4)
4. SWAG (SGD, lr=0.01, momentum=0.9, weight_decay=1e-4) ← Already running

---

### Experiment Set 2: **Adam for Everything** (Practical)
Keep current Adam models, retrain SWAG with Adam properly

**Benefits:**
- ✓ Fast (only 1 model to retrain)
- ✓ Fair comparison (all models use Adam)
- ✓ Practical (Adam works well for medical imaging)

**Drawbacks:**
- Deviates from SWAG paper (they use SGD)
- Need to understand what changes with Adam

**What Changes with Adam for SWAG?**

According to the paper's appendix/future work, with Adam:
1. **No momentum** (Adam has built-in momentum via moving averages)
2. **Different LR schedule** (Adam is less sensitive to LR)
3. **Weight averaging still works** (core SWAG idea is optimizer-agnostic)
4. **May need different collection frequency** (Adam converges differently)

**Implementation:**
```python
# SWAG with Adam (modified from paper)
optimizer = Adam(
    lr=0.001,        # Same as other models
    weight_decay=1e-4  # Keep L2 regularization
)

# Simpler schedule (Adam doesn't need aggressive annealing)
scheduler = CosineAnnealingLR(T_max=30)
swa_scheduler = ConstantLR(0.0001)  # Constant low LR for SWAG phase

# Rest stays the same:
- Collect snapshots epochs 30-50
- Update batch norm after collection
- Sample T=30 models for prediction
```

---

### Experiment Set 3: **Conformal Calibration Analysis**
Check if calibration set size is appropriate

**Current Setup:**
- Train: 4,172 samples
- **Calibration: 1,044 samples** ← Is this enough?
- Test: 624 samples

**Questions to answer:**
1. Is 1,044 calibration samples sufficient?
2. How does calibration set size affect coverage guarantees?
3. Should we use different train/cal/test split?

**Experiments:**
```python
# Test different calibration set sizes
cal_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]  # of validation+test combined

For each fraction:
    1. Split data differently
    2. Calibrate conformal predictor
    3. Measure:
       - Empirical coverage
       - Set size
       - Coverage stability (run 5 times with different random splits)
```

**Expected insight:**
- Smaller cal set → less stable coverage (might miss target 90%)
- Larger cal set → more stable, but less test data
- Paper suggests: n_cal ≥ 1000 for stable guarantees

**Our 1,044 calibration samples should be okay**, but we should verify!

---

## 🎯 **Recommended Action Plan**

### **Option A: Quick Fix (RECOMMENDED for tight deadline)**
1. **Retrain SWAG with Adam** (same as other models) - 24 hours
2. **Verify conformal calibration** with current 1,044 samples - 1 hour
3. **Compare all methods fairly** (all Adam) - immediate
4. **Document** why we used Adam (practical choice, still valid SWAG)

**Timeline:** 1 day  
**Result:** Fair comparison, practical solution

---

### **Option B: Rigorous Approach (if you have time)**
1. **Retrain all 4 methods with SGD** - 4-5 days
2. **Compare SGD vs Adam** for each method - 1 day analysis
3. **Verify conformal calibration** - 1 hour
4. **Document** full ablation study

**Timeline:** 5-6 days  
**Result:** Publication-ready, scientifically rigorous

---

### **Option C: Both (Best but time-consuming)**
1. **Do Option A** (Adam for everything) - 1 day
2. **Then do Option B** (SGD for everything) - 5 days
3. **Show both comparisons** in thesis/defense
4. **Discuss** optimizer choice trade-offs

**Timeline:** 6 days  
**Result:** Most complete, shows you understand the issues deeply

---

## 📝 **What I'll Create Now**

Based on your timeline, I recommend **Option A**. I'll create:

1. **`src/retrain_swag_adam.py`** - SWAG with Adam optimizer
2. **`scripts/retrain_swag_adam.sbatch`** - SLURM script for Amarel
3. **`src/analyze_conformal_calibration.py`** - Verify cal set size
4. **`scripts/conformal_calibration_analysis.sbatch`** - Run analysis

This gives you:
- ✅ Fair comparison (all Adam)
- ✅ Verified conformal calibration
- ✅ Quick turnaround (1-2 days)
- ✅ Can still do Option B later if needed

**Should I proceed with Option A scripts?** Or do you prefer Option B (retrain everything with SGD)?
