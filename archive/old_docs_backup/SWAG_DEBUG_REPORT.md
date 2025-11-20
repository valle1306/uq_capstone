# SWAG Debugging Report
**Date:** October 10, 2025  
**Status:** Bug Identified & Fixed (Testing in Progress)

## 🔍 Problem Summary

SWAG evaluation produced anomalous results:
- **Dice Score:** 0.1420 (vs expected ~0.74)
- **Uncertainty:** NaN
- **Training was successful:** val_loss 0.1651, 15 snapshots collected

##  Root Cause Analysis

### Primary Issue: Extreme Variance Leading to Weight Explosion

**Discovered:** The variance computed from SWAG statistics was **unbounded** and contained extreme values.

```python
# Original variance computation
var = torch.clamp(self.sq_mean - self.mean ** 2, 1e-30)  # Only lower bound!

# Variance statistics from checkpoint:
#   Min: -8.34e-07 (NEGATIVE!)
#   Max: 2.26e+08 (HUGE!)
#   Mean: 9.48e+02
#   Negative values: 5,269 out of 31,054,163 parameters
```

**Impact:**
```python
sqrt_var = torch.sqrt(var)
# Max sqrt_var: 15,000!

noise = scale * sqrt_var * randn()
# noise range: [-19,000, +16,200]

sampled_weights = mean + noise  
# sampled weight range: [-187, +249,000] (EXTREME!)
```

### Contributing Factors

1. **Model Snapshots Had Large Deviations**
   - Early snapshots: small deviations (~0.03 mean abs)
   - Later snapshots: **massive deviations** (up to 29,000!)
   - First deviation was all zeros
   - Suggests model was still changing significantly during collection phase

2. **Numerical Precision Issues**
   - Variance = E[W²] - E[W]² is numerically unstable
   - Small differences in large numbers → negative variance

3. **No Upper Variance Bound**
   - Original code only clamped minimum (1e-30)
   - No protection against extreme variance values

## 🔧 Fixes Applied

### Fix 1: Bounded Variance Clamping
```python
# BEFORE:
var = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)

# AFTER:
var = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp, self.max_var)
```

Added `max_var` parameter (default=100.0) to cap maximum variance.

### Fix 2: Correct K Calculation
```python
# BEFORE:
low_rank_sample = (1.0 / np.sqrt(2 * (self.max_num_models - 1))) * D^T @ z2

# AFTER:
K = len(self.cov_mat_sqrt)  # Actual number collected
low_rank_sample = (1.0 / np.sqrt(2 * (K - 1))) * D^T @ z2
```

Uses actual number of collected models, not maximum possible.

### Fix 3: Device Consistency
```python
# BEFORE:
predictions.append(pred.cpu())  # Moving to CPU during loop

# AFTER:
predictions.append(pred)  # Keep on same device
```

Prevents device transfer overhead and potential numerical issues.

### Fix 4: Documentation & Parameters
- Added `max_var` parameter to `SWAG.__init__()`
- Updated docstrings to explain variance capping
- Added comments explaining numerical stability fixes

## 📊 Testing Strategy

Created `test_swag_quick.py` to evaluate different `max_var` values:
- **max_var=0.5**: Very conservative, minimal perturbation
- **max_var=1.0**: Conservative, recommended starting point
- **max_var=5.0**: Moderate
- **max_var=10.0**: Standard
- **max_var=50.0**: Liberal, closer to original

Each test runs on 5 test samples with 10 SWAG samples per prediction.

## 📈 Expected Outcomes

With proper variance capping:
1. **Dice Score:** Should improve to ~0.74 (matching other methods)
2. **Uncertainty:** Should be valid numbers (no NaN)
3. **Calibration:** Should be reasonable (ECE ~0.96)
4. **Uncertainty Range:** Should be moderate (0.001-0.05 typical)

## 🎯 Next Steps

### Immediate (Testing Phase)
1. ✅ Applied fixes to `src/swag.py`
2. ⏳ Running `test_swag_quick.py` to find optimal `max_var`
3. ⏳ Once optimal `max_var` identified, update evaluation script
4. ⏳ Re-run full evaluation on all 80 test samples

### Short Term (Evaluation)
1. **Update `evaluate_uq_FIXED_v2.py`:**
   ```python
   swag_model = SWAG(base_model, max_num_models=20, max_var=1.0)  # Add max_var
   ```

2. **Re-run evaluation:**
   ```bash
   sbatch scripts/evaluate_uq_v2.sbatch
   ```

3. **Compare results:**
   - SWAG Dice should match Baseline/MC Dropout (~0.74)
   - SWAG uncertainty should be valid and meaningful

### Medium Term (Analysis & Visualization)
1. **Uncertainty Quality Analysis:**
   - Correlation between uncertainty and errors
   - Calibration curves
   - Reliability diagrams
   - Sharpness metrics

2. **Visualization Creation:**
   - Uncertainty maps overlaid on predictions
   - Calibration plots for all methods
   - Error vs Uncertainty scatter plots
   - Method comparison heatmaps

3. **Comprehensive UQ Report:**
   - Detailed metrics for all 4 methods
   - Uncertainty quality assessment
   - Recommendations for each method
   - When to use which method

### Long Term (Improvements)
1. **Retrain SWAG with Better Settings:**
   - Collect snapshots after model fully converged
   - Use cyclic learning rate schedule
   - Reduce snapshot collection frequency
   - Ensure smaller weight deviations

2. **Add Additional UQ Methods:**
   - Temperature Scaling (calibration)
   - Conformal Prediction (distribution-free)
   - Test-Time Augmentation
   - Bayesian Neural Networks (if time permits)

## 📝 Files Modified

- `/scratch/hpl14/uq_capstone/src/swag.py` - Applied all fixes
- `/scratch/hpl14/uq_capstone/src/swag_ORIGINAL.py` - Backup of original
- `/scratch/hpl14/uq_capstone/src/test_swag_quick.py` - Testing script
- `c:\Users\lpnhu\Downloads\uq_capstone\src\swag_FIXED.py` - Local fixed version

## 🐛 Lessons Learned

1. **Always bound variance on both sides** when using variance for sampling
2. **Check for numerical stability** in running average computations
3. **SWAG requires careful tuning:**
   - Start collection only after convergence
   - Use small learning rate during collection
   - Monitor weight deviation magnitudes

4. **Model training matters:**
   - Quality of collected snapshots affects SWAG performance
   - Large deviations → unstable sampling
   - Need balance: enough diversity but not too much

## 📖 References

- **SWAG Paper:** Maddox et al. "A Simple Baseline for Bayesian Uncertainty Estimation in Deep Learning" (NeurIPS 2019)
- **Key Insight:** SWAG approximates posterior as Gaussian, requires snapshots from converged region
- **Common Pitfall:** Collecting snapshots while model still training → large variance

## ✅ Success Criteria

SWAG will be considered "fixed" when:
- ✓ Dice score ≥ 0.70 (within 5% of other methods)
- ✓ Uncertainty values are valid (no NaN/Inf)
- ✓ Uncertainty correlates with prediction errors
- ✓ Calibration is reasonable (ECE < 1.0)
- ✓ Produces diverse but sensible predictions

---

**Status:** Fixes applied, testing in progress. Will update with optimal `max_var` value once testing completes.
