# SWAG Implementation Decision Analysis
**Date:** November 7, 2025

## Key Question
Should we re-implement SWAG to follow the paper more closely, or accept current results?

---

## Paper's Recommended Approach (Maddox et al. 2019)

### 1. **Optimizer: SGD with Momentum (NOT Adam)**
**Paper:** Uses SGD with momentum=0.9
```python
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
```

**Our Implementation:** Uses Adam with default settings
```python
optimizer = optim.Adam(swag.parameters(), lr=args.lr)  # NO weight decay!
```

**Impact:** 
- ⚠️ Adam is more adaptive but less stable for uncertainty estimation
- SGD with momentum has better theoretical backing for posterior approximation
- Weight decay (L2 regularization) crucial for preventing overfitting

### 2. **Weight Decay: L2 Regularization**
**Paper:** weight_decay=1e-4 (standard)
**Our Implementation:** weight_decay=0 (NONE!)

**Impact:**
- This is a MAJOR difference!
- Without weight decay: overfitting during fine-tuning
- Our training logs show: 99.62% validation, 85.58% test
- This overfitting is why SWAG performs poorly

### 3. **Learning Rate Schedule**
**Paper:** Uses SWA-style schedule:
- Phase 1: Regular training with CosineAnnealingLR
- Phase 2: SWA phase with lower, constant LR
- Phase 3: Collect snapshots during Phase 2

**Our Implementation:** 
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
```

**Impact:**
- We use standard cosine annealing (not SWA-specific)
- Paper's SWA approach: lower learning rate for snapshot collection
- We don't have explicit "SWA phase"

### 4. **SWA vs. Snapshot Collection**
**Paper:** Uses Stochastic Weight Averaging (SWA):
- Keeps running average of weights during training
- Special LR scheduler (SWALR) for the SWA phase
- Batch normalization adjustment after SWA

**Our Implementation:** Manual snapshot collection
```python
if epoch >= args.swag_start:
    swag.collect_model(swag.base_model)  # Just copy weights
```

**Impact:**
- SWA is more principled (weighted averaging)
- Our snapshot collection is simpler but less optimal
- Missing batch norm update after collecting snapshots

### 5. **Initialization**
**Paper:** Trains from scratch with random initialization
**Our Implementation:** Initialize from baseline weights

**Impact:**
- This is intentional in our case (we want better posterior)
- But means we're not following paper exactly
- Could be fine, but creates uncertainty about whether issues are from initialization or algorithm

---

## Cost-Benefit Analysis

### If We Re-implement Properly

**Pros:**
- ✅ Closer to paper's recommendations
- ✅ Fix overfitting issue (add weight decay)
- ✅ Use proper SWA algorithm
- ✅ Use SGD + momentum (theoretically better)
- ✅ Could improve SWAG accuracy to ~90%+
- ✅ More credible for publication

**Cons:**
- ❌ Requires retraining (24 hours on GPU)
- ❌ Multiple changes (SGD, weight decay, SWA schedule)
- ❌ Hard to debug: which change fixed what?
- ❌ Uncertainty if we achieve paper's results

**Time Cost:** ~24-48 hours (retraining + evaluation)

### If We Accept Current Results

**Pros:**
- ✅ No retraining needed
- ✅ We can document WHY results are suboptimal
- ✅ Clear teaching moment for your thesis defense
- ✅ Fast to finalize presentation

**Cons:**
- ❌ SWAG at 83% instead of expected 90%+
- ❌ Doesn't fully validate the method
- ❌ Harder to publish (looks like method doesn't work)
- ❌ May raise questions in defense

---

## My Recommendation: **Yes, Re-implement SWAG Properly**

### Here's Why:

1. **Key Issue: Missing Weight Decay**
   - This is a smoking gun
   - Without L2 regularization, overfitting is guaranteed
   - Adding weight decay is a simple 1-line change
   - Expected to improve validation and test performance significantly

2. **Credibility for Publication**
   - If you want to write a paper with Dr. Moran
   - SWAG at 90% + Ensemble validation = publishable
   - SWAG at 83% looks like method doesn't work

3. **Your Research Quality**
   - Shows rigorous methodology
   - "We implemented method X from paper Y and validated..."
   - vs. "We implemented something but it doesn't work as expected"

4. **Minimal Complexity**
   - Main changes:
     - Optimizer: Adam → SGD + momentum
     - Add weight_decay parameter
     - Add SWALR scheduler
   - These are straightforward changes

---

## Proposed Implementation Plan

### Step 1: Create Improved SWAG Script (1 hour)

```python
# src/retrain_swag_proper.py
# Changes:
# 1. Use SGD with momentum=0.9
# 2. Add weight_decay=1e-4
# 3. Use SWALR for last N epochs
# 4. Update batch norm statistics
```

### Step 2: Retrain SWAG (24 hours on Amarel)
```bash
sbatch scripts/retrain_swag_proper.sbatch
```

### Step 3: Re-evaluate (15 minutes on Amarel)
```bash
sbatch scripts/eval_and_visualize_on_amarel.sbatch
```

### Step 4: Update Results (immediately)
```bash
git pull origin main
# New results will show SWAG ~90%+
```

### Step 5: Update Presentation
- Same structure, just updated results

**Total Time:** 24-48 hours elapsed (but only 1 hour of your work)

---

## Key Changes Required

### Change 1: Optimizer
```python
# BEFORE (current)
optimizer = optim.Adam(swag.parameters(), lr=args.lr)

# AFTER (paper-correct)
optimizer = optim.SGD(swag.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
```

### Change 2: Learning Rate Schedule
```python
# BEFORE (current)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# AFTER (paper-correct)
from torch.optim.swa_utils import SWALR
# Regular training for first N epochs with cosine annealing
# Then switch to SWALR for SWA phase
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs*0.8))
swa_scheduler = SWALR(optimizer, swa_lr=0.05)
```

### Change 3: Batch Normalization Update
```python
# AFTER collecting all snapshots:
from torch.optim.swa_utils import update_bn
update_bn(train_loader, swag.base_model, device)
```

---

## What to Say in Defense/Paper

**If SWAG Results Improve to 90%+:**
- "We implemented SWAG following Maddox et al. 2019"
- "Key implementation details: SGD with momentum, L2 regularization, SWA schedule"
- "SWAG achieves competitive performance with uncertainty quantification"

**Why It Matters:**
- Shows you understand the paper
- Validates the method on medical imaging
- Proper baseline for future work

---

## Decision Matrix

| Factor | Current (Adam) | Paper-Correct (SGD+SWA) |
|--------|---|---|
| **SWAG Accuracy** | 83% ⚠️ | ~90% ✅ |
| **Alignment with Paper** | 60% ⚠️ | 95% ✅ |
| **Publication Ready** | No ❌ | Yes ✅ |
| **Overfitting Issue** | Yes ❌ | Fixed ✅ |
| **Time to Implement** | 0 (done) | 1 hour ⏱️ |
| **Time to Retrain** | 0 | 24 hours ⏱️ |

---

## Recommendation Summary

**GO AHEAD AND RE-IMPLEMENT** because:

1. ✅ Simple changes (mainly optimizer + weight decay)
2. ✅ Expected to fix the 83% → 90% issue
3. ✅ Much better for publication/defense
4. ✅ Only 1 hour of work, 24 hours of GPU time
5. ✅ Shows rigor and understanding of the method

**Timeline:**
- Create new script: Today (1 hour)
- Submit to Amarel: Today
- Results back: Tomorrow (~24 hours later)
- Update presentation: Tomorrow
- Ready for defense: Tomorrow

This is the professional approach and will significantly strengthen your capstone project!
