# Sync Status & Next Steps

## ✅ Local Files Updated

The following files have been updated locally with SWAG fixes:

### Updated Files
1. **`src/swag.py`** ✅
   - Added `max_var` parameter (default=100.0)
   - Fixed variance clamping: `torch.clamp(var, min, max)`
   - Fixed K calculation to use actual snapshots
   - Fixed device consistency in predictions

2. **`GIT_FIX_GUIDE.md`** ✅
   - Complete guide to fix git issues with large files
   - Step-by-step unstaging instructions
   - .gitignore template

3. **`download_from_amarel.ps1`** ✅  
   - PowerShell script to download files from Amarel

### Files to Download from Amarel

You still need to download these files from Amarel (they exist on server):

```powershell
# Option 1: Download the package
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/swag_fix_package.tar.gz .

# Option 2: Download individually
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/evaluate_uq_FIXED_v2.py src/
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/SWAG_FIXED_SUCCESS.md .
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/evaluation/results.json .
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/evaluation/eval_v2_47441209.out .
```

---

## 🔧 Git Fix Steps

### Quick Fix (Recommended)

Open **PowerShell** in `C:\Users\lpnhu\Downloads\uq_capstone` and run:

```powershell
# 1. Create .gitignore
@"
# Data files
BraTS2020_TrainingData/
BraTS2020_ValidationData/
data/
*.npz
*.tar.gz
*.nii.gz

# Model checkpoints
runs/
*.pth
*.pt

# Python
__pycache__/
*.pyc

# IDE
.vscode/
.idea/
"@ | Out-File -FilePath .gitignore -Encoding utf8

# 2. Unstage everything
git reset HEAD .

# 3. Remove large files from git tracking
git rm -r --cached BraTS2020_TrainingData/ 2>$null
git rm -r --cached BraTS2020_ValidationData/ 2>$null
git rm -r --cached data/ 2>$null
git rm -r --cached runs/ 2>$null
git rm --cached *.tar.gz 2>$null

# 4. Check status
git status

# 5. Add only code files
git add src/ scripts/ notebooks/ envs/ *.md *.py *.txt .gitignore

# 6. Commit
git commit -m "Add SWAG fix - all 4 UQ methods working (Dice: Ensemble=0.755, SWAG=0.742, MC=0.740, Baseline=0.740)"

# 7. Push
git push origin main
```

---

## 📊 Current Status Summary

### SWAG Fix Complete ✅
- **Before:** Dice = 0.14, Uncertainty = NaN
- **After:** Dice = 0.74, Uncertainty = 0.0026
- **Fix:** Added `max_var=1.0` to cap variance

### All Methods Working ✅

| Method | Dice | ECE | Uncertainty |
|--------|------|-----|-------------|
| Deep Ensemble 🥇 | 0.7550 | 0.9589 | 0.0158 |
| SWAG 🥈 | 0.7419 | 0.9656 | 0.0026 |
| MC Dropout 🥉 | 0.7403 | 0.9663 | 0.0011 |
| Baseline | 0.7401 | 0.9673 | N/A |

---

## 🎯 Next Steps (After Sync)

### Step 1: Analyze UQ Quality Metrics
Create `src/analyze_uq_metrics.py` to compute:
- Calibration metrics (ECE, MCE, Brier Score)
- Uncertainty-error correlation (Pearson, Spearman, AUROC)
- Reliability curves
- Answer: **Which method has the best uncertainty?**

### Step 2: Create Visualizations
Create `src/visualize_uq.py` to generate:
- Uncertainty maps (show where models are uncertain)
- Calibration plots
- Uncertainty-error scatter plots  
- ROC curves for error detection
- Method comparison radar charts

### Step 3: Generate Comprehensive Report
Create `src/generate_uq_report.py` to produce:
- Executive summary
- Detailed performance analysis
- Method-by-method breakdown
- Use case recommendations
- All metrics and visualizations

---

## 📝 Todo Checklist

### Git & Sync
- [ ] Create/update `.gitignore`
- [ ] Unstage large files
- [ ] Download files from Amarel
- [ ] Verify `src/swag.py` has `max_var` parameter
- [ ] Verify `results.json` has SWAG Dice = 0.7419
- [ ] Commit and push code changes

### UQ Analysis (After Sync)
- [ ] Create `analyze_uq_metrics.py`
- [ ] Run analysis on Amarel
- [ ] Create `visualize_uq.py`
- [ ] Generate all visualizations
- [ ] Create `generate_uq_report.py`
- [ ] Produce final comprehensive report

---

## 💡 Pro Tips

1. **Git Issue:** If you're still stuck, you can also do:
   ```powershell
   # Nuclear option: start fresh
   git checkout main
   git pull origin main
   git reset --hard origin/main
   # Then re-apply your changes
   ```

2. **File Download:** If SCP is slow, use FileZilla or WinSCP GUI

3. **Verification:** After downloading, check:
   ```powershell
   # Check SWAG has the fix
   Select-String "max_var" src\swag.py
   
   # Check results
   Get-Content results.json | ConvertFrom-Json | Where-Object {$_.method -eq "SWAG"}
   ```

---

**Ready to proceed?** Once you've fixed git and downloaded the files, let me know and we'll create the UQ analysis scripts! 🚀
