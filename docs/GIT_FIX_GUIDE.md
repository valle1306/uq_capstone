# Git Fix & Sync Guide

## Problem
You have staged commits with large data files that are stuck and preventing push.

## Solution: Fix Git Issue

### Step 1: Create .gitignore (if not exists)

Open PowerShell in `C:\Users\lpnhu\Downloads\uq_capstone` and run:

```powershell
# Create or update .gitignore
@"
# Data files - too large for git
BraTS2020_TrainingData/
BraTS2020_ValidationData/
data/
*.npz
*.tar.gz
*.nii.gz

# Model checkpoints - too large
runs/
*.pth
*.pt

# Python cache
__pycache__/
*.pyc
*.pyo

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
"@ | Out-File -FilePath .gitignore -Encoding utf8
```

### Step 2: Unstage Everything

```powershell
# Unstage all files
git reset HEAD .

# Or if you want to unstage everything and start fresh:
git reset --soft HEAD~1  # Goes back one commit but keeps changes
```

### Step 3: Remove Large Files from Git Tracking

```powershell
# Remove data directories from git cache (but keep files on disk)
git rm -r --cached BraTS2020_TrainingData/ 2>$null
git rm -r --cached BraTS2020_ValidationData/ 2>$null
git rm -r --cached data/ 2>$null
git rm -r --cached runs/ 2>$null
git rm --cached *.tar.gz 2>$null
```

### Step 4: Stage Only Code Files

```powershell
# Add only code files
git add src/
git add scripts/
git add notebooks/
git add envs/
git add *.md
git add *.py
git add *.txt
git add .gitignore
```

### Step 5: Check Status

```powershell
git status
```

You should see:
- ✅ Small code files staged
- ❌ Large data files ignored

### Step 6: Commit and Push

```powershell
# Commit
git commit -m "Add SWAG fix and UQ analysis - all 4 methods working"

# Push
git push origin main
```

---

## Files to Download from Amarel

After fixing git, download these updated files from Amarel:

### Method 1: Using SCP (from PowerShell)

```powershell
cd C:\Users\lpnhu\Downloads\uq_capstone

# Download fixed SWAG implementation
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/swag.py src/

# Download updated evaluation script
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/evaluate_uq_FIXED_v2.py src/

# Download success report
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/SWAG_FIXED_SUCCESS.md .

# Download updated results
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/evaluation/results.json .

# Download evaluation output log
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/evaluation/eval_v2_47441209.out .
```

### Method 2: Using WinSCP or FileZilla

1. Open WinSCP/FileZilla
2. Connect to: `hpl14@amarel.rutgers.edu`
3. Navigate to: `/scratch/hpl14/uq_capstone`
4. Download these files:
   - `src/swag.py`
   - `src/evaluate_uq_FIXED_v2.py`
   - `SWAG_FIXED_SUCCESS.md`
   - `runs/evaluation/results.json`
   - `runs/evaluation/eval_v2_47441209.out`

---

## Verification Checklist

After syncing:

- [ ] `.gitignore` exists and includes data directories
- [ ] Git status shows no large files staged
- [ ] `src/swag.py` contains `max_var` parameter
- [ ] `results.json` has SWAG Dice = 0.7419
- [ ] All 4 markdown reports exist locally
- [ ] Ready to commit and push code changes

---

## Quick Commands Summary

```powershell
# Navigate to project
cd C:\Users\lpnhu\Downloads\uq_capstone

# Fix git
git reset HEAD .
git rm -r --cached data/ runs/ BraTS* 2>$null
git add src/ scripts/ *.md *.py .gitignore
git commit -m "Add SWAG fix - all 4 UQ methods working"
git push origin main

# Download files from Amarel
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/swag.py src/
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/SWAG_FIXED_SUCCESS.md .
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/evaluation/results.json .
```

---

**Next:** After syncing, we'll proceed with creating the UQ analysis scripts!
