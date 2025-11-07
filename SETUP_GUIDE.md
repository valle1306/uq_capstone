# Quick Setup Guide - Download and Organize Results

**Created:** November 6, 2025  
**Purpose:** Download all results from Amarel and organize the repository

## 📋 Overview

I've created 4 PowerShell scripts to help you:
1. `preview_changes.ps1` - Preview what will happen
2. `download_results.ps1` - Download all results from Amarel
3. `reorganize_repo.ps1` - Clean up the directory structure
4. `setup_final.ps1` - Do all of the above automatically

## 🚀 Quick Start (Recommended)

### Option A: Do Everything Automatically

```powershell
# Run the master script (does everything)
.\setup_final.ps1
```

This will:
- ✅ Download all metrics, CSVs, and visualizations from Amarel
- ✅ Reorganize docs into logical subfolders
- ✅ Copy results to `results/` for easy access
- ✅ Update README.md with proper structure
- ✅ Commit and push everything to GitHub

### Option B: Step by Step

```powershell
# 1. Preview what will happen
.\preview_changes.ps1

# 2. Download results from Amarel
.\download_results.ps1

# 3. Reorganize the repository
.\reorganize_repo.ps1

# 4. Manually commit and push
git add .
git commit -m "Reorganize repository: move docs to subfolders, copy results for easy access"
git push origin main
```

## 📂 New Repository Structure

After running the scripts, your repo will look like:

```
uq_capstone/
├── README.md                    # Updated with proper overview
├── results/                     # NEW: Easy access to final results
│   ├── metrics/                # Comprehensive metrics JSON & CSV
│   ├── figures/                # Visualization PNG files
│   └── final/                  # Final analysis summary
├── docs/
│   ├── guides/                 # Setup and workflow guides
│   │   ├── QUICK_START_RETRAIN.md
│   │   ├── RETRAINING_COMMANDS.md
│   │   └── ... (all guide docs)
│   ├── status/                 # Status tracking documents
│   │   ├── RETRAINING_STATUS.md
│   │   ├── TRAINING_STATUS.md
│   │   └── ... (all status docs)
│   └── *.md                    # Other documentation
├── runs/classification/        # Original training outputs (unchanged)
├── src/                        # Source code
├── scripts/                    # SLURM scripts
└── analysis/                   # Analysis scripts
```

## 📥 What Gets Downloaded

From Amarel `/scratch/hpl14/uq_capstone/`:

- `runs/classification/metrics/comprehensive_metrics.json`
- `runs/classification/metrics/metrics_summary.csv`
- `runs/classification/metrics/*.png` (visualizations)
- `runs/classification/mc_dropout/config.json`
- `runs/classification/swag_classification/config.json`
- Recent `logs/eval_*.out` files

**Note:** Model checkpoints (`.pth` files) are NOT downloaded (they're large, ~100MB each)

## 📊 What You'll Have After

### Results Ready for Presentation

1. **`results/metrics/metrics_summary.csv`** - Quick summary table
   ```
   Method,Accuracy,ECE,Brier,FNR,Mean Unc
   Baseline,91.67,0.0498,0.0704,0.0833,
   MC Dropout,85.26,0.1171,0.1247,0.1474,8.2e-05
   Deep Ensemble,91.67,0.0285,0.0630,0.0833,0.0167
   SWAG,83.33,0.1518,0.1526,0.1667,9.9e-05
   ```

2. **`results/figures/*.png`** - Ready-to-use visualizations
   - `comprehensive_metrics_visualization.png` - 9-panel dashboard
   - `method_comparisons.png` - Scatter plot comparisons

3. **`results/final/README.md`** - Summary of findings

### Clean Documentation Structure

- All guides in `docs/guides/`
- All status tracking in `docs/status/`
- Main README.md updated with proper overview

## ⚠️ Important Notes

### Before Running

1. **Make sure you're on Amarel VPN or network** (for SCP to work)
2. **Check that Amarel has the latest code:**
   ```bash
   ssh hpl14@amarel.rutgers.edu
   cd /scratch/$USER/uq_capstone
   git pull origin main
   ```

3. **Verify results exist on Amarel:**
   ```bash
   ls -lh /scratch/$USER/uq_capstone/runs/classification/metrics/
   ```

### What If Files Don't Exist?

If `download_results.ps1` shows errors, you need to run the evaluation on Amarel first:

```bash
# On Amarel
cd /scratch/$USER/uq_capstone
git pull origin main
sbatch scripts/eval_and_visualize_on_amarel.sbatch
```

Wait for it to complete (~10-15 minutes), then run the download script again.

## 🔧 Troubleshooting

### SCP Permission Denied
```powershell
# Test SSH connection first
ssh hpl14@amarel.rutgers.edu "ls /scratch/hpl14/uq_capstone"
```

### Files Already Moved
If you run the scripts multiple times, some files might already be moved. That's OK - the scripts handle this gracefully.

### Git Conflicts
If you have uncommitted changes:
```powershell
git status
git add .
git commit -m "Save work before reorganization"
```

## ✅ Verification

After running all scripts, verify:

```powershell
# Check that results are downloaded
ls results\metrics\
ls results\figures\

# Check that docs are organized
ls docs\guides\
ls docs\status\

# Check git status
git status
git log -1
```

You should see:
- ✅ `results/` directory with metrics and figures
- ✅ `docs/guides/` and `docs/status/` with organized docs
- ✅ Updated README.md
- ✅ Everything committed and pushed to GitHub

## 🎯 Next Steps

After running the setup:

1. **Review Results:**
   - Open `results/figures/*.png` to see visualizations
   - Check `results/metrics/metrics_summary.csv`

2. **Prepare Presentation:**
   - Use figures from `results/figures/`
   - Reference metrics from `results/metrics/`
   - Review analysis in `results/final/README.md`

3. **Share Repository:**
   - GitHub repo is now clean and organized
   - Easy for others to find results
   - Professional structure for capstone submission

## 📞 Need Help?

If something goes wrong:
1. Check the output of each script carefully
2. Verify you have SSH access to Amarel
3. Make sure results exist on Amarel before downloading
4. Check that you don't have git conflicts

---

**Ready to go?** Run `.\setup_final.ps1` to do everything automatically! 🚀
