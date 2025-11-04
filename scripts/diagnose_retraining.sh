#!/bin/bash
# Diagnostic script to check what happened with retraining

cd /scratch/$USER/uq_capstone

echo "========================================="
echo "DIAGNOSTIC: Checking Retraining Status"
echo "========================================="
echo ""

# 1. Check job queue
echo "1. CURRENT JOBS:"
squeue -u hpl14
echo ""

# 2. Check log files
echo "2. AVAILABLE LOG FILES:"
ls -lh logs/retrain*.out logs/retrain*.err 2>/dev/null || echo "No retrain log files found"
echo ""

# 3. Check what exists in runs/classification
echo "3. DIRECTORY STRUCTURE:"
find runs/classification -maxdepth 2 -type f -name "best_model.pth" -o -name "swag_model.pth" 2>/dev/null | head -20
echo ""

# 4. Check if old models exist (backup)
echo "4. BACKUP MODELS:"
ls -lh runs/classification/mc_dropout_old/ 2>/dev/null || echo "No MC Dropout old backup"
ls -lh runs/classification/swag_classification_old/ 2>/dev/null || echo "No SWAG old backup"
echo ""

# 5. Check git status
echo "5. GIT STATUS (latest code):"
git log --oneline -3
echo ""

# 6. Check baseline exists
echo "6. BASELINE MODEL:"
ls -lh runs/classification/baseline/best_model.pth
echo ""

# 7. Check if retrain scripts exist
echo "7. RETRAIN SCRIPTS:"
ls -lh src/retrain*.py
echo ""

# 8. Check SBATCH scripts
echo "8. SBATCH SCRIPTS:"
ls -lh scripts/retrain*.sbatch
echo ""

echo "========================================="
echo "END DIAGNOSTIC"
echo "========================================="
