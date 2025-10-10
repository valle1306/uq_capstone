#!/bin/bash
# Quick diagnostic script to check what went wrong

echo "=========================================="
echo "🔍 DIAGNOSTIC CHECK"
echo "=========================================="
echo ""

# Check if we're on Amarel
echo "Hostname: $HOSTNAME"
echo "User: $USER"
echo "Working directory: $(pwd)"
echo ""

# Check job queue
echo "Current job queue:"
squeue -u hpl14
echo ""

# Check if training jobs were submitted
echo "Checking recent SLURM job history (last 10 jobs):"
sacct -u hpl14 -S today -o JobID,JobName,State,ExitCode,Elapsed -n | head -10
echo ""

# Check if output directories exist
echo "Checking output directories:"
ls -ld runs/ 2>/dev/null || echo "❌ runs/ directory doesn't exist"
ls -ld runs/baseline/ 2>/dev/null || echo "❌ runs/baseline/ doesn't exist"
ls -ld runs/mc_dropout/ 2>/dev/null || echo "❌ runs/mc_dropout/ doesn't exist"
ls -ld runs/ensemble/ 2>/dev/null || echo "❌ runs/ensemble/ doesn't exist"
ls -ld runs/swag/ 2>/dev/null || echo "❌ runs/swag/ doesn't exist"
echo ""

# Check if Python files exist
echo "Checking if new UQ files exist:"
ls -lh src/uq_methods.py 2>/dev/null || echo "❌ uq_methods.py missing"
ls -lh src/swag.py 2>/dev/null || echo "❌ swag.py missing (needs upload)"
ls -lh src/train_swag.py 2>/dev/null || echo "❌ train_swag.py missing (needs upload)"
ls -lh src/train_baseline.py 2>/dev/null || echo "❌ train_baseline.py missing"
echo ""

# Check if SLURM scripts exist
echo "Checking if SLURM scripts exist:"
ls -lh scripts/train_baseline.sbatch 2>/dev/null || echo "❌ train_baseline.sbatch missing"
ls -lh scripts/train_swag.sbatch 2>/dev/null || echo "❌ train_swag.sbatch missing (needs upload)"
ls -lh scripts/run_all_experiments.sh 2>/dev/null || echo "❌ run_all_experiments.sh missing"
echo ""

echo "=========================================="
echo "💡 DIAGNOSIS"
echo "=========================================="

if [ ! -f "src/swag.py" ]; then
    echo "🔴 SWAG files NOT uploaded yet"
    echo "   Action: Run upload script from Windows"
    echo "   Command: scripts\upload_uq_code.bat"
fi

if [ -z "$(squeue -u hpl14 | grep -v JOBID)" ]; then
    echo "🔴 No jobs running"
    echo "   Action: Submit jobs with bash scripts/run_all_experiments.sh"
fi

echo ""
