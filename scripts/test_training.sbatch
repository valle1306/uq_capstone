#!/bin/bash
#SBATCH --job-name=brats_test          # Job name
#SBATCH --output=runs/test_%j.out      # Output file (%j = job ID)
#SBATCH --error=runs/test_%j.err       # Error file
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=4              # CPUs per task
#SBATCH --mem=16G                      # Memory per node
#SBATCH --time=02:00:00                # Time limit (2 hours)
#SBATCH --partition=main               # Partition name
#SBATCH --gres=gpu:1                   # Request 1 GPU

# BRATS Training Test Script for Amarel
# This is a quick test to verify data loading and model training

echo "========================================="
echo "BRATS Training Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Load modules
module purge
module load conda
module load cuda/11.8

# Activate conda environment
source activate uq_capstone

# Print environment info
echo "Python version:"
python --version
echo ""

echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Verify data
echo "Verifying data..."
python scripts/validate_brats_data.py --data_root data/brats --n_samples 3
if [ $? -ne 0 ]; then
    echo "ERROR: Data validation failed!"
    exit 1
fi
echo ""

# Check data statistics
echo "Data statistics:"
python -c "
import csv
for split in ['train', 'val', 'test']:
    with open(f'data/brats/{split}.csv') as f:
        n = sum(1 for _ in f) - 1  # subtract header
        print(f'  {split}: {n} samples')
"
echo ""

# Quick training test (just a few iterations to verify everything works)
echo "Running quick training test..."
python src/train_seg.py \
    --data_root data/brats \
    --epochs 2 \
    --batch_size 8 \
    --lr 1e-4 \
    --model_name test_run \
    --save_dir runs/test_${SLURM_JOB_ID}

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Test completed successfully!"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ Test failed!"
    echo "========================================="
    exit 1
fi

echo "End time: $(date)"
