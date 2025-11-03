#!/bin/bash
#SBATCH --job-name=clf_swag
#SBATCH --output=runs/classification/swag_classification/train_%j.out
#SBATCH --error=runs/classification/swag_classification/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Medical Image Classification - SWAG Training
# Stochastic Weight Averaging-Gaussian for Bayesian uncertainty quantification
# 50 epochs with SWAG snapshot collection starting at epoch 30

echo "=========================================="
echo "Training SWAG Classifier (50 epochs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Navigate to project directory
cd /scratch/$USER/uq_capstone || exit 1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uq_capstone

# Set PYTHONPATH
export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH

# Print environment info
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Create output directory
mkdir -p runs/classification/swag_classification

# Run SWAG training with 50 epochs, snapshot collection starts at epoch 30
python src/train_swag_classification.py \
    --dataset chest_xray \
    --data_dir data/chest_xray \
    --arch resnet18 \
    --epochs 50 \
    --swag_start 30 \
    --swag_lr 0.0001 \
    --lr 0.001 \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda \
    --max_models 20 \
    --output_dir runs/classification/swag_classification

echo ""
echo "=========================================="
echo "SWAG Training Complete!"
echo "End time: $(date)"
echo "=========================================="

