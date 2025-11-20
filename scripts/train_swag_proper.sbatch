#!/bin/bash
#SBATCH --job-name=swag_proper       # Job name
#SBATCH --partition=gpu              # Partition (queue)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=8            # CPU cores per task
#SBATCH --mem=32GB                   # Memory per node
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --time=12:00:00              # Time limit (12 hours)
#SBATCH --output=/scratch/hpl14/uq_capstone/logs/swag_proper_%j.out
#SBATCH --error=/scratch/hpl14/uq_capstone/logs/swag_proper_%j.err

# SWAG Training Based on Maddox et al. (2019)
# Using exact hyperparameters from wjmaddox/swa_gaussian repository

echo "=========================================="
echo "SWAG Training (Proper Implementation)"
echo "Based on: https://github.com/wjmaddox/swa_gaussian"
echo "Paper: arxiv.org/abs/1902.02476"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

cd /scratch/hpl14/uq_capstone

# Activate conda environment
module purge
module load python/3.9
source ~/.bashrc
conda activate uq_capstone

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Train SWAG with SGD (from original paper)
echo "Training SWAG with SGD optimizer..."
echo "Key hyperparameters:"
echo "  - Optimizer: SGD(lr=0.01, momentum=0.9, wd=1e-4)"
echo "  - Epochs: 50 (scaled from 300 in paper)"
echo "  - SWAG start: epoch 27 (54% of training, like 161/300)"
echo "  - SWAG LR: 0.005"
echo "  - Collection: every epoch after start"
echo "  - Max models: 20"
echo ""

python src/train_swag_proper.py \
    --dataset chest_xray \
    --batch_size 32 \
    --num_workers 8 \
    --arch resnet18 \
    --pretrained \
    --epochs 50 \
    --lr_init 0.01 \
    --momentum 0.9 \
    --wd 1e-4 \
    --swa_start 27 \
    --swa_lr 0.005 \
    --swa_c_epochs 1 \
    --max_num_models 20 \
    --output_dir runs/classification/swag_proper \
    --device cuda \
    --seed 42

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Results saved to: runs/classification/swag_proper/"
echo "=========================================="
