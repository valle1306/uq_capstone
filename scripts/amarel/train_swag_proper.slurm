#!/bin/bash
#SBATCH --job-name=swag_proper
#SBATCH --output=logs/swag_proper_%j.out
#SBATCH --error=logs/swag_proper_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# SWAG Training for Chest X-Ray Classification
# Based on Maddox et al. "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (NeurIPS 2019)

echo "========================================="
echo "SWAG Training (Proper Implementation)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Load modules and activate conda
module purge
module load conda
conda activate uq_capstone

# Navigate to project directory
cd /scratch/hpl14/uq_capstone

# Check GPU
nvidia-smi

# Run SWAG training
python src/train_swag_proper.py \
    --dataset chest_xray \
    --data_dir data/chest_xray \
    --arch resnet18 \
    --pretrained \
    --epochs 50 \
    --batch_size 32 \
    --num_workers 4 \
    --lr_init 0.01 \
    --swa_lr 0.005 \
    --swa_start 27 \
    --swa_c_epochs 1 \
    --max_num_models 20 \
    --momentum 0.9 \
    --wd 1e-4 \
    --output_dir runs/classification/swag_proper \
    --device cuda \
    --seed 1

echo "========================================="
echo "Training completed at: $(date)"
echo "========================================="
