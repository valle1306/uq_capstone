#!/bin/bash
#SBATCH --job-name=ensemble_sgd
#SBATCH --output=logs/ensemble_sgd_%j.out
#SBATCH --error=logs/ensemble_sgd_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "========================================="
echo "Deep Ensemble Training with SGD (5 members)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

# Navigate to project directory
cd /scratch/hpl14/uq_capstone

# Check GPU
nvidia-smi

# Run training
python src/train_ensemble_sgd.py \
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
    --momentum 0.9 \
    --wd 1e-4 \
    --num_models 5 \
    --output_dir runs/classification/ensemble_sgd \
    --device cuda

echo "========================================="
echo "Training completed at: $(date)"
echo "========================================="
