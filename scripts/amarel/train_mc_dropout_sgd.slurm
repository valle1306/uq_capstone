#!/bin/bash
#SBATCH --job-name=mc_dropout_sgd
#SBATCH --output=logs/mc_dropout_sgd_%j.out
#SBATCH --error=logs/mc_dropout_sgd_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "========================================="
echo "MC Dropout Training with SGD"
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
python src/train_mc_dropout_sgd.py \
    --dataset chest_xray \
    --data_dir data/chest_xray \
    --arch resnet18 \
    --pretrained \
    --epochs 50 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 0.01 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --dropout_rate 0.2 \
    --output_dir runs/classification/mc_dropout_sgd \
    --device cuda \
    --seed 1

echo "========================================="
echo "Training completed at: $(date)"
echo "========================================="
