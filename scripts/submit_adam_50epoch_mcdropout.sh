#!/bin/bash
#SBATCH --job-name=mcdropout_adam_50
#SBATCH --output=/scratch/hpl14/uq_capstone/logs/mcdropout_adam_50_%j.out
#SBATCH --error=/scratch/hpl14/uq_capstone/logs/mcdropout_adam_50_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

# Load modules
module purge
module load cuda/11.8.0
module load python/3.9.6

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Change to project directory
cd /scratch/hpl14/uq_capstone

# Run MC Dropout training
python src/train_mc_dropout_adam.py

echo "MC Dropout Adam 50-epoch training completed"
