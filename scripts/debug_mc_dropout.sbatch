#!/bin/bash
#SBATCH --job-name=debug_mc_dropout
#SBATCH --output=logs/debug_mc_dropout_%j.out
#SBATCH --error=logs/debug_mc_dropout_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "Debug MC Dropout Evaluation"
echo "============================"

source activate uq_capstone
cd /scratch/$USER/uq_capstone

python scripts/debug_mc_dropout.py
