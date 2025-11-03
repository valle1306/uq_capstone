#!/bin/bash
#SBATCH --job-name=investigate_swag
#SBATCH --output=logs/investigate_swag_%j.out
#SBATCH --error=logs/investigate_swag_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "Investigate SWAG Weight Differences"
echo "===================================="

source activate uq_capstone
cd /scratch/$USER/uq_capstone

python scripts/investigate_swag_weights.py
