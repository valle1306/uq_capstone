#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=runs/baseline/train_%j.out
#SBATCH --error=runs/baseline/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "========================================="
echo "BASELINE TRAINING"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Setup
export PROJECT_ROOT=/scratch/hpl14/uq_capstone
cd $PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

module load cuda/12.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uq_capstone

# Environment info
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Create output directory
mkdir -p runs/baseline

# Train baseline model
python src/train_baseline_FIXED.py \
    --data_dir /scratch/hpl14/uq_capstone/data/brats_subset_npz \
    --output_dir runs/baseline \
    --epochs 30 \
    --batch 8 \
    --lr 1e-3 \
    --in_ch 1 \
    --device cuda

echo "End time: $(date)"
