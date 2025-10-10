#!/bin/bash
#SBATCH --job-name=eval_uq
#SBATCH --output=runs/evaluation/eval_%j.out
#SBATCH --error=runs/evaluation/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "========================================="
echo "UQ EVALUATION"
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

# Install matplotlib if not already installed
pip install matplotlib --quiet

# Create output directory
mkdir -p runs/evaluation

# Run evaluation
python src/evaluate_uq.py \
    --data_root /scratch/hpl14/uq_capstone/data/brats_subset_npz \
    --baseline_model runs/baseline/best_model.pth \
    --mc_dropout_model runs/mc_dropout/best_model.pth \
    --ensemble_dir runs/ensemble \
    --swag_model runs/swag/swag_model.pth \
    --num_ensemble 5 \
    --batch 8 \
    --in_ch 1 \
    --mc_samples 20 \
    --swag_samples 30 \
    --dropout_rate 0.2 \
    --save_dir runs/evaluation

echo "End time: $(date)"
echo "========================================="
echo "Check results at: runs/evaluation/results.json"
echo "Check plots at: runs/evaluation/comparison.png"
echo "========================================="
