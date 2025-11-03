#!/bin/bash
#SBATCH --job-name=clf_ensemble
#SBATCH --output=runs/classification/ensemble/train_all_%j.out
#SBATCH --error=runs/classification/ensemble/train_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[012,018]

# Medical Image Classification - Deep Ensemble Training
# Train 5 independent models with different initializations
# Longer training with 50 epochs each for improved performance

echo "=========================================="
echo "Training Deep Ensemble (5 members x 50 epochs)"
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

# Create output directory
mkdir -p runs/classification/ensemble

# Train each ensemble member
for member_id in 0 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "Training Ensemble Member $member_id"
    echo "=========================================="
    
    python src/train_classifier_ensemble_member.py \
        --dataset chest_xray \
        --data_dir data/chest_xray \
        --arch resnet18 \
        --pretrained \
        --member_id $member_id \
        --epochs 50 \
        --batch_size 32 \
        --lr 0.001 \
        --weight_decay 1e-4 \
        --scheduler cosine \
        --output_dir runs/classification/ensemble \
        --save_freq 10 \
        --device cuda \
        --num_workers 4 \
        --seed 42
    
    echo "✓ Member $member_id complete!"
done

echo ""
echo "=========================================="
echo "All Ensemble Members Trained!"
echo "End time: $(date)"
echo "=========================================="
