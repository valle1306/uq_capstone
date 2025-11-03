#!/bin/bash
#SBATCH --job-name=clf_baseline
#SBATCH --output=runs/classification/baseline/train_%j.out
#SBATCH --error=runs/classification/baseline/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Medical Image Classification - Baseline Training
# Standard ResNet-18 classifier without uncertainty quantification
# Longer training with 50 epochs for improved performance

echo "=========================================="
echo "Training Baseline Classifier (50 epochs)"
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
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run training with 50 epochs
python src/train_classifier_baseline.py \
    --dataset chest_xray \
    --data_dir data/chest_xray \
    --arch resnet18 \
    --pretrained \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --output_dir runs/classification/baseline \
    --save_freq 10 \
    --device cuda \
    --num_workers 4 \
    --seed 42

echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
