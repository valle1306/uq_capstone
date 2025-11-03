#!/bin/bash
#SBATCH --job-name=clf_eval
#SBATCH --output=runs/classification/evaluation/eval_%j.out
#SBATCH --error=runs/classification/evaluation/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Medical Image Classification - Comprehensive UQ Evaluation
# Evaluates all methods: Baseline, MC Dropout, Deep Ensemble, SWAG, and Conformal Risk Control

echo "=========================================="
echo "Comprehensive UQ Evaluation (All Methods)"
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

# Create output directory
mkdir -p runs/classification/evaluation

# Run comprehensive evaluation with all methods
python src/evaluate_uq_classification.py \
    --dataset chest_xray \
    --data_dir data/chest_xray \
    --arch resnet18 \
    --baseline_path runs/classification/baseline/best_model.pth \
    --mc_dropout_path runs/classification/mc_dropout/best_model.pth \
    --ensemble_dir runs/classification/ensemble \
    --n_ensemble 5 \
    --swag_path runs/classification/swag_classification/swag_model.pth \
    --swag_samples 30 \
    --mc_samples 20 \
    --dropout_rate 0.3 \
    --batch_size 32 \
    --num_workers 4 \
    --output_dir runs/classification/evaluation \
    --device cuda

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "End time: $(date)"
echo "=========================================="

# Display results
if [ -f runs/classification/evaluation/all_results.json ]; then
    echo ""
    echo "Results Summary:"
    cat runs/classification/evaluation/all_results.json
fi
