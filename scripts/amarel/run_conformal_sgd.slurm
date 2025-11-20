#!/bin/bash
#SBATCH --job-name=conformal_sgd
#SBATCH --output=logs/conformal_sgd_%j.out
#SBATCH --error=logs/conformal_sgd_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Run Conformal Prediction on all completed SGD 50-epoch experiments
# Applies conformal risk control to: baseline, MC dropout, ensemble

echo "========================================"
echo "Conformal Prediction - SGD Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

# Navigate to project root
cd /scratch/hpl14/uq_capstone

echo "Python: $(which python)"
echo "========================================"

# Run conformal prediction for each completed experiment
echo ""
echo "=== Running Conformal Prediction for Baseline SGD ==="
python src/conformal_risk_control_proper.py \
    --model_dir runs/classification/baseline_sgd \
    --model_path runs/classification/baseline_sgd/best_model.pth \
    --output_dir runs/classification/baseline_sgd/conformal \
    --alpha 0.1 \
    --batch_size 32

echo ""
echo "=== Running Conformal Prediction for MC Dropout SGD ==="
python src/conformal_risk_control_proper.py \
    --model_dir runs/classification/mc_dropout_sgd \
    --model_path runs/classification/mc_dropout_sgd/best_model.pth \
    --output_dir runs/classification/mc_dropout_sgd/conformal \
    --alpha 0.1 \
    --batch_size 32 \
    --dropout 0.2 \
    --mc_samples 30

echo ""
echo "=== Running Conformal Prediction for SWAG SGD ==="
python src/conformal_risk_control_proper.py \
    --model_dir runs/classification/swag_proper \
    --model_path runs/classification/swag_proper/best_swag_model.pth \
    --output_dir runs/classification/swag_proper/conformal \
    --alpha 0.1 \
    --batch_size 32 \
    --swag \
    --num_samples 30

echo ""
echo "=== Running Conformal Prediction for Ensemble SGD ==="
python src/conformal_risk_control_proper.py \
    --model_dir runs/classification/ensemble_sgd \
    --ensemble_dir runs/classification/ensemble_sgd \
    --output_dir runs/classification/ensemble_sgd/conformal \
    --alpha 0.1 \
    --batch_size 32 \
    --num_members 5

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
