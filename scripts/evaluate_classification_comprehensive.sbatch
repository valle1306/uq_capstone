#!/bin/bash
#SBATCH --job-name=eval_comprehensive_classification
#SBATCH --output=logs/eval_comprehensive_classification_%j.out
#SBATCH --error=logs/eval_comprehensive_classification_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=hpl14@rutgers.edu

# Comprehensive Classification UQ Metrics Evaluation
# Evaluates: Baseline, MC Dropout, Deep Ensemble, SWAG, + Conformal Risk Control
# Computes: Calibration, Uncertainty, Error Analysis, Coverage/Set Size

echo "========================================="
echo "Starting Comprehensive Classification Metrics Evaluation"
echo "========================================="

# Activate conda environment
source activate uq_capstone

# Navigate to project directory
cd /scratch/$USER/uq_capstone || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run comprehensive metrics evaluation
echo ""
echo "Running comprehensive metrics evaluation on GPU..."
echo ""

python src/comprehensive_metrics.py \
    --baseline_path runs/classification/baseline/best_model.pth \
    --mc_dropout_path runs/classification/mc_dropout/best_model.pth \
    --ensemble_dir runs/classification/ensemble \
    --n_ensemble 5 \
    --swag_path runs/classification/swag_classification/swag_model.pth \
    --output_dir runs/classification/metrics \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Comprehensive Metrics Evaluation COMPLETED SUCCESSFULLY"
    echo "========================================="
    echo ""
    echo "Results saved to:"
    echo "  - runs/classification/metrics/comprehensive_metrics.json"
    echo "  - runs/classification/metrics/metrics_summary.csv"
    echo ""
    echo "To view results:"
    echo "  cat runs/classification/metrics/comprehensive_metrics.json"
    echo "  cat runs/classification/metrics/metrics_summary.csv"
    echo ""
else
    echo ""
    echo "========================================="
    echo "ERROR: Comprehensive Metrics Evaluation FAILED"
    echo "========================================="
    exit 1
fi
