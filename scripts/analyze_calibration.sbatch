#!/bin/bash
#SBATCH --job-name=cal_analysis    # Job name
#SBATCH --partition=gpu             # Partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=4           # CPU cores
#SBATCH --mem=16GB                  # Memory
#SBATCH --time=00:30:00             # 30 minutes
#SBATCH --output=logs/cal_analysis_%j.out
#SBATCH --error=logs/cal_analysis_%j.err

# Analyze Conformal Calibration Set Size

echo "========================================"
echo "Conformal Calibration Analysis"
echo "Start time: $(date)"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

# Navigate to project directory
cd /scratch/$USER/uq_capstone

# Create output directory
mkdir -p runs/classification/calibration_analysis

echo ""
echo "Analyzing calibration set size for all models..."
echo ""

# Analyze baseline
echo "1. Baseline Model"
python src/analyze_conformal_calibration.py \
    --dataset chest_xray \
    --model_path runs/classification/baseline_model.pth \
    --arch resnet18 \
    --output_dir runs/classification/calibration_analysis/baseline \
    --alpha 0.1 \
    --n_trials 10 \
    --batch_size 32

echo ""
echo "2. MC Dropout Model"
python src/analyze_conformal_calibration.py \
    --dataset chest_xray \
    --model_path runs/classification/mc_dropout_model.pth \
    --arch resnet18 \
    --output_dir runs/classification/calibration_analysis/mc_dropout \
    --alpha 0.1 \
    --n_trials 10 \
    --batch_size 32

echo ""
echo "3. Deep Ensemble Model"
python src/analyze_conformal_calibration.py \
    --dataset chest_xray \
    --model_path runs/classification/ensemble_model_0.pth \
    --arch resnet18 \
    --output_dir runs/classification/calibration_analysis/ensemble \
    --alpha 0.1 \
    --n_trials 10 \
    --batch_size 32

echo ""
echo "========================================"
echo "Analysis Complete!"
echo "End time: $(date)"
echo "========================================"

# Summary
echo ""
echo "Summary of calibration set analysis:"
echo ""
for method in baseline mc_dropout ensemble; do
    if [ -f "runs/classification/calibration_analysis/$method/calibration_size_analysis.json" ]; then
        echo "=== $method ===" 
        python -c "
import json
with open('runs/classification/calibration_analysis/$method/calibration_size_analysis.json') as f:
    res = json.load(f)
    curr_size = res['total_cal_samples']
    curr_res = res['results'][str(curr_size)]
    print(f\"Current cal set: {curr_size} samples\")
    print(f\"Target coverage: {100*res['target_coverage']:.1f}%\")
    print(f\"Empirical coverage: {100*curr_res['mean_coverage']:.2f}% ± {100*curr_res['std_coverage']:.2f}%\")
    print(f\"Range: [{100*curr_res['min_coverage']:.2f}%, {100*curr_res['max_coverage']:.2f}%]\")
    print()
"
    fi
done

echo "Plots and detailed results saved to runs/classification/calibration_analysis/"
