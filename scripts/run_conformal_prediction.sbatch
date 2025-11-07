#!/bin/bash
#SBATCH --job-name=conformal      # Job name
#SBATCH --partition=gpu            # Partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=4          # CPU cores
#SBATCH --mem=16GB                 # Memory
#SBATCH --time=01:00:00            # 1 hour (quick!)
#SBATCH --output=logs/conformal_%j.out
#SBATCH --error=logs/conformal_%j.err

# Conformal Prediction for All Models

echo "========================================"
echo "Conformal Prediction Evaluation"
echo "Start time: $(date)"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

# Navigate to project directory
cd /scratch/$USER/uq_capstone

# Create output directory
mkdir -p runs/classification/conformal

echo ""
echo "Running conformal prediction for all models..."
echo ""

# 1. Baseline Model
echo "1. Baseline Model"
python src/conformal_prediction.py \
    --dataset chest_xray \
    --model_path runs/classification/baseline_model.pth \
    --arch resnet18 \
    --output_dir runs/classification/conformal/baseline \
    --alpha 0.1 \
    --batch_size 32 \
    --num_workers 4

echo ""
echo "2. MC Dropout Model"
python src/conformal_prediction.py \
    --dataset chest_xray \
    --model_path runs/classification/mc_dropout_model.pth \
    --arch resnet18 \
    --output_dir runs/classification/conformal/mc_dropout \
    --alpha 0.1 \
    --batch_size 32 \
    --num_workers 4

echo ""
echo "3. Deep Ensemble (Model 0)"
python src/conformal_prediction.py \
    --dataset chest_xray \
    --model_path runs/classification/ensemble_model_0.pth \
    --arch resnet18 \
    --output_dir runs/classification/conformal/ensemble \
    --alpha 0.1 \
    --batch_size 32 \
    --num_workers 4

echo ""
echo "4. SWAG Model (best checkpoint)"
python src/conformal_prediction.py \
    --dataset chest_xray \
    --model_path runs/classification/swag_classification/best_model.pth \
    --arch resnet18 \
    --output_dir runs/classification/conformal/swag \
    --alpha 0.1 \
    --batch_size 32 \
    --num_workers 4

echo ""
echo "========================================"
echo "Conformal Prediction Complete!"
echo "End time: $(date)"
echo "========================================"

# Show results summary
echo ""
echo "Results summary:"
for method in baseline mc_dropout ensemble swag; do
    if [ -f "runs/classification/conformal/$method/conformal_prediction_results.json" ]; then
        echo ""
        echo "=== $method ==="
        python -c "
import json
with open('runs/classification/conformal/$method/conformal_prediction_results.json') as f:
    res = json.load(f)
    pr = res['primary_results']
    print(f\"Target Coverage: {100*pr['target_coverage']:.1f}%\")
    print(f\"Empirical Coverage: {100*pr['empirical_coverage']:.2f}%\")
    print(f\"Avg Set Size: {pr['avg_set_size']:.2f}\")
    print(f\"Singleton Accuracy: {100*pr['singleton_accuracy']:.2f}%\")
"
    fi
done

echo ""
echo "All conformal prediction results saved to runs/classification/conformal/"
