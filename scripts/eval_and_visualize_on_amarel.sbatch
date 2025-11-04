#!/bin/bash
#SBATCH --job-name=eval_visualize_comprehensive
#SBATCH --output=logs/eval_visualize_comprehensive_%j.out
#SBATCH --error=logs/eval_visualize_comprehensive_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=hpl14@rutgers.edu

echo "========================================="
echo "Starting Comprehensive Metrics + Visualizations"
echo "Running on Amarel GPU directly"
echo "========================================="

# Activate conda environment
source activate uq_capstone

# Navigate to project directory
cd /scratch/$USER/uq_capstone || exit 1

# Create necessary directories
mkdir -p logs runs/classification/metrics

echo ""
echo "========================================="
echo "STEP 1: Verifying Models"
echo "========================================="
echo ""

# Verify all models exist
if [ ! -f "runs/classification/baseline/best_model.pth" ]; then
    echo "ERROR: Baseline model not found!"
    exit 1
fi
echo "✓ Baseline model exists"

if [ ! -f "runs/classification/mc_dropout/best_model.pth" ]; then
    echo "ERROR: MC Dropout model not found!"
    exit 1
fi
echo "✓ MC Dropout model exists"

if [ ! -f "runs/classification/swag_classification/swag_model.pth" ]; then
    echo "ERROR: SWAG model not found!"
    exit 1
fi
echo "✓ SWAG model exists"

if [ ! -d "runs/classification/ensemble" ]; then
    echo "ERROR: Ensemble directory not found!"
    exit 1
fi
echo "✓ Ensemble directory exists"

# Verify CUDA
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "========================================="
echo "STEP 2: Running Comprehensive Metrics Evaluation"
echo "========================================="
echo ""

python src/comprehensive_metrics.py \
    --baseline_path runs/classification/baseline/best_model.pth \
    --mc_dropout_path runs/classification/mc_dropout/best_model.pth \
    --ensemble_dir runs/classification/ensemble \
    --n_ensemble 5 \
    --swag_path runs/classification/swag_classification/swag_model.pth \
    --output_dir runs/classification/metrics \
    --device cuda

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Comprehensive metrics evaluation FAILED!"
    exit 1
fi

echo ""
echo "✓ Comprehensive metrics evaluation completed"

# Check if results were generated
if [ -f "runs/classification/metrics/comprehensive_metrics.json" ]; then
    echo "✓ Results file generated: runs/classification/metrics/comprehensive_metrics.json"
    echo ""
    echo "Sample results (first 50 lines):"
    head -50 runs/classification/metrics/comprehensive_metrics.json
else
    echo "WARNING: Results file not found at expected location"
fi

echo ""
echo "========================================="
echo "STEP 3: Generating Visualizations"
echo "========================================="
echo ""

python analysis/visualize_metrics.py \
    --metrics_path runs/classification/metrics/comprehensive_metrics.json \
    --output_dir runs/classification/metrics

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Visualization generation FAILED!"
    exit 1
fi

echo ""
echo "✓ Visualizations generated"

# Check generated files
echo ""
echo "Generated output files:"
find runs/classification/metrics -type f \( -name "*.json" -o -name "*.csv" -o -name "*.png" \) -exec ls -lh {} \;

echo ""
echo "========================================="
echo "STEP 4: Creating Summary Report"
echo "========================================="
echo ""

# Create a summary report
cat > runs/classification/metrics/EVALUATION_REPORT.txt << 'REPORT'
COMPREHENSIVE METRICS EVALUATION REPORT
Generated on Amarel HPC

Models Evaluated:
1. Baseline (ResNet18, pretrained, fine-tuned)
2. MC Dropout (T=20 sampling, dropout_rate=0.2, retrained from baseline)
3. Deep Ensemble (5 members, retrained from baseline)
4. SWAG (20 snapshots, scale=0.5, retrained from baseline)
5. Conformal Risk Control (post-hoc, multiple loss functions)

Metrics Computed:
- Accuracy, Precision, Recall, F1 Score
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- False Positive Rate (FPR), False Negative Rate (FNR)
- ROC-AUC
- Uncertainty Separation (correct vs incorrect predictions)
- Conformal Coverage & Set Size
- Risk Bounds

Output Files:
- comprehensive_metrics.json (full results in JSON format)
- metrics_summary.csv (tabular summary)
- accuracy_comparison.png (accuracy by method)
- calibration_curves.png (calibration analysis)
- uncertainty_distributions.png (uncertainty histograms)
- roc_curves.png (ROC curves for each method)
- conformal_analysis.png (CRC coverage and set size)

Next Steps:
1. Review results in comprehensive_metrics.json
2. Check if MC Dropout and SWAG accuracies are ~90%
3. Verify uncertainty metrics show proper separation
4. Pull visualization files to local machine if results look good
5. Run final analysis and create presentation

Command to View Results:
  cat runs/classification/metrics/comprehensive_metrics.json | python -m json.tool | head -100

Command to Pull Results to Local (Windows):
  scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics ./runs/

REPORT

echo "✓ Summary report created at runs/classification/metrics/EVALUATION_REPORT.txt"

echo ""
echo "========================================="
echo "✅ EVALUATION COMPLETE"
echo "========================================="
echo ""
echo "Results Location: /scratch/$USER/uq_capstone/runs/classification/metrics/"
echo ""
echo "Next: Review results and pull if they look good!"
echo ""
