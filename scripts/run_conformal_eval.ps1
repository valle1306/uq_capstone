# Run the full UQ evaluation including Conformal Risk Control
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_conformal_eval.ps1

param(
    [string]$dataset = 'chest_xray',
    [string]$data_dir = 'data/chest_xray',
    [string]$arch = 'resnet18',
    [string]$baseline_path = '',
    [string]$mc_dropout_path = '',
    [string]$ensemble_dir = 'runs/classification/ensemble',
    [int]$n_ensemble = 5,
    [string]$output_dir = 'runs/classification/evaluation',
    [string]$device = 'cpu'
)

if (!(Test-Path $output_dir)) { New-Item -ItemType Directory -Path $output_dir -Force | Out-Null }

Write-Host "Running evaluation. Output: $output_dir"
conda activate uq_local 2>$null

$cmd = "python -u src/evaluate_uq_classification.py --dataset $dataset --data_dir $data_dir --arch $arch --baseline_path '$baseline_path' --mc_dropout_path '$mc_dropout_path' --ensemble_dir '$ensemble_dir' --n_ensemble $n_ensemble --output_dir $output_dir --device $device"

Write-Host "Command: $cmd"
Invoke-Expression $cmd | Tee-Object -FilePath (Join-Path $output_dir 'evaluation_log.txt')

Write-Host "Evaluation complete. Results: $output_dir/all_results.json"