# Run SWAG training locally (PowerShell wrapper)
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_swag.ps1

param(
    [int]$epochs = 5,
    [int]$batch_size = 8,
    [int]$num_workers = 0,
    [string]$device = 'cpu',
    [string]$output_dir = 'runs/classification/swag'
)

$repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (!(Test-Path $output_dir)) { New-Item -ItemType Directory -Path $output_dir -Force | Out-Null }
$log = Join-Path $output_dir 'swag_train.log'

Write-Host "Running SWAG training: epochs=$epochs, batch_size=$batch_size, device=$device"

# Activate conda env if available
Write-Host "Activating conda environment 'uq_local' (if available)"
conda activate uq_local 2>$null

# Run with unbuffered output
python -u src/train_swag.py --epochs $epochs --batch_size $batch_size --num_workers $num_workers --device $device --output_dir $output_dir 2>&1 | Tee-Object -FilePath $log

Write-Host "SWAG run complete. Logs: $log"