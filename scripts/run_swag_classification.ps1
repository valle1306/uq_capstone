Param(
    [string]$Dataset = 'chest_xray',
    [int]$Epochs = 5,
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [string]$Device = 'cpu',
    [string]$OutputDir = 'runs/classification/swag_classification'
)

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

$python = 'python'


$pyArgs = @(
    'src/train_swag_classification.py'
    '--dataset', $Dataset
    '--epochs', $Epochs
    '--batch_size', $BatchSize
    '--num_workers', $NumWorkers
    '--device', $Device
    '--output_dir', $OutputDir
    '--swag_start', 3
    '--max_models', 20
)

Write-Host "Running SWAG classification trainer: $($pyArgs -join ' ')"

& $python -u $pyArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir 'swag_train.log')

Pop-Location
