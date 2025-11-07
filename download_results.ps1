# Download All Results from Amarel
# Run this script from the uq_capstone directory

$AMAREL_USER = "hpl14"
$AMAREL_HOST = "amarel.rutgers.edu"
$REMOTE_PATH = "/scratch/$AMAREL_USER/uq_capstone"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Downloading Results from Amarel" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Create local directories if they don't exist
Write-Host "`nCreating local directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "runs\classification\metrics" | Out-Null
New-Item -ItemType Directory -Force -Path "logs\analysis" | Out-Null
New-Item -ItemType Directory -Force -Path "results\figures" | Out-Null

# Download comprehensive metrics
Write-Host "`n[1/6] Downloading comprehensive metrics JSON..." -ForegroundColor Green
scp "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_PATH}/runs/classification/metrics/comprehensive_metrics.json" "runs\classification\metrics\"

# Download metrics summary CSV
Write-Host "`n[2/6] Downloading metrics summary CSV..." -ForegroundColor Green
scp "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_PATH}/runs/classification/metrics/metrics_summary.csv" "runs\classification\metrics\"

# Download visualizations
Write-Host "`n[3/6] Downloading visualization PNG files..." -ForegroundColor Green
scp "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_PATH}/runs/classification/metrics/comprehensive_metrics_visualization.png" "runs\classification\metrics\"
scp "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_PATH}/runs/classification/metrics/method_comparisons.png" "runs\classification\metrics\"

# Download latest log files (last 5 evaluation logs)
Write-Host "`n[4/6] Downloading recent log files..." -ForegroundColor Green
ssh "${AMAREL_USER}@${AMAREL_HOST}" "cd ${REMOTE_PATH}/logs && ls -t eval_*.out | head -5" | ForEach-Object {
    scp "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_PATH}/logs/$_" "logs\"
}

# Download model checkpoints info (just list them, don't download large files)
Write-Host "`n[5/6] Checking model checkpoint sizes..." -ForegroundColor Green
Write-Host "Model checkpoints on Amarel:" -ForegroundColor Cyan
ssh "${AMAREL_USER}@${AMAREL_HOST}" "ls -lh ${REMOTE_PATH}/runs/classification/*/best_model.pth ${REMOTE_PATH}/runs/classification/swag_classification/swag_model.pth 2>/dev/null || echo 'Some models not found'"

# Download training configs
Write-Host "`n[6/6] Downloading training configs..." -ForegroundColor Green
scp "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_PATH}/runs/classification/mc_dropout/config.json" "runs\classification\mc_dropout\" 2>$null
scp "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_PATH}/runs/classification/swag_classification/config.json" "runs\classification\swag_classification\" 2>$null

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "Download Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan

Write-Host "`nDownloaded files:" -ForegroundColor Yellow
Write-Host "  - runs\classification\metrics\comprehensive_metrics.json"
Write-Host "  - runs\classification\metrics\metrics_summary.csv"
Write-Host "  - runs\classification\metrics\*.png"
Write-Host "  - logs\eval_*.out (recent logs)"
Write-Host "  - runs\classification\*\config.json"

Write-Host "`nNote: Model checkpoints (.pth files) are NOT downloaded (they are large)." -ForegroundColor Cyan
Write-Host "If you need them, use:" -ForegroundColor Cyan
Write-Host "  scp -r ${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_PATH}/runs/classification/mc_dropout/best_model.pth runs\classification\mc_dropout\" -ForegroundColor Gray
