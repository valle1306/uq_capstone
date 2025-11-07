# Quick Preview: What Will Be Committed

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Preview: Repository Reorganization" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`nFiles that will be moved:" -ForegroundColor Yellow

Write-Host "`n  To docs/status/:" -ForegroundColor Cyan
$statusDocs = @(
    "RETRAINING_STATUS.md",
    "TRAINING_STATUS.md",
    "FINAL_STATUS.md",
    "DATASET_READY.md",
    "IMPLEMENTATION_COMPLETE.md",
    "AMAREL_READY_50EPOCHS.md"
)
foreach ($doc in $statusDocs) {
    if (Test-Path $doc) {
        Write-Host "    OK $doc" -ForegroundColor Green
    } else {
        Write-Host "    -- $doc (not found)" -ForegroundColor Gray
    }
}

Write-Host "`n  To docs/guides/:" -ForegroundColor Cyan
$guideDocs = @(
    "QUICK_START_RETRAIN.md",
    "README_RETRAIN.md",
    "RETRAINING_COMMANDS.md",
    "TROUBLESHOOTING_RETRAINING.md",
    "EXECUTION_CHECKLIST.md",
    "POST_TRAINING_WORKFLOW.md",
    "IMPORT_FIX_GUIDE.md",
    "NEXT_STEPS.md",
    "UPLOAD_COMMANDS.md"
)
foreach ($doc in $guideDocs) {
    if (Test-Path $doc) {
        Write-Host "    OK $doc" -ForegroundColor Green
    } else {
        Write-Host "    -- $doc (not found)" -ForegroundColor Gray
    }
}

Write-Host "`nFiles that will be copied to results/:" -ForegroundColor Yellow
Write-Host "  (Original files in runs/ will remain)" -ForegroundColor Gray

if (Test-Path "runs\classification\metrics\comprehensive_metrics.json") {
    Write-Host "`n  To results/metrics/:" -ForegroundColor Cyan
    Write-Host "    OK comprehensive_metrics.json" -ForegroundColor Green
    Write-Host "    OK metrics_summary.csv" -ForegroundColor Green
}

$pngFiles = Get-ChildItem "runs\classification\metrics\*.png" -ErrorAction SilentlyContinue
if ($pngFiles) {
    Write-Host "`n  To results/figures/:" -ForegroundColor Cyan
    foreach ($file in $pngFiles) {
        Write-Host "    OK $($file.Name)" -ForegroundColor Green
    }
}

Write-Host "`nFiles that will be created:" -ForegroundColor Yellow
Write-Host "  OK results/final/README.md (summary document)" -ForegroundColor Green
Write-Host "  OK README.md (updated main README)" -ForegroundColor Green

Write-Host "`nGit Status Preview:" -ForegroundColor Yellow
Write-Host "  Current branch: " -NoNewline -ForegroundColor Gray
git branch --show-current
Write-Host "  Current commit: " -NoNewline -ForegroundColor Gray
git log -1 --oneline

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Ready to proceed!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`nTo proceed, run:" -ForegroundColor Yellow
Write-Host "  .\setup_final.ps1" -ForegroundColor White -BackgroundColor DarkBlue

Write-Host "`nOr run steps individually:" -ForegroundColor Yellow
Write-Host "  1. .\download_results.ps1" -ForegroundColor White
Write-Host "  2. .\reorganize_repo.ps1" -ForegroundColor White
Write-Host "  3. Then use git commands to commit and push" -ForegroundColor White
