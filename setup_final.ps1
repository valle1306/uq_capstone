# Master Script: Download, Reorganize, and Push
# This script does everything in sequence

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "UQ Capstone - Complete Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`nThis script will:" -ForegroundColor Yellow
Write-Host "  1. Download all results from Amarel" -ForegroundColor White
Write-Host "  2. Reorganize the repository structure" -ForegroundColor White
Write-Host "  3. Stage and commit changes to git" -ForegroundColor White
Write-Host "  4. Push to GitHub" -ForegroundColor White

$continue = Read-Host "`nContinue? (y/n)"
if ($continue -ne "y") {
    Write-Host "Cancelled." -ForegroundColor Red
    exit
}

# Step 1: Download results
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 1: Downloading Results from Amarel" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
& ".\download_results.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nDownload had some errors. Continue anyway? (y/n)" -ForegroundColor Yellow
    $continue = Read-Host
    if ($continue -ne "y") {
        Write-Host "Cancelled." -ForegroundColor Red
        exit
    }
}

# Step 2: Reorganize
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 2: Reorganizing Repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
& ".\reorganize_repo.ps1"

# Step 3: Git operations
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 3: Git Operations" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nChecking git status..." -ForegroundColor Yellow
git status --short

Write-Host "`nStage all changes? (y/n)" -ForegroundColor Yellow
$continue = Read-Host
if ($continue -ne "y") {
    Write-Host "Skipping git operations. You can do them manually." -ForegroundColor Yellow
    exit
}

Write-Host "`nStaging changes..." -ForegroundColor Green
git add .

Write-Host "`nCreating commit..." -ForegroundColor Green
git commit -m "Reorganize repository: move docs to subfolders, copy results for easy access, update README"

Write-Host "`nPushing to GitHub..." -ForegroundColor Green
git push origin main

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "ALL DONE" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`nYour repository is now:" -ForegroundColor Yellow
Write-Host "  Downloaded with latest results" -ForegroundColor Green
Write-Host "  Reorganized with clean structure" -ForegroundColor Green
Write-Host "  Committed and pushed to GitHub" -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  - Check results/figures/ for visualizations" -ForegroundColor White
Write-Host "  - Review results/metrics/metrics_summary.csv" -ForegroundColor White
Write-Host "  - Read results/final/README.md for analysis" -ForegroundColor White
Write-Host "  - Prepare your presentation using the figures" -ForegroundColor White
