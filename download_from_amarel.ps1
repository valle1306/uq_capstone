# Download Script for Updated Amarel Files
# Run this in PowerShell from C:\Users\lpnhu\Downloads\uq_capstone

Write-Host "🔄 Downloading updated files from Amarel..." -ForegroundColor Cyan
Write-Host ""

# Download the package
Write-Host "Downloading swag_fix_package.tar.gz..."
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/swag_fix_package.tar.gz .

Write-Host ""
Write-Host "✅ Download complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📦 To extract the files, you'll need to:"
Write-Host "   1. Install 7-Zip or WinRAR"
Write-Host "   2. Right-click swag_fix_package.tar.gz"
Write-Host "   3. Extract Here"
Write-Host ""
Write-Host "Or use WSL/Git Bash:"
Write-Host "   tar xzf swag_fix_package.tar.gz"
Write-Host ""

# Alternative: Download files individually
Write-Host "Or download files individually:"
Write-Host ""
Write-Host "  scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/swag.py src/"
Write-Host "  scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/evaluate_uq_FIXED_v2.py src/"
Write-Host "  scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/SWAG_FIXED_SUCCESS.md ."
Write-Host "  scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/evaluation/results.json ."
Write-Host ""
