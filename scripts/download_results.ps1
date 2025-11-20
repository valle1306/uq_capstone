# Download SWAG and conformal results from Amarel to local workspace
# Usage: Edit $remoteUser and $remoteBase then run in PowerShell

$remoteUser = "hpl14@amarel.rutgers.edu"
$remoteBase = "/scratch/hpl14/uq_capstone"
$localBase = "C:\Users\lpnhu\Downloads\uq_capstone"

# Create local directories
New-Item -ItemType Directory -Force -Path "$localBase\runs\classification\swag_adam" | Out-Null
New-Item -ItemType Directory -Force -Path "$localBase\runs\classification\conformal\swag" | Out-Null

Write-Host "Downloading SWAG Adam model and logs..."
scp $remoteUser:"$remoteBase/runs/classification/swag_adam/*" "$localBase\runs\classification\swag_adam\"
scp $remoteUser:"$remoteBase/logs/swag_adam_48295659.out" "$localBase\logs\"

Write-Host "Downloading conformal results..."
scp -r $remoteUser:"$remoteBase/runs/classification/conformal/swag" "$localBase\runs\classification\conformal\"

Write-Host "Download complete. Check $localBase\runs\classification for files."