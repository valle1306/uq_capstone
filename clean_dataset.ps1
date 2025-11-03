# Chest X-Ray Dataset Cleanup Script
# Run this BEFORE uploading to Amarel

Write-Host "=========================================="
Write-Host "Chest X-Ray Dataset Cleanup"
Write-Host "=========================================="
Write-Host ""

$dataPath = "data\chest_xray"

# Issue 1: Extra nested folder
Write-Host "Checking for nested folder issue..."
if (Test-Path "$dataPath\chest_xray") {
    Write-Host "✓ Found extra nested folder: $dataPath\chest_xray"
    Write-Host "  Moving contents up one level..."
    
    # Move everything from chest_xray/chest_xray/ to chest_xray/
    Move-Item -Path "$dataPath\chest_xray\*" -Destination "$dataPath\" -Force
    Remove-Item -Path "$dataPath\chest_xray" -Recurse -Force
    
    Write-Host "  ✓ Fixed!"
} else {
    Write-Host "✓ No nested folder issue"
}

Write-Host ""

# Issue 2: Remove __MACOSX folder (Mac metadata)
Write-Host "Removing Mac metadata (__MACOSX)..."
if (Test-Path "$dataPath\__MACOSX") {
    Remove-Item -Path "$dataPath\__MACOSX" -Recurse -Force
    Write-Host "✓ Removed __MACOSX folder"
} else {
    Write-Host "✓ No __MACOSX folder found"
}

Write-Host ""

# Issue 3: Remove .DS_Store files
Write-Host "Removing .DS_Store files..."
$dsStoreFiles = Get-ChildItem -Path $dataPath -Recurse -File -Filter ".DS_Store" -Force
if ($dsStoreFiles.Count -gt 0) {
    $dsStoreFiles | Remove-Item -Force
    Write-Host "✓ Removed $($dsStoreFiles.Count) .DS_Store files"
} else {
    Write-Host "✓ No .DS_Store files found"
}

Write-Host ""

# Issue 4: Remove Thumbs.db files (Windows)
Write-Host "Removing Thumbs.db files..."
$thumbsFiles = Get-ChildItem -Path $dataPath -Recurse -File -Filter "Thumbs.db" -Force
if ($thumbsFiles.Count -gt 0) {
    $thumbsFiles | Remove-Item -Force
    Write-Host "✓ Removed $($thumbsFiles.Count) Thumbs.db files"
} else {
    Write-Host "✓ No Thumbs.db files found"
}

Write-Host ""

# Issue 5: Remove hidden files starting with ._
Write-Host "Removing ._ files (Mac resource forks)..."
$hiddenFiles = Get-ChildItem -Path $dataPath -Recurse -File -Force | Where-Object {$_.Name -like "._*"}
if ($hiddenFiles.Count -gt 0) {
    $hiddenFiles | Remove-Item -Force
    Write-Host "✓ Removed $($hiddenFiles.Count) ._ files"
} else {
    Write-Host "✓ No ._ files found"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Validation"
Write-Host "=========================================="
Write-Host ""

# Check final structure
Write-Host "Final directory structure:"
Get-ChildItem -Path $dataPath -Directory | ForEach-Object {
    Write-Host "  $($_.Name)/"
    Get-ChildItem -Path $_.FullName -Directory | ForEach-Object {
        $fileCount = (Get-ChildItem -Path $_.FullName -File).Count
        Write-Host "    $($_.Name)/  ($fileCount)"
    }
}

Write-Host ""

# Count total files
$totalFiles = (Get-ChildItem -Path $dataPath -Recurse -File).Count
Write-Host "Total image files: $totalFiles"

# Calculate size
$sizeGB = [math]::Round(((Get-ChildItem -Path $dataPath -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB), 2)
Write-Host "Total size: $sizeGB GB"

Write-Host ""
Write-Host "=========================================="
Write-Host "✓ Cleanup Complete!"
Write-Host "=========================================="
Write-Host ""
Write-Host "Expected counts:"
Write-Host "  train/NORMAL:    ~1,341 files"
Write-Host "  train/PNEUMONIA: ~3,875 files"
Write-Host "  test/NORMAL:     ~234 files"
Write-Host "  test/PNEUMONIA:  ~390 files"
Write-Host "  val/NORMAL:      ~8 files"
Write-Host "  val/PNEUMONIA:   ~8 files"
Write-Host ""
Write-Host "Ready to upload to Amarel!"
