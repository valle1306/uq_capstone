# Queue ensemble members 2-4 sequentially after member 1 finishes
# Usage: powershell -File scripts\queue_ensemble_members.ps1

$members = 2..4
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python = "python"

function Start-Member($id) {
    $outDir = Join-Path $repoRoot "..\runs\classification\ensemble\member_$id"
    if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

    $trainLog = Join-Path $outDir "member_${id}_train.log"
    $errLog = Join-Path $outDir "member_${id}_err.log"

    $args = "src\train_classifier_ensemble_member.py --member_id $id --epochs 20 --batch_size 8 --device cpu --output_dir runs/classification/ensemble --save_freq 1 --num_workers 0"

    Write-Host "Starting member $id... Logs: $trainLog, $errLog"

    # Start-Process with redirection
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $python
    $psi.Arguments = $args
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = $repoRoot

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $proc.Start() | Out-Null

    $stdOut = $proc.StandardOutput
    $stdErr = $proc.StandardError

    # Asynchronously write stdout and stderr to files
    $outWriter = [System.IO.File]::CreateText($trainLog)
    $errWriter = [System.IO.File]::CreateText($errLog)

    Start-Job -ScriptBlock {
        param($reader, $writer)
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            $writer.WriteLine($line)
            $writer.Flush()
        }
    } -ArgumentList $stdOut, $outWriter | Out-Null

    Start-Job -ScriptBlock {
        param($reader, $writer)
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            $writer.WriteLine($line)
            $writer.Flush()
        }
    } -ArgumentList $stdErr, $errWriter | Out-Null

    # Wait for process to exit
    $proc.WaitForExit()
    $outWriter.Close()
    $errWriter.Close()

    Write-Host "Member $id finished with exit code $($proc.ExitCode)"
}

# Wait for member_1 to finish: look for a running python process with member_id 1 in args or check for member_1_train.log growth
Write-Host "Queue script started. Waiting for member 1 to finish (if running)."

while ($true) {
    # Check for python processes containing 'member_id 1' in commandline
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "member_id\s+1" }
    if ($procs -eq $null -or $procs.Count -eq 0) {
        Write-Host "No running member_1 found. Verifying outputs..."

        # Verify member_1 saved a model (best_model.pth or final_model.pth)
        $m1Dir = Join-Path $repoRoot "..\runs\classification\ensemble\member_1"
        $found = $false
        $waited = 0
        $timeout = 300  # seconds to wait for model file (5 min)

        while ($waited -lt $timeout) {
            if (Test-Path (Join-Path $m1Dir "best_model.pth") -PathType Leaf -ErrorAction SilentlyContinue -or Test-Path (Join-Path $m1Dir "final_model.pth") -PathType Leaf -ErrorAction SilentlyContinue) {
                $found = $true
                break
            }
            Start-Sleep -Seconds 5
            $waited += 5
        }

        if (-not $found) {
            Write-Warning "member_1 finished but no model file found in $m1Dir after $timeout seconds. Proceeding to queued members (you may want to inspect member_1 logs)."
        } else {
            Write-Host "member_1 outputs verified. Starting queued members..."
        }

        break
    } else {
        Write-Host "member_1 still running. Sleeping 60s..."
        Start-Sleep -Seconds 60
    }
}

# Start queued members sequentially
foreach ($id in $members) {
    # Marker file for audit
    $marker = Join-Path $repoRoot "..\runs\classification\ensemble\queue_status_member_${id}.txt"
    "START: $(Get-Date -Format o)" | Out-File -FilePath $marker -Encoding ascii

    Start-Member $id

    "END: $(Get-Date -Format o) ExitCode: $($LASTEXITCODE)" | Out-File -FilePath $marker -Append -Encoding ascii
}

Write-Host "All queued members completed."