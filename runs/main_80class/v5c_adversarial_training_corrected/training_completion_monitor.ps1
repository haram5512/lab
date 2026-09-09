param(
    [int]$TrainingPid = 22436,
    [int]$LauncherPid = 14048
)

$ErrorActionPreference = "Stop"
$runDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = (Resolve-Path (Join-Path $runDir "..\..\..")).Path
$metricsPath = Join-Path $runDir "metrics.jsonl"
$metadataPath = Join-Path $runDir "run_metadata.json"
$summaryPath = Join-Path $runDir "training_completion_summary.json"
$monitorLog = Join-Path $runDir "training_completion_monitor.log"
$requiredNames = @("last.pt", "best_clean_map.pt", "best_seen_map.pt", "metrics.jsonl", "run_metadata.json")

function Write-MonitorLog([string]$Message) {
    $line = "{0:o} {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $monitorLog -Value $line -Encoding UTF8
}

function Save-Summary([hashtable]$Summary) {
    $tempPath = "$summaryPath.tmp"
    $json = $Summary | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($tempPath, $json, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tempPath -Destination $summaryPath -Force
}

Write-MonitorLog "Monitoring training PID $TrainingPid; shutdown is gated by completion checks."
Wait-Process -Id $TrainingPid -ErrorAction Stop
Write-MonitorLog "Training PID $TrainingPid ended; waiting for launcher and file flush."

Wait-Process -Id $LauncherPid -ErrorAction Stop
Start-Sleep -Seconds 10

$errors = [System.Collections.Generic.List[string]]::new()
if (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) { $errors.Add("training_process_still_running") }
if (Get-Process -Id $LauncherPid -ErrorAction SilentlyContinue) { $errors.Add("launcher_process_still_running") }

$fileInfo = @{}
foreach ($name in $requiredNames) {
    $path = Join-Path $runDir $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $errors.Add("missing:$name")
        continue
    }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -le 0) { $errors.Add("empty:$name") }
    $fileInfo[$name] = @{ path = $item.FullName; bytes = $item.Length; last_write = $item.LastWriteTime.ToString("o") }
}

$rows = @()
$metadata = $null
if (Test-Path -LiteralPath $metricsPath) {
    try { $rows = @(Get-Content -LiteralPath $metricsPath | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }) }
    catch { $errors.Add("metrics_parse_failed:$($_.Exception.Message)") }
}
if (Test-Path -LiteralPath $metadataPath) {
    try { $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json }
    catch { $errors.Add("metadata_parse_failed:$($_.Exception.Message)") }
}
if ($rows.Count -eq 0) { $errors.Add("metrics_has_no_records") }

$final = if ($rows.Count) { $rows | Sort-Object {[int]$_.epoch} | Select-Object -Last 1 } else { $null }
$validationRows = @($rows | Where-Object { $null -ne $_.clean_person_ap -and $null -ne $_.seen_person_ap })
$bestClean = if ($validationRows.Count) { $validationRows | Sort-Object {[double]$_.clean_person_ap} -Descending | Select-Object -First 1 } else { $null }
$bestSeen = if ($validationRows.Count) { $validationRows | Sort-Object {[double]$_.seen_person_ap} -Descending | Select-Object -First 1 } else { $null }
if ($validationRows.Count -eq 0) { $errors.Add("no_completed_validation_record") }

$rawMetrics = if (Test-Path -LiteralPath $metricsPath) { Get-Content -LiteralPath $metricsPath -Raw } else { "" }
if ($rawMetrics -match '(?i)(?<![A-Za-z])[+-]?(nan|infinity|inf)(?![A-Za-z])') { $errors.Add("nan_or_inf_in_metrics") }
foreach ($row in $validationRows) {
    if ([int]$row.clean_person_gt -le 0 -or [int]$row.seen_person_gt -le 0 -or [int]$row.clean_person_gt -ne [int]$row.seen_person_gt) {
        $errors.Add("person_gt_filtering_invalid_at_epoch_$($row.epoch)")
    }
}

$finalEpoch = if ($final) { [int]$final.epoch } else { 0 }
$earlyStopped = [bool]($final -and $final.early_stop_triggered)
$maxEpoch = if ($metadata) { [int]$metadata.epochs } else { 100 }
$exitReason = if ($earlyStopped) { "early_stopping" } elseif ($finalEpoch -eq $maxEpoch) { "max_epoch_completed" } else { "abnormal_or_incomplete" }
if ($exitReason -eq "abnormal_or_incomplete") { $errors.Add("training_did_not_reach_normal_exit_condition") }
if ($earlyStopped -and $metadata -and $finalEpoch -lt [int]$metadata.early_stopping.min_epochs) { $errors.Add("early_stop_before_min_epochs") }

$checkpointEpochs = @{}
$pythonPath = Join-Path $repoDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $errors.Add("venv_python_missing")
} elseif (($requiredNames[0..2] | ForEach-Object { Test-Path -LiteralPath (Join-Path $runDir $_) }) -notcontains $false) {
    try {
        $validatorPath = Join-Path $runDir "validate_completion_checkpoints.py"
        $checkpointJson = & $pythonPath $validatorPath (Join-Path $runDir "last.pt") (Join-Path $runDir "best_clean_map.pt") (Join-Path $runDir "best_seen_map.pt")
        if ($LASTEXITCODE -ne 0) { throw "checkpoint validator exited $LASTEXITCODE" }
        $checkpointEpochs = $checkpointJson | ConvertFrom-Json
        if ([int]$checkpointEpochs.(Join-Path $runDir "last.pt") -ne $finalEpoch) { $errors.Add("last_checkpoint_epoch_mismatch") }
    } catch { $errors.Add("checkpoint_load_failed:$($_.Exception.Message)") }
}

# Confirm all result files are unchanged for 30 seconds.
$before = @{}
foreach ($name in $requiredNames) {
    $path = Join-Path $runDir $name
    if (Test-Path -LiteralPath $path) { $item = Get-Item -LiteralPath $path; $before[$name] = "$($item.Length)|$($item.LastWriteTimeUtc.Ticks)" }
}
Start-Sleep -Seconds 30
foreach ($name in $before.Keys) {
    $item = Get-Item -LiteralPath (Join-Path $runDir $name)
    if ($before[$name] -ne "$($item.Length)|$($item.LastWriteTimeUtc.Ticks)") { $errors.Add("file_not_stable:$name") }
}

$gitDeadline = (Get-Date).AddMinutes(10)
while (@(Get-Process -Name git,git-lfs -ErrorAction SilentlyContinue).Count -gt 0 -and (Get-Date) -lt $gitDeadline) {
    Write-MonitorLog "Git/Git LFS process is active; waiting before finalization."
    Start-Sleep -Seconds 10
}
if (@(Get-Process -Name git,git-lfs -ErrorAction SilentlyContinue).Count -gt 0) { $errors.Add("git_or_lfs_process_still_running") }

$summary = @{
    training_completed = ($errors.Count -eq 0)
    exit_reason = $exitReason
    final_epoch = $finalEpoch
    early_stopping = $earlyStopped
    best_clean_epoch = if ($bestClean) { [int]$bestClean.epoch } else { $null }
    best_seen_epoch = if ($bestSeen) { [int]$bestSeen.epoch } else { $null }
    best_clean_ap = if ($bestClean) { [double]$bestClean.clean_person_ap } else { $null }
    best_seen_ap = if ($bestSeen) { [double]$bestSeen.seen_person_ap } else { $null }
    clean_ap50 = if ($bestClean) { [double]$bestClean.clean_person_ap50 } else { $null }
    seen_ap50 = if ($bestSeen) { [double]$bestSeen.seen_person_ap50 } else { $null }
    clean_recall = if ($bestClean) { [double]$bestClean.clean_person_recall } else { $null }
    seen_recall = if ($bestSeen) { [double]$bestSeen.seen_person_recall } else { $null }
    seen_failure_rate = if ($bestSeen) { [double]$bestSeen.seen_failure_rate } else { $null }
    nan_or_inf_found = ($errors -contains "nan_or_inf_in_metrics")
    checkpoint_paths = @{
        last = Join-Path $runDir "last.pt"
        best_clean = Join-Path $runDir "best_clean_map.pt"
        best_seen = Join-Path $runDir "best_seen_map.pt"
    }
    checkpoint_epochs = $checkpointEpochs
    metrics_path = $metricsPath
    metadata_path = $metadataPath
    timestamp = (Get-Date).ToString("o")
    git_commit_hash = if ($metadata) { $metadata.git_commit } else { $null }
    validation_errors = @($errors)
}
Save-Summary $summary

if ($errors.Count -gt 0) {
    Write-MonitorLog "Validation failed; shutdown NOT executed: $($errors -join '; ')"
    exit 2
}

Start-Sleep -Seconds 2
if ((Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) -or (Get-Process -Id $LauncherPid -ErrorAction SilentlyContinue)) {
    $summary.training_completed = $false
    $summary.validation_errors = @("training_process_reappeared")
    Save-Summary $summary
    Write-MonitorLog "Final process check failed; shutdown NOT executed."
    exit 3
}
if (@(Get-Process -Name git,git-lfs -ErrorAction SilentlyContinue).Count -gt 0) {
    $summary.training_completed = $false
    $summary.validation_errors = @("git_or_lfs_process_running_at_final_check")
    Save-Summary $summary
    Write-MonitorLog "Final Git/LFS check failed; shutdown NOT executed."
    exit 4
}

Write-MonitorLog "All completion checks passed. Executing the authorized shutdown command."
shutdown -s -t 0
