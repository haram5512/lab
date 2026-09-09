param(
    [int]$TrainingPid = 15396,
    [int]$WrapperPid = 17916,
    [int]$Patience = 3,
    [double]$MinDelta = 0.001,
    [int]$PollSeconds = 60
)

$outputDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$metricsPath = Join-Path $outputDir 'metrics.jsonl'
$monitorLog = Join-Path $outputDir 'early_stop.log'

while ($true) {
    $training = Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue
    if (-not $training) {
        Add-Content -LiteralPath $monitorLog -Value "$(Get-Date -Format s) training process ended; monitor exiting"
        break
    }

    if (Test-Path -LiteralPath $metricsPath) {
        $validations = @(
            Get-Content -LiteralPath $metricsPath |
                ForEach-Object { $_ | ConvertFrom-Json } |
                Where-Object { $null -ne $_.map }
        )

        $best = [double]::NegativeInfinity
        $stale = 0
        foreach ($record in $validations) {
            $score = [double]$record.map
            if ($score -gt ($best + $MinDelta)) {
                $best = $score
                $stale = 0
            } else {
                $stale += 1
            }
        }

        if ($validations.Count -gt 0) {
            $latest = $validations[-1]
            $status = "$(Get-Date -Format s) epoch=$($latest.epoch) map=$($latest.map) best=$best stale=$stale/$Patience"
            $previous = if (Test-Path -LiteralPath $monitorLog) { Get-Content -LiteralPath $monitorLog -Tail 1 } else { '' }
            if ($previous -notmatch "epoch=$($latest.epoch) ") {
                Add-Content -LiteralPath $monitorLog -Value $status
            }
        }

        if ($stale -ge $Patience) {
            # metrics.jsonl is written immediately before the checkpoints. Wait
            # until last.pt is newer than the metrics record before stopping.
            $lastPath = Join-Path $outputDir 'last.pt'
            $metricsWrite = (Get-Item -LiteralPath $metricsPath).LastWriteTimeUtc
            $saved = $false
            for ($attempt = 0; $attempt -lt 30; $attempt += 1) {
                if (Test-Path -LiteralPath $lastPath) {
                    $saved = (Get-Item -LiteralPath $lastPath).LastWriteTimeUtc -ge $metricsWrite
                }
                if ($saved) { break }
                Start-Sleep -Seconds 2
            }
            if ($saved) {
                Add-Content -LiteralPath $monitorLog -Value "$(Get-Date -Format s) early stopping: AP failed to improve by $MinDelta for $Patience validations; checkpoint saved"
                Stop-Process -Id $TrainingPid -ErrorAction SilentlyContinue
                Stop-Process -Id $WrapperPid -ErrorAction SilentlyContinue
                break
            }
            Add-Content -LiteralPath $monitorLog -Value "$(Get-Date -Format s) early stop deferred: final checkpoint not confirmed"
        }
    }

    Start-Sleep -Seconds $PollSeconds
}
