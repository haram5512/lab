param(
    [int]$TrainingPid = 15396,
    [int]$WrapperPid = 17916,
    [int]$PollSeconds = 60
)

$outputDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$watchLog = Join-Path $outputDir 'shutdown_watch.log'
$python = Join-Path (Split-Path -Parent (Split-Path -Parent $outputDir)) '.venv\Scripts\python.exe'
$verifier = Join-Path $outputDir 'verify_training_completion.py'

Add-Content -LiteralPath $watchLog -Value "$(Get-Date -Format s) shutdown watcher started"
while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $PollSeconds
}

# Allow redirected streams and filesystem buffers to flush after training exits.
Start-Sleep -Seconds 30
if (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    Add-Content -LiteralPath $watchLog -Value "$(Get-Date -Format s) abort: training process still running"
    exit 1
}

& $python $verifier *> (Join-Path $outputDir 'shutdown_verification.log')
if ($LASTEXITCODE -ne 0) {
    Add-Content -LiteralPath $watchLog -Value "$(Get-Date -Format s) abort: artifact verification failed; system will remain on"
    exit 1
}

Add-Content -LiteralPath $watchLog -Value "$(Get-Date -Format s) verification passed; shutting down Windows"
Stop-Process -Id $WrapperPid -ErrorAction SilentlyContinue
shutdown.exe /s /t 0
