$ErrorActionPreference = "Stop"
$project = "C:\논문\adversarial_robust_detector"
$run = Join-Path $project "runs\main_80class\v5c_adversarial_training_corrected"
$log = Join-Path $project "runs\v5c_corrected_followup.log"
Wait-Process -Id 22436
$required = @("last.pt", "best_clean_map.pt", "best_seen_map.pt", "metrics.jsonl")
foreach ($name in $required) {
    if (-not (Test-Path (Join-Path $run $name))) {
        "Training ended without $name; final evaluation was not started." | Set-Content -Encoding UTF8 $log
        exit 1
    }
}
$last = Get-Content (Join-Path $run "metrics.jsonl") -Tail 1 | ConvertFrom-Json
if ($last.epoch -lt 30) {
    "Training ended before min_epochs at epoch $($last.epoch); final evaluation was not started." | Set-Content -Encoding UTF8 $log
    exit 1
}
$python = Join-Path $project ".venv\Scripts\python.exe"
$seen = Join-Path $project "runs\v5_diagnostic\patch_V5-C.pt"
$unseen = Join-Path $project "artifacts\adversarial_patches\v2\unseen\patch_F_v2.pt"
$models = @(
    @{ Name = "pretrained"; Path = (Join-Path $project "yolo11n.pt") },
    @{ Name = "best_clean"; Path = (Join-Path $run "best_clean_map.pt") },
    @{ Name = "best_seen"; Path = (Join-Path $run "best_seen_map.pt") },
    @{ Name = "last"; Path = (Join-Path $run "last.pt") }
)
foreach ($model in $models) {
    foreach ($condition in @(@{ Name = "seen"; Patch = $seen }, @{ Name = "unseen"; Patch = $unseen })) {
        $output = Join-Path $run ("eval_" + $model.Name + "_" + $condition.Name)
        & $python -m adversarial_robust_detector.evaluate_patch_attack --checkpoint $model.Path --patch $condition.Patch --max-images 0 --mode deterministic --output $output
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $($model.Name) $($condition.Name)" }
    }
}
"Training ended at epoch $($last.epoch); final evaluations completed." | Set-Content -Encoding UTF8 $log
