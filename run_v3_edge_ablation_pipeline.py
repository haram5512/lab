"""Long-running coordinator for the controlled v3 edge-operator ablation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
RUNS = ROOT / "runs" / "main_80class"
OUT = RUNS / "v3_edge_operator_ablation"
STATUS = OUT / "pipeline_status.json"
CANNY = RUNS / "proposed_v3_canny_30ep"
LAPLACIAN = RUNS / "proposed_v3_laplacian_30ep"


def epoch_count(run: Path) -> tuple[int, int | None]:
    metrics = run / "metrics.jsonl"
    if not metrics.exists():
        return 0, None
    lines = [line for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return 0, None
    return len(lines), int(json.loads(lines[-1])["epoch"])


def write_status(stage: str, **details: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        **details,
    }
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def validate_training(run: Path, operator: str) -> None:
    records, last_epoch = epoch_count(run)
    required = ("best_clean_map.pt", "best_seen_map.pt", "last.pt", "metrics.jsonl", "run_metadata.json")
    missing = [name for name in required if not (run / name).exists()]
    if records != 30 or last_epoch != 30 or missing:
        raise RuntimeError(
            f"{operator} training incomplete: records={records}, last_epoch={last_epoch}, missing={missing}"
        )
    checkpoint_epoch = int(torch.load(run / "last.pt", map_location="cpu", weights_only=False)["epoch"])
    if checkpoint_epoch != 30:
        raise RuntimeError(f"{operator} last.pt epoch is {checkpoint_epoch}, expected 30")


def wait_for_existing_canny(pid: int) -> None:
    previous_epoch: int | None = None
    last_progress = time.monotonic()
    while True:
        records, last_epoch = epoch_count(CANNY)
        if last_epoch == 30:
            break
        if last_epoch != previous_epoch:
            previous_epoch = last_epoch
            last_progress = time.monotonic()
        idle_seconds = time.monotonic() - last_progress
        if idle_seconds > 7200:
            raise RuntimeError(f"Canny produced no new epoch for {idle_seconds / 3600:.1f} hours")
        write_status("training_canny", pid=pid, records=records, current_epoch=last_epoch, target_epoch=30)
        time.sleep(60)
    validate_training(CANNY, "canny")


def run_stage(stage: str, command: list[str], log_name: str, training_run: Path | None = None) -> None:
    log_path = OUT / log_name
    command = [command[0], "-u", *command[1:]]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nStage {stage} started {datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=PARENT, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            details: dict[str, Any] = {"pid": process.pid, "log": str(log_path)}
            if training_run is not None:
                records, last_epoch = epoch_count(training_run)
                details.update(records=records, current_epoch=last_epoch, target_epoch=30)
            write_status(stage, **details)
            time.sleep(60)
        if process.returncode != 0:
            raise RuntimeError(f"{stage} failed with exit code {process.returncode}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-canny-pid", type=int, default=0)
    parser.add_argument("--train-then-test", action="store_true",
                        help="Use completed Canny; train Laplacian then evaluate both; stop after tests")
    args = parser.parse_args()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this experiment")
        if args.train_then_test:
            validate_training(CANNY, "canny")
            if LAPLACIAN.exists() and any(LAPLACIAN.iterdir()):
                raise RuntimeError("Laplacian output already contains files; refusing to overwrite or restart training")
            write_status("preparing_laplacian", planned_stages=[
                "training_laplacian", "evaluating_canny", "evaluating_laplacian"
            ])
        else:
            write_status("waiting_for_canny", pid=args.wait_canny_pid)
            wait_for_existing_canny(args.wait_canny_pid)
            run_stage("evaluating_canny", [
                sys.executable, "-m", "adversarial_robust_detector.run_v3_edge_final_evaluation",
                "--operator", "canny",
            ], "canny_final_evaluation.log")
        laplacian_command = [
            sys.executable, "-m", "adversarial_robust_detector.train_robust",
            "--model", "proposed_v3", "--edge-operator", "laplacian",
            "--weights", str(ROOT / "yolo11n.pt"),
            "--patch-dir", str(ROOT / "artifacts" / "adversarial_patches" / "train_seen"),
            "--max-images", "0", "--epochs", "30", "--image-size", "640",
            "--batch-size", "16", "--lr", "0.0001", "--seed", "42",
            "--patch-probability", "0.5", "--num-workers", "0", "--pin-memory",
            "--val-every", "5", "--val-max-images", "0", "--output", str(LAPLACIAN),
        ]
        run_stage("training_laplacian", laplacian_command, "laplacian_training.log", LAPLACIAN)
        validate_training(LAPLACIAN, "laplacian")
        if args.train_then_test:
            run_stage("evaluating_canny", [
                sys.executable, "-m", "adversarial_robust_detector.run_v3_edge_final_evaluation",
                "--operator", "canny",
            ], "canny_final_evaluation.log")
        run_stage("evaluating_laplacian", [
            sys.executable, "-m", "adversarial_robust_detector.run_v3_edge_final_evaluation",
            "--operator", "laplacian",
        ], "laplacian_final_evaluation.log")
        if args.train_then_test:
            outputs = {}
            for operator, run in (("canny", CANNY), ("laplacian", LAPLACIAN)):
                result_path = run / "final_evaluation" / "metrics.json"
                metrics = json.loads(result_path.read_text(encoding="utf-8"))
                if set(metrics) != {"clean", "seen", "unseen"}:
                    raise RuntimeError(f"{operator} evaluation is missing conditions")
                outputs[operator] = str(result_path)
            write_status("complete", scope="laplacian_30ep_and_both_final_tests", results=outputs,
                         combined_csv=str(OUT / "canny_laplacian_final_metrics.csv"))
            return
        run_stage("benchmarking", [
            sys.executable, "-m", "adversarial_robust_detector.benchmark_v3_edge_operators",
        ], "latency_benchmark.log")
        run_stage("summarizing", [
            sys.executable, "-m", "adversarial_robust_detector.summarize_v3_edge_operator_ablation",
        ], "summarize.log")
        write_status("complete", report=str(OUT / "FINAL_REPORT.md"))
    except Exception as error:
        write_status("failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
