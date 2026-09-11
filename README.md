현재 프로젝트는 `C:\논문\adversarial_robust_detector`이며, 크게 모델 구현·공격 생성·학습/평가 스크립트·실험 결과로 구성되어 있습니다.

## 전체 구성

```text
adversarial_robust_detector/
├─ models/          제안 모델 v1, v2, v3
├─ attacks/         적대적 패치 생성 및 변환
├─ tests/           모델·공격·평가지표 테스트
├─ artifacts/       생성된 패치와 기준 체크포인트
├─ runs/            학습 체크포인트와 전체 실험 결과
├─ datasets/        데이터셋 위치(현재 폴더 내부는 비어 있음)
├─ Ultralytics/     Ultralytics 로컬 설정
├─ *.py             학습·평가·분석 실행 스크립트
├─ *.yaml           학습·데이터·공격 설정
├─ *.md / *.docx    사용법과 연구 문서
└─ yolo11n.pt       기본 YOLO11n 사전학습 모델
```

## 1. 모델 구현

[models 폴더](</C:/논문/adversarial_robust_detector/models>)에 핵심 제안 모델이 있습니다.

- [proposed_rgb_shape.py](</C:/논문/adversarial_robust_detector/models/proposed_rgb_shape.py>): Proposed RGB/Shape v1
- [proposed_rgb_shape_v2.py](</C:/논문/adversarial_robust_detector/models/proposed_rgb_shape_v2.py>): v2, P3/P4/P5 다중 스케일 Shape 융합
- [proposed_rgb_shape_v3.py](</C:/논문/adversarial_robust_detector/models/proposed_rgb_shape_v3.py>): v3, YOLO11n과 유사한 Shape backbone
- [model.py](</C:/논문/adversarial_robust_detector/model.py>): 기존 모델의 공통 구조
- [losses.py](</C:/논문/adversarial_robust_detector/losses.py>): 학습 손실 함수
- [yolo11n.pt](</C:/논문/adversarial_robust_detector/yolo11n.pt>): RGB backbone 초기화 등에 사용되는 YOLO11n 가중치

## 2. 적대적 패치 공격

[attacks 폴더](</C:/논문/adversarial_robust_detector/attacks>)에는 다음 코드가 있습니다.

- `patch_generator.py`: 패치 최적화 및 생성
- `patch_transforms.py`: 패치 크기·회전·배치 변환
- `patch_losses.py`: 패치 공격용 손실
- `patch_splits.py`: Seen/Unseen 패치 분할
- `patch_config.py`: 공격 설정 관리

[artifacts/adversarial_patches](</C:/논문/adversarial_robust_detector/artifacts/adversarial_patches>)에는 다음 결과가 저장되어 있습니다.

- 학습에 사용한 Seen 패치 A~E
- 평가용 Unseen 패치 F~H
- 패치 이미지 `.png`
- 생성 정보 `.json`
- v2 패치 텐서 `.pt`

## 3. 학습 및 평가 코드

대표적인 실행 파일은 다음과 같습니다.

- [train_baseline.py]: Baseline 학습
- [train_robust.py]: 적대적 강건 학습
- [evaluate_coco.py]: COCO 방식 성능 평가
- [evaluate_patch_attack.py]: 패치 공격 평가
- [run_unified_model_comparison.py]: 모델 통합 비교
- [run_v3_final_evaluation.py]: v3 최종 평가
- [run_v3_shape_ablation.py]: v3 Shape ablation
- [generate_visual_samples.py]: 검출 결과 시각화
- `benchmark_*.py`: 모델별 추론 지연시간 측정
- `diagnose_*.py`, `compare_*.py`: 실험 결과 진단 및 비교
- `finalize_*.py`: 최종 CSV 및 보고서 생성

## 4. 설정 및 데이터 처리

- [dataset.py]: 데이터셋 로딩과 전처리
- [dataset_config.main80.yaml]: 80클래스 본 실험 데이터 설정
- [dataset_config.yaml]: 일반 데이터 설정
- [baseline_train_config.yaml]: Baseline 학습 설정
- [adversarial_patch_config.yaml]: 패치 공격 설정
- [requirements.txt]: Python 패키지 목록

`datasets` 폴더 자체는 현재 비어 있습니다. 실제 COCO 이미지나 라벨처럼 큰 데이터셋은 Git에 포함하지 않고 YAML 설정에서 외부 위치를 참조하는 구조입니다.

## 5. 실험 결과

[runs 폴더](</C:/논문/adversarial_robust_detector/runs>)에는 총 532개 파일, 약 776MB가 있습니다.

핵심은 [runs/main_80class](</C:/논문/adversarial_robust_detector/runs/main_80class>)이며 다음 결과를 포함합니다.

- `proposed_rgb_shape_v1_full`: v1 본 학습 체크포인트
- `proposed_rgb_shape_v1_final`: v1 최종 평가
- `proposed_rgb_shape_v2_full`: v2 본 학습 체크포인트
- `proposed_rgb_shape_v3_full`: v3 본 학습 체크포인트
- `proposed_rgb_shape_v3_final_eval`: v3 최종 평가
- `proposed_rgb_shape_v3_yolo_backbone`: YOLO형 Shape backbone 실험
- `v3_rgb_only_vs_shape_ablation`: RGB-only와 Shape 사용 모델 비교
- `final_model_comparison`: v1/v2/v3 및 Baseline 최종 비교
- `visual_samples_150`: 시각화 결과 150장
- `adversarial_training_final_evaluation`: 강건 학습 최종 평가

결과 파일 형식은 대략 다음과 같습니다.

| 종류 | 개수 | 용도 |
|---|---:|---|
| `.jpg` | 200 | 검출 결과 시각화 |
| `.json` | 139 | 상세 평가 결과 및 설정 |
| `.log` | 65 | 학습·평가 로그 |
| `.pt` | 64 | 모델 체크포인트와 패치 텐서 |
| `.csv` | 19 | 모델 성능 비교표 |
| `.jsonl` | 19 | Epoch별 학습 기록 |
| `.png` | 13 | 패치 및 그래프 이미지 |

현재 열어둔 [all_models_metrics_with_v3.csv](</C:/논문/adversarial_robust_detector/runs/main_80class/final_model_comparison/all_models_metrics_with_v3.csv>)도 이 최종 비교 결과에 해당합니다.

## 6. 테스트와 문서

[tests 폴더](</C:/논문/adversarial_robust_detector/tests>)에는 다음 테스트가 있습니다.

- 공격 모듈 테스트
- 기본 모델 테스트
- Proposed v1 테스트
- Recall/IoU 지표 테스트

문서는 다음과 같습니다.

- [README.md]: 프로젝트 기본 설명
- [TRANSFER_GUIDE.md]: 다른 컴퓨터로 이전하는 방법
- [TWO_PC_WORKFLOW.md]: 두 PC 작업 절차
- [research_model_design.docx]: 연구 모델 설계 문서

## Git 상태와 용량

- Git 추적 파일: 638개
- `runs` 추적 파일: 532개
- 현재 브랜치: `main`
- `origin/main`과 동기화됨
- 미커밋 변경사항 없음
- `.pt` 체크포인트는 Git LFS로 관리

로컬 용량은 대략 다음과 같습니다.

- `.venv`: 약 4.8GB — Git 제외
- `runs`: 약 776MB — Git 포함
- `.git`: 약 719MB — Git/LFS 내부 데이터
- `artifacts`: 약 12.5MB
- 소스코드는 대부분 수십 KB 수준


# Adversarial Robust Person Detector

Research code for evaluating general COCO object-detector robustness to unseen
adversarial patches, while retaining the original person-focused experiment as
an auxiliary result.

## Scope

- Main Baseline: official COCO-pretrained YOLO11n (`yolo11n.pt`), no clean
  re-training, evaluated on all 80 COCO classes.
- Baseline + Adversarial Training: 80-class fine-tuning from the official
  pretrained weights with clean/synthetic-patch mixtures.
- Proposed RGB/Texture + Shape/Edge Model: 80-class shape-aware fusion.
- Auxiliary experiment: the completed person-only runs and checkpoints remain
  preserved under their existing `runs/` paths.
- Seen Patch and Unseen Patch evaluation: train patches and held-out patches are separated.
- Physical-world evaluation: AdvT-shirt-1K is not used for training or model selection.
- CrowdHuman is optional for occlusion and crowded-scene shape-branch experiments.

## Environment and data

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

COCO is intentionally not stored in GitHub. The portable default expects it at
`../datasets/coco`. For a different PC-specific path, copy
`dataset_config.local.example.yaml` to the ignored `dataset_config.local.yaml`.

## Smoke test and Baseline

```powershell
python -m adversarial_robust_detector.train_baseline --config dataset_config.local.yaml --max-images 100 --val-max-images 100 --epochs 1 --image-size 640 --batch-size 4 --lr 0.0001 --val-every 1 --num-workers 2 --pin-memory --persistent-workers --output runs\baseline_clean_smoke_100
```

The two-PC development/full-training workflow and current commands are in
`TWO_PC_WORKFLOW.md`. Each run records its source commit, GPU, dependencies,
data configuration, and training arguments in `run_metadata.json`.

The main 80-class dataset configuration is `dataset_config.main80.yaml`.
Evaluate the official pretrained clean Baseline with
`python -m adversarial_robust_detector.evaluate_coco`; install
`pycocotools` first for official COCO AP/AP50/AP75 and per-class AP.

Gradient-based patch generation and the 80-class clean/patch fine-tuning entry
point are documented in `TWO_PC_WORKFLOW.md`. Existing person-only runs remain
untouched.

## Outputs

Each run may contain `last.pt`, `best_map.pt`, `best_loss.pt`, and
`metrics.jsonl`. Checkpoints and experiment outputs are ignored by default to
avoid accidentally committing large or private artifacts. A selected baseline
checkpoint may be copied to `artifacts/checkpoints/` and tracked with Git LFS.
