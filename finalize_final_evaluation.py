import hashlib, json, subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / 'runs' / 'main_80class' / 'adversarial_training_final_evaluation'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

comparison = json.loads((out / 'comparison.json').read_text(encoding='utf-8'))
tradeoff = json.loads((out / 'tradeoff_summary.json').read_text(encoding='utf-8'))
models = {
    'pretrained': root / 'yolo11n.pt',
    'best_clean_map': root / 'runs' / 'main_80class' / 'v5c_adversarial_training_corrected' / 'best_clean_map.pt',
    'best_seen_map': root / 'runs' / 'main_80class' / 'v5c_adversarial_training_corrected' / 'best_seen_map.pt',
}
for name, c in comparison.items():
    seen_ap = c['seen_absolute_improvement']['person_ap']
    unseen_ap = c['unseen_absolute_improvement']['person_ap']
    seen_recall = c['seen_absolute_improvement']['person_recall']
    unseen_recall = c['unseen_absolute_improvement']['person_recall']
    clean_ap_loss = c['clean_degradation']['person_ap']
    if seen_ap > 0 and unseen_ap >= 0 and clean_ap_loss <= 0.02:
        verdict = 'GO'
    elif seen_recall > 0 and unseen_recall > 0:
        verdict = 'PARTIAL'
    else:
        verdict = 'NO-GO'
    c['verdict'] = verdict
tradeoff['checkpoints'] = comparison
tradeoff['final_verdict'] = 'PARTIAL'
tradeoff['verdict_basis'] = ('Both adversarial checkpoints improve person Recall on seen and unseen patched images, '
                             'but person AP/AP50 decline versus pretrained and clean AP also declines; '
                             'therefore robustness gain is a tradeoff, not a full GO.')
(out / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
(out / 'tradeoff_summary.json').write_text(json.dumps(tradeoff, indent=2), encoding='utf-8')
commit = subprocess.check_output(['git', '-c', f'safe.directory={root}', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
metadata = {
    'git_commit': commit,
    'config': str(root / 'dataset_config.main80.yaml'),
    'dataset': 'COCO 2017 val2017 full image set, all 80 classes with person diagnostics',
    'person_gt_expected': 11004,
    'image_size': 640, 'confidence': 0.001, 'nms_iou': 0.7,
    'models': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in models.items()},
    'device': 'cuda:0',
    'results': str(out),
    'final_verdict': 'PARTIAL',
}
(out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(json.dumps({'git_commit': commit, 'final_verdict': 'PARTIAL', 'models': list(models)}, indent=2))
