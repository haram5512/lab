"""Measure per-image YOLO latency for the three research checkpoints."""
import json, os, time
from pathlib import Path
import torch
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parent / 'runs' / '.ultralytics'))
from baseline import BaselineDetector

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs' / 'main_80class' / 'adversarial_training_final_evaluation' / 'latency_benchmark.json'
MODELS = {
    'pretrained': ROOT / 'yolo11n.pt',
    'best_clean_map': ROOT / 'runs' / 'main_80class' / 'v5c_adversarial_training_corrected' / 'best_clean_map.pt',
    'best_seen_map': ROOT / 'runs' / 'main_80class' / 'v5c_adversarial_training_corrected' / 'best_seen_map.pt',
}

def load_model(path, device):
    if path.name == 'yolo11n.pt':
        model = BaselineDetector(str(path))
    else:
        model = BaselineDetector('yolo11n.pt')
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state, strict=True)
    return model.to(device).eval()

def measure(model, x, reps=30, warmup=10):
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.forward(x)
        torch.cuda.synchronize()
        fwd = []
        for _ in range(reps):
            start = time.perf_counter(); _ = model.forward(x); torch.cuda.synchronize()
            fwd.append((time.perf_counter() - start) * 1000)
        pred = []
        for _ in range(warmup):
            _ = model.predict(x, confidence=0.25, iou=0.7, classes=[0])
        torch.cuda.synchronize()
        for _ in range(reps):
            start = time.perf_counter(); _ = model.predict(x, confidence=0.25, iou=0.7, classes=[0]); torch.cuda.synchronize()
            pred.append((time.perf_counter() - start) * 1000)
    def stats(a):
        t = torch.tensor(a)
        return {'median_ms': float(t.median()), 'mean_ms': float(t.mean()), 'p95_ms': float(torch.quantile(t, .95)), 'fps': float(1000.0 / t.median())}
    return {'forward_only': stats(fwd), 'forward_plus_nms': stats(pred)}

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this benchmark')
    device = torch.device('cuda:0')
    results = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'image_size': [1,3,640,640], 'confidence': .25, 'nms_iou': .7, 'repetitions': 30, 'warmup': 10, 'models': {}}
    x = torch.rand((1, 3, 640, 640), device=device)
    for name, path in MODELS.items():
        model = load_model(path, device)
        results['models'][name] = {'checkpoint': str(path), **measure(model, x)}
        del model; torch.cuda.empty_cache()
    OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))

if __name__ == '__main__': main()
