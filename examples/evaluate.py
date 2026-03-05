from __future__ import annotations

import argparse

from ultralytics import YOLO

"""
Evaluate YOLOv8 models: mAP (val) and simple performance benchmarking.
Use CPU by default on unsupported GPUs.
"""


def run_map(model_path: str, data: str, imgsz: int, device: str, conf: float, classes: list[int] | None = None):
    model = YOLO(model_path)
    metrics = model.val(data=data, imgsz=imgsz, device=device, conf=conf)
    print("=== mAP Metrics ===")
    try:
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP75: {metrics.box.map75:.4f}")
        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            print(f"Per-class mAP50-95 (first 10): {metrics.box.maps[:10]}")
            # Optional: subset mAP for selected class IDs
            if classes:
                subset_values = [metrics.box.maps[i] for i in classes if i < len(metrics.box.maps)]
                if subset_values:
                    subset_map = sum(subset_values) / len(subset_values)
                    print(f"Subset mAP50-95 (classes={classes}): {subset_map:.4f}")
    except Exception:
        print("Could not parse metrics; raw object:")
        print(metrics)


def run_perf(
    model_path: str,
    source: str,
    device: str,
    conf: float,
    runs: int = 50,
    warmup: int = 10,
    classes: list[int] | None = None,
):
    import time
    from pathlib import Path

    model = YOLO(model_path)
    p = Path(source)
    if p.is_dir():
        paths = sorted([str(f) for f in p.glob("*.jpg")])
    else:
        paths = [str(p)]

    for s in paths[:warmup]:
        _ = model.predict(source=s, device=device, conf=conf, verbose=False, classes=classes)

    times = []
    for s in paths[:runs]:
        start = time.perf_counter()
        _ = model.predict(source=s, device=device, conf=conf, verbose=False, classes=classes)
        times.append((time.perf_counter() - start) * 1000.0)

    if times:
        print("=== Performance Metrics ===")
        print(f"Samples: {len(times)}")
        print(f"Avg latency (ms): {sum(times) / len(times):.2f}")
        print(f"Min latency (ms): {min(times):.2f}")
        print(f"Max latency (ms): {max(times):.2f}")
    else:
        print("No samples measured.")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOv8: mAP and performance")
    p.add_argument("--model", default="yolov8n.pt", help="Weights (.pt or .engine)")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--data", default="coco128.yaml", help="Ultralytics data config or path to your YAML")
    p.add_argument("--perf_source", default=None, help="Image path or folder for performance benchmark")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--classes", default=None, help="Comma-separated COCO class IDs to focus on (e.g., 0,63,67,47,73)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classes = None
    if args.classes:
        classes = [int(x) for x in str(args.classes).split(",") if x.strip().isdigit()]
    try:
        run_map(args.model, args.data, args.imgsz, args.device, args.conf, classes)
    except Exception as e:
        print(f"mAP evaluation skipped: {e}")
    if args.perf_source:
        run_perf(args.model, args.perf_source, args.device, args.conf, args.runs, args.warmup, classes)
    else:
        print("No perf_source provided; skipping performance benchmark.")
