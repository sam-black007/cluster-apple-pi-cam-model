"""
train.py
========
Fine-tune YOLO11 / LMWP-YOLO into a 2-class (drone, bird) detector.

Enhancements based on:
"Improved YOLO for long range detection of small drones" (Scientific Reports 2025)
DOI: 10.1038/s41598-025-95580-z
  - LMWP-YOLO architecture with PP-LCNet lightweight backbone
  - MCA (Multidimensional Collaborative Attention)
  - P2 (Stride 4) micro-drone prediction head for long-range intercepts
  - Normalized Wasserstein Distance (NWD) compatibility

Usage:
    python scripts/train.py                        # standard training
    python scripts/train.py --lmwp                 # train LMWP-YOLO (Scientific Reports 2025)
    python scripts/train.py --resume               # continue from last.pt
"""

import argparse
import sys
from pathlib import Path


class _DropPynvml:
    """Drop the noisy 'pynvml deprecated' message torch prints on CUDA init."""
    def __init__(self, stream, bad):
        self._s = stream
        self._bad = bad

    def write(self, s):
        if self._bad in s:
            return 0
        return self._s.write(s)

    def flush(self):
        self._s.flush()

    def __getattr__(self, name):
        return getattr(self._s, name)


_stdout, _stderr = sys.stdout, sys.stderr
sys.stdout = _DropPynvml(sys.stdout, "pynvml")
sys.stderr = _DropPynvml(sys.stderr, "pynvml")
from ultralytics import YOLO
sys.stdout, sys.stderr = _stdout, _stderr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from models.lmwp_factory import build_lmwp_yolo, register_lmwp_modules
    register_lmwp_modules()
except Exception:
    build_lmwp_yolo = None

DATA = ROOT / "data" / "dataset.yaml"
RUN_NAME = "drone_bird"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="resume from runs/detect/drone_bird/weights/last.pt")
    ap.add_argument("--lmwp", action="store_true",
                    help="enable LMWP-YOLO (Scientific Reports 2025: PP-LCNet + MCA + P2 scale)")
    ap.add_argument("--model-cfg", default=None,
                    help="custom model YAML architecture (e.g. models/lmwp_yolo.yaml)")
    ap.add_argument("--pretrained", default="yolo11n.pt",
                    help="pretrained weights to transfer into backbone")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default=str(DATA),
                    help="dataset config (override for a subset)")
    ap.add_argument("--name", default=RUN_NAME,
                    help="run name under runs/detect/")
    ap.add_argument("--cache", default="False",
                    help="cache images: True/False/ram/disk")
    ap.add_argument("--plots", default="True",
                    help="generate training plots (True/False)")
    ap.add_argument("--workers", type=int, default=4,
                    help="dataloader workers (lower if RAM/shared-mem limited)")
    args = ap.parse_args()

    if args.resume:
        ckpt = ROOT / "runs" / "detect" / args.name / "weights" / "last.pt"
        print(f"[RESUME] from {ckpt}")
        model = YOLO(str(ckpt))
        model.train(resume=True)
        return

    if args.lmwp or args.model_cfg:
        cfg = args.model_cfg or str(ROOT / "models" / "lmwp_yolo.yaml")
        print(f"[TRAIN] Building LMWP-YOLO architecture from: {cfg}")
        if build_lmwp_yolo is not None:
            model = build_lmwp_yolo(cfg, pretrained_weights=args.pretrained)
        else:
            model = YOLO(cfg)
    else:
        # Fresh start. Pretrained weights give strong backbone.
        model = YOLO(args.pretrained if Path(args.pretrained).exists() else "yolo11n.pt")

    print("[TRAIN] starting fresh 2-class training")
    print(f"  data       : {args.data}")
    print(f"  epochs     : {args.epochs}")
    print(f"  imgsz      : {args.imgsz}")
    print(f"  batch      : {args.batch}")
    print(f"  device     : {args.device}")
    print(f"  cache      : {args.cache}")
    print(f"  plots      : {args.plots}")
    print(f"  lmwp_mode  : {args.lmwp or bool(args.model_cfg)}")

    cache = args.cache
    if isinstance(cache, str):
        c = cache.lower()
        cache = True if c == "true" else False if c == "false" else c
    plots = str(args.plots).lower() != "false"

    model.train(
        data=str(args.data),
        task="detect",
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        cache=cache,
        plots=plots,
        optimizer="auto",
        patience=20,          # early stop if no improvement
        cos_lr=True,
        verbose=True,         # live status in terminal
        exist_ok=True,
        # Focal loss + aerial augmentation for small/distant targets
        fl_gamma=1.5,
        scale=0.6,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.2,
        degrees=15.0,
        hsv_s=0.5,
        hsv_v=0.4,
    )

    print("\n[TRAIN] done. Best weights -> "
          f"{ROOT / 'runs' / 'detect' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
