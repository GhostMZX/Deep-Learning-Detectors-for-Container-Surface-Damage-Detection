import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from ultralytics import YOLO

PAPER_ROOT = Path(r"D:\projects\mathorcup\paper")
DATA_YAML = PAPER_ROOT / "dataset" / "dataset.yaml"
RESULTS_DIR = PAPER_ROOT / "results"
WEIGHTS_DIR = PAPER_ROOT / "weights"
CONFIGS_DIR = PAPER_ROOT / "configs"


def apply_loss(name):
    if name == "ciou":
        return
    import ultralytics.utils.loss as ul
    if name == "wiou":
        from wiou_loss import BboxLossWIoU
        ul.BboxLoss = BboxLossWIoU
    else:
        raise ValueError(f"unknown loss: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", default=str(DATA_YAML), help="dataset yaml path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--loss", default="ciou", choices=["ciou", "wiou"])
    p.add_argument("--pretrained", default="yolo11m.pt",
                   help="weights to .load() when --model is a yaml; empty string disables")
    p.add_argument("--extra", default="", help="key=val pairs comma separated, e.g. copy_paste=0.1,mosaic=1.0")
    p.add_argument("--resume", action="store_true", help="resume from results/<name>_s<seed>/weights/last.pt")
    args = p.parse_args()

    apply_loss(args.loss)

    model_path = args.model
    for cand in (Path(model_path), WEIGHTS_DIR / model_path, CONFIGS_DIR / model_path):
        if cand.exists():
            model_path = str(cand)
            break

    extra = {}
    for kv in filter(None, args.extra.split(",")):
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        extra[k] = v

    run_name = f"{args.name}_s{args.seed}"
    t0 = time.time()

    last_pt = RESULTS_DIR / run_name / "weights" / "last.pt"
    if args.resume and last_pt.exists():
        model = YOLO(str(last_pt))
        model.train(resume=True)
        metrics = model.val()
        _save_summary(args, run_name, str(last_pt), metrics, t0)
        return 0

    model = YOLO(model_path)
    if model_path.endswith(".yaml") and args.pretrained:
        pt = WEIGHTS_DIR / args.pretrained
        model.load(str(pt))
        print(f"transferred pretrained weights from {pt}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        project=str(RESULTS_DIR),
        name=run_name,
        workers=args.workers,
        close_mosaic=10,
        amp=True,
        **extra,
    )
    metrics = model.val()
    _save_summary(args, run_name, model_path, metrics, t0)
    return 0


def _save_summary(args, run_name, model_path, metrics, t0):
    summary = {
        "name": run_name,
        "model": model_path,
        "seed": args.seed,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_mAP50": {
            n: float(v) for n, v in zip(metrics.names.values(), metrics.box.ap50)
        },
        "wall_minutes": round((time.time() - t0) / 60, 1),
    }
    out = RESULTS_DIR / f"{run_name}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
