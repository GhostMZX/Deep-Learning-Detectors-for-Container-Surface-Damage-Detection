import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

ROBUST_ROOT = Path(r"D:\projects\mathorcup\paper\dataset_robust")
OUT = Path(r"D:\projects\mathorcup\paper\results\robustness")
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = ["L1", "L2", "L3"]
DEGS = ["gaussian_noise", "motion_blur", "low_light"]


def eval_one(model, deg, level, device):
    img_dir = ROBUST_ROOT / deg / level / "images"
    tmp_yaml = OUT / f"tmp_{deg}_{level}.yaml"
    tmp_yaml.write_text(yaml.safe_dump({
        "path": str(ROBUST_ROOT / deg / level).replace("\\", "/"),
        "train": "images", "val": "images", "nc": 7,
        "names": ["broken", "hd", "hole", "rusty", "scratch", "sd", "vd"],
    }), encoding="utf-8")
    m = model.val(data=str(tmp_yaml), split="val", device=device, verbose=False)
    tmp_yaml.unlink()
    return float(m.box.map50), float(m.box.map)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--clean-map50", type=float, required=True)
    p.add_argument("--clean-map5095", type=float, required=True)
    args = p.parse_args()

    model = YOLO(args.weights)
    result = {"tag": args.tag, "clean": [args.clean_map50, args.clean_map5095]}
    for deg in DEGS:
        result[deg] = {"mAP50": [], "mAP50-95": []}
        for lv in LEVELS:
            m50, m95 = eval_one(model, deg, lv, args.device)
            result[deg]["mAP50"].append(round(m50, 4))
            result[deg]["mAP50-95"].append(round(m95, 4))
            print(f"{args.tag} {deg}/{lv}: mAP50={m50:.4f}", flush=True)
    out_file = OUT / f"{args.tag}_robust.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved ->", out_file)


if __name__ == "__main__":
    main()
