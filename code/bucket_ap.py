import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

DATASET = Path(r"D:\projects\mathorcup\paper\dataset")
IMG_AREA = 1280 * 720
SMALL, MEDIUM = 32 * 32 / IMG_AREA, 96 * 96 / IMG_AREA


def bucket_of(area_ratio):
    if area_ratio < SMALL:
        return "small"
    if area_ratio < MEDIUM:
        return "medium"
    return "large"


def load_gt():
    gt = {}
    for txt in sorted((DATASET / "labels" / "val").glob("*.txt")):
        boxes = []
        for ln in txt.read_text().splitlines():
            parts = ln.split()
            if len(parts) == 5:
                c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                boxes.append({"cls": c, "cx": cx, "cy": cy, "w": w, "h": h, "area": w * h})
        gt[txt.stem] = boxes
    return gt


def iou(a, b):
    ax1, ay1, ax2, ay2 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2, a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1, bx2, by2 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a["area"] + b["area"] - inter
    return inter / union if union > 0 else 0.0


def ap_at_iou(preds, gts_by_img, iou_thr):
    n_gt = sum(len(v) for v in gts_by_img.values())
    if n_gt == 0:
        return None
    matched = {k: set() for k in gts_by_img}
    tp_fp = []
    for p in sorted(preds, key=lambda x: -x["conf"]):
        gts = gts_by_img.get(p["img"], [])
        used = matched[p["img"]]
        best, best_iou = -1, iou_thr
        for gi, g in enumerate(gts):
            if gi in used or g["cls"] != p["cls"]:
                continue
            v = iou(p, g)
            if v >= best_iou:
                best, best_iou = gi, v
        if best >= 0:
            used.add(best)
            tp_fp.append(1)
        else:
            tp_fp.append(0)
    tp_fp = np.array(tp_fp)
    cum_tp = np.cumsum(tp_fp)
    cum_fp = np.cumsum(1 - tp_fp)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(model_path, iou_thr=0.5, conf=0.001, device="cpu"):
    model = YOLO(model_path)
    gt_all = load_gt()
    preds_all = []
    for stem, boxes in gt_all.items():
        img = DATASET / "images" / "val" / f"{stem}.jpg"
        r = model.predict(str(img), conf=conf, iou=0.6, verbose=False, device=device)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = (x2 - x1) / 1280, (y2 - y1) / 720
            preds_all.append({"img": stem, "cls": int(b.cls[0]), "conf": float(b.conf[0]),
                              "cx": (x1 + x2) / 2 / 1280, "cy": (y1 + y2) / 2 / 720,
                              "w": w, "h": h, "area": w * h})

    out = {}
    for bucket in ["small", "medium", "large"]:
        gt_b = {k: [g for g in v if bucket_of(g["area"]) == bucket] for k, v in gt_all.items()}
        pr_b = [p for p in preds_all if bucket_of(p["area"]) == bucket]
        out[f"AP50_{bucket}"] = ap_at_iou(pr_b, gt_b, iou_thr)
        out[f"n_gt_{bucket}"] = sum(len(v) for v in gt_b.values())
    out["AP50_all"] = ap_at_iou(preds_all, gt_all, iou_thr)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = evaluate(args.weights, device=args.device)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
