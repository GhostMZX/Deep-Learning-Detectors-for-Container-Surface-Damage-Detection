import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from ultralytics import YOLO

VAL_IMG = Path(r"D:\projects\mathorcup\paper\dataset\images\val")
VAL_LBL = Path(r"D:\projects\mathorcup\paper\dataset\labels\val")
CLASS_NAMES = ["broken", "hd", "hole", "rusty", "scratch", "sd", "vd"]


def build_gt_coco():
    images, annotations, ann_id = [], [], 0
    for i, img_path in enumerate(sorted(VAL_IMG.glob("*.jpg"))):
        w, h = Image.open(img_path).size
        images.append({"id": i, "file_name": img_path.name, "width": w, "height": h})
        lbl = VAL_LBL / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        for ln in lbl.read_text().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            bw_px, bh_px = bw * w, bh * h
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": c,
                "bbox": [x1, y1, bw_px, bh_px], "area": bw_px * bh_px, "iscrowd": 0,
            })
            ann_id += 1
    cats = [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]
    return {"images": images, "annotations": annotations, "categories": cats}


def predict_coco(model, device, conf=0.001):
    preds = []
    img_ids = {p.stem: i for i, p in enumerate(sorted(VAL_IMG.glob("*.jpg")))}
    for img_path in sorted(VAL_IMG.glob("*.jpg")):
        r = model.predict(str(img_path), conf=conf, iou=0.7, max_det=300, verbose=False, device=device)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            preds.append({
                "image_id": img_ids[img_path.stem],
                "category_id": int(b.cls[0]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(b.conf[0]),
            })
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    gt = build_gt_coco()
    model = YOLO(args.weights)
    preds = predict_coco(model, args.device)

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(preds)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats_keys = ["AP50-95_all", "AP50_all", "AP75_all",
                  "AP50-95_small", "AP50-95_medium", "AP50-95_large"]
    res = {args.tag: {k: round(float(v), 4) for k, v in zip(stats_keys, ev.stats[:6])}}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
