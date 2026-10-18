import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

VAL_IMG = Path(r"D:\projects\mathorcup\paper\dataset\images\val")
VAL_LBL = Path(r"D:\projects\mathorcup\paper\dataset\labels\val")
OUT_ROOT = Path(r"D:\projects\mathorcup\paper\results\det_vis")
CLASS_NAMES = ["broken", "hd", "hole", "rusty", "scratch", "sd", "vd"]
GT_COLOR = (0, 200, 0)
PRED_COLORS = [(255, 80, 80), (255, 160, 60), (80, 160, 255), (200, 120, 255), (255, 220, 80), (80, 220, 160), (180, 180, 180)]


def draw_gt(img, stem):
    lbl = VAL_LBL / f"{stem}.txt"
    h, w = img.shape[:2]
    for ln in lbl.read_text().splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(img, f"GT:{CLASS_NAMES[c]}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_COLOR, 2)
    return img


def draw_pred(img, model, conf):
    h, w = img.shape[:2]
    r = model.predict(img, conf=conf, verbose=False, device="cpu")[0]
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        c = int(b.cls[0]); cf = float(b.conf[0])
        color = PRED_COLORS[c % len(PRED_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{CLASS_NAMES[c]}:{cf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--ours", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    base = YOLO(args.base)
    ours = YOLO(args.ours)
    stems = args.images.split(",")

    for stem in stems:
        src = cv2.imread(str(VAL_IMG / f"{stem}.jpg"))
        gt = draw_gt(src.copy(), stem)
        pb = draw_pred(src.copy(), base, args.conf)
        po = draw_pred(src.copy(), ours, args.conf)

        def tag(im, text):
            cv2.rectangle(im, (0, 0), (220, 30), (255, 255, 255), -1)
            cv2.putText(im, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            return im

        row = cv2.hconcat([tag(gt, "Ground Truth"), tag(pb, "Baseline"), tag(po, "Ours")])
        out_dir = OUT_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}_cmp.jpg"), row)
        print("saved", stem)


if __name__ == "__main__":
    main()
