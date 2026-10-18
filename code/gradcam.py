import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO

VAL_IMG = Path(r"D:\projects\mathorcup\paper\dataset\images\val")
LEVELS = {8: (16, 80), 16: (19, 40), 32: (22, 20)}


class DetGradCAM:
    def __init__(self, weights):
        self.model = YOLO(weights)
        m = self.model.model
        self.act = {}
        self.grad = {}
        for stride, (idx, _) in LEVELS.items():
            layer = m.model[idx]
            layer.register_forward_hook(self._mk_fwd(stride))
            layer.register_full_backward_hook(self._mk_bwd(stride))

    def _mk_fwd(self, s):
        def hook(mod, inp, out):
            self.act[s] = out
        return hook

    def _mk_bwd(self, s):
        def hook(mod, gin, gout):
            self.grad[s] = gout[0]
        return hook

    def cam_for_detections(self, img_path, imgsz=640, conf=0.25, max_det=3):
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.tal import make_anchors

        img = cv2.imread(str(img_path))
        oh, ow = img.shape[:2]
        lb = LetterBox((imgsz, imgsz), auto=False, stride=32)
        inp_img = lb(image=img)
        ratio = min(imgsz / ow, imgsz / oh)
        cw, ch = int(ow * ratio), int(oh * ratio)

        self.model.model.eval()
        det = self.model.predict(inp_img, conf=conf, verbose=False, device="cpu")[0]
        if len(det.boxes) == 0:
            return None, img, []
        keep = det.boxes.conf.argsort(descending=True)[:max_det]
        sel = det.boxes[keep]

        inp = torch.from_numpy(inp_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        inp.requires_grad_(True)
        self.model.model.train()
        preds = self.model.model(inp)
        scores = preds["scores"]

        strides_t = torch.tensor([8, 16, 32], dtype=torch.float32)
        anchors, strides = make_anchors(preds["feats"], strides_t, 0.5)
        anchor_px = anchors * strides
        n_p3 = 80 * 80
        n_p4 = 40 * 40

        cams = []
        labels = []
        for i in range(len(sel)):
            bx = sel.xyxy[i].cpu().numpy()
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            d = (anchor_px[:, 0] - cx).abs() + (anchor_px[:, 1] - cy).abs()
            aidx = int(d.argmin())
            if aidx < n_p3:
                stride, gy, gx = 8, aidx // 80, aidx % 80
            elif aidx < n_p3 + n_p4:
                j = aidx - n_p3
                stride, gy, gx = 16, j // 40, j % 40
            else:
                j = aidx - n_p3 - n_p4
                stride, gy, gx = 32, j // 20, j % 20

            cls_idx = int(sel.cls[i])
            score = scores[0, cls_idx, aidx]
            self.model.model.zero_grad()
            score.backward(retain_graph=True)

            a = self.act[stride][0]
            g = self.grad[stride][0]
            cam = torch.relu((g * a).sum(0)).detach().cpu().numpy()
            cell_cam = np.zeros_like(cam)
            r = 2
            y0, y1 = max(0, gy - r), min(cam.shape[0], gy + r + 1)
            x0, x1 = max(0, gx - r), min(cam.shape[1], gx + r + 1)
            cell_cam[y0:y1, x0:x1] = cam[y0:y1, x0:x1]
            cams.append(cell_cam)
            labels.append(self.model.names[cls_idx])

        cam = np.max(np.stack(cams), axis=0)
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=2)
        if cam.max() > 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (imgsz, imgsz))
        cam = cam[:ch, :cw]
        cam_full = cv2.resize(cam, (ow, oh))
        boxes_orig = []
        for i in range(len(sel)):
            bx = sel.xyxy[i].cpu().numpy()
            boxes_orig.append((int(bx[0] / ratio), int(bx[1] / ratio), int(bx[2] / ratio), int(bx[3] / ratio),
                               int(sel.cls[i]), float(sel.conf[i])))
        return cam_full, img, labels, boxes_orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = DetGradCAM(args.weights)

    for stem in args.images.split(","):
        img_path = VAL_IMG / f"{stem}.jpg"
        cam, img, labels, boxes_orig = gen.cam_for_detections(img_path, conf=args.conf)
        if cam is None:
            print(stem, "no detections, skipped")
            continue
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
        names = ["broken", "hd", "hole", "rusty", "scratch", "sd", "vd"]
        colors = [(255, 80, 80), (255, 160, 60), (80, 160, 255), (200, 120, 255), (255, 220, 80), (80, 220, 160), (180, 180, 180)]
        for x1, y1, x2, y2, c, cf in boxes_orig:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colors[c % len(colors)], 3)
            cv2.putText(overlay, f"{names[c]} {cf:.2f}", (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"{stem}_cam.jpg"), overlay)
        print(stem, "labels:", labels)


if __name__ == "__main__":
    main()
