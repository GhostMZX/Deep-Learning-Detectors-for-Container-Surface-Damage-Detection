#!/bin/bash
# Download all pretrained weights used in the study (YOLOv5su..RT-DETR, yolo11 variants).
# Mirrors listed because github.com may be unreachable in some regions; any working
# GitHub mirror or direct download is fine.
set -e
mkdir -p weights
BASE=${1:-https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download}
for w in yolov5su:v0.0.0 yolov8n:v8.3.0 yolov8s:v8.3.0 yolov8m:v8.3.0 yolov9c:v8.3.0 \
         yolov10m:v8.3.0 yolo11n:v8.3.0 yolo11s:v8.3.0 yolo11m:v8.3.0 rtdetr-l:v8.2.0; do
  name="${w%%:*}"; tag="${w##*:}"
  echo "downloading $name.pt"
  curl -L -o "weights/$name.pt" "$BASE/$tag/$name.pt" --retry 3 --max-time 600
done
ls -lh weights/
