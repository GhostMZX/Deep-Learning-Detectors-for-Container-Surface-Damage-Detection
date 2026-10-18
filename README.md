# Code and Data Availability — Container Damage Detection Benchmark

This capsule contains all code needed to reproduce the experiments in:

## Contents

| Path | Description |
|------|-------------|
| `code/train_run.py` | Single training run with fixed protocol (150 epochs, 640 px, batch 8, 3-seed capable). |
| `code/run_benchmark.py` | Benchmark queue: 9 detectors x 3 seeds. |
| `code/patch_ultralytics.py` | Registers custom attention modules (C3k2_LSK / C3k2_EMA / C3k2_ECBAM / ...) into Ultralytics. Run once after install. |
| `code/modules/custom_modules.py` | Implementations of LSK, EMA, ECBAM, SE, CBAM, CA attention blocks. |
| `code/configs/*.yaml` | Model definitions used in the study. |
| `code/coco_eval.py` | COCO-style (pycocotools) evaluation with per-scale AP. |
| `code/eval_robust.py` | Robustness evaluation under noise / blur / low-light corruptions. |
| `code/bucket_ap.py`, `code/gradcam.py`, `code/det_vis.py` | Analysis tools (size-bucketed AP, Grad-CAM, detection visualization). |
| `code/download_weights.sh` | Downloads all pretrained weights used in the study. |

## Data

The **target dataset** (1,271 images, 6 categories) derives from a container-damage
competition and is not redistributed here. The **source dataset** for the transfer
learning experiments (3,300 images) is public:
https://github.com/CXJ-jzx/Container-Surface-Defect-Detection-Based-on-Deep-Learning

Prepare the data in YOLO format and point `DATA_YAML` in `train_run.py` to it.

## Reproduction Steps

```bash
# 1. Environment (see environment.yml): Python 3.11, torch 2.5.1+cu121, ultralytics 8.4.135
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.4.135 pycocotools pandas matplotlib PyMuPDF opencv-python pillow numpy

# 2. One-time patch: register custom modules
python code/patch_ultralytics.py

# 3. Download pretrained weights
bash code/download_weights.sh        # -> weights/

# 4. Run any experiment (example: baseline, seed 0)
python code/train_run.py --model yolo11m.pt --name baseline --seed 0 --epochs 150 --batch 8

# 5. Benchmark queue (9 detectors x 3 seeds)
python code/run_benchmark.py
```

Every run writes `results/<name>_s<seed>_summary.json` with all reported metrics.
Results in the paper are mean +/- std over seeds {0, 1, 2} of these summaries.
