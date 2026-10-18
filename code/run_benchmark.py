import subprocess
import sys
import time
from pathlib import Path

PY = r"C:\Users\Administrator\.conda\envs\yolo11\python.exe"
TRAIN = Path(r"D:\projects\mathorcup\paper\scripts\train_run.py")
LOG_DIR = Path(r"C:\Users\ADMINI~1\AppData\Local\Temp\opencode\bench_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

BENCH = [
    ("yolov5su.pt", "bench_yolov5su", 8),
    ("yolov8n.pt", "bench_yolov8n", 8),
    ("yolov8s.pt", "bench_yolov8s", 8),
    ("yolov8m.pt", "bench_yolov8m", 8),
    ("yolov9c.pt", "bench_yolov9c", 4),
    ("yolov10m.pt", "bench_yolov10m", 8),
    ("yolo11n.pt", "bench_yolo11n", 8),
    ("yolo11s.pt", "bench_yolo11s", 8),
    ("rtdetr-l.pt", "bench_rtdetr_l", 2),
]
SEEDS = [0, 1, 2]


def run(model, name, batch, seed):
    log = LOG_DIR / f"{name}_s{seed}.log"
    cmd = [PY, str(TRAIN), "--model", model, "--name", name,
           "--seed", str(seed), "--epochs", "150", "--batch", str(batch),
           "--workers", "4"]
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M')}] {name} s{seed} exit={r.returncode}", flush=True)
    return r.returncode


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for model, name, batch in BENCH:
        if only and name not in only:
            continue
        for seed in SEEDS:
            run(model, name, batch, seed)
    print("ALL DONE", flush=True)
