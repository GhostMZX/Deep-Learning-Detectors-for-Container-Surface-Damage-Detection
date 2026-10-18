import shutil
import sys
from pathlib import Path

CUSTOM_SRC = Path(r"D:\projects\mathorcup\paper\modules\custom_modules.py")
MARKERS = ["C3k2_LSK", "C3k2_CA"]


def find_ultralytics():
    import ultralytics
    return Path(ultralytics.__file__).parent


def patch():
    u = find_ultralytics()
    modules_dir = u / "nn" / "modules"
    dst = modules_dir / "custom_lsk.py"
    if not dst.exists():
        shutil.copy2(CUSTOM_SRC, dst)
        print(f"copied -> {dst}")

    tasks = u / "nn" / "tasks.py"
    text = tasks.read_text(encoding="utf-8")
    missing = [m for m in MARKERS if m not in text]
    if missing:
        imp = "from ultralytics.nn.modules.custom_lsk import C3k2_LSK, C3k2_CA\n"
        anchor = "from ultralytics.nn.modules import ("
        assert anchor in text, "import anchor not found"
        if "from ultralytics.nn.modules.custom_lsk import" not in text:
            text = text.replace(anchor, imp + anchor, 1)
        else:
            text = text.replace(
                "from ultralytics.nn.modules.custom_lsk import C3k2_LSK\n",
                imp, 1)

        for m in missing:
            base_anchor = "            A2C2f,\n        }\n    )"
            text = text.replace(base_anchor, f"            A2C2f,\n            {m},\n        }}\n    )", 1)
            rep_anchor = "            A2C2f,\n        }\n    )"
            text = text.replace(rep_anchor, f"            A2C2f,\n            {m},\n        }}\n    )", 1)

        tasks.write_text(text, encoding="utf-8")
        print(f"patched -> {tasks} with {missing}")
    else:
        print("tasks.py already patched")
    print("PATCH OK")


if __name__ == "__main__":
    patch()
    sys.path.insert(0, str(Path(__file__).parent))
    from ultralytics import YOLO
    m = YOLO(r"D:\projects\mathorcup\paper\configs\yolo11m-lsk.yaml")
    print("yaml parse OK, params:", sum(p.numel() for p in m.parameters()))
