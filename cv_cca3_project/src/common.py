"""Shared helpers used by all tasks (paths, loading, saving).

You normally never run this file directly.
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # figures are saved to files, no pop-up windows needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Project root = the folder that contains main.py
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Phone photos are huge (4000x3000). We shrink them for faster processing.
MAX_SIDE = 1000


def find_input(stem, exts):
    """Return the path of input/<stem>.<any allowed extension>, or None."""
    for p in sorted(INPUT_DIR.glob(stem + ".*")):
        if p.suffix.lower() in exts:
            return p
    return None


def require_input(stem, exts, description):
    """Same as find_input but raises a clear error if the file is missing."""
    p = find_input(stem, exts)
    if p is None:
        raise FileNotFoundError(
            f"Missing input file '{stem}.*' ({description}).\n"
            f"        Copy your file into this folder: {INPUT_DIR}\n"
            f"        Allowed extensions: {sorted(exts)}"
        )
    return p


def load_image(path, max_side=MAX_SIDE):
    """Read an image (BGR). Large images are shrunk so the longest side <= max_side."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # EXIF rotation is applied by OpenCV
    if img is None:
        raise ValueError(f"OpenCV could not read the image: {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def task_dir(task_name, *sub):
    """Create (if needed) and return outputs/<task_name>/<sub...>."""
    d = OUTPUT_DIR.joinpath(task_name, *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_image(folder, filename, img):
    path = Path(folder) / filename
    if not cv2.imwrite(str(path), img):
        raise IOError(f"Could not write image: {path}")
    return path


def save_figure(fig, path):
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _to_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_metrics(name, data):
    """Store measured numbers as JSON; the report generator reads these files."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_to_python)
    return path


def load_metrics(name):
    path = METRICS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def label(img, text):
    """Return a copy of img (gray or colour) with a black title bar on top."""
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
