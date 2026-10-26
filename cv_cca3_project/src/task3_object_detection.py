"""Task 3 - Object Recognition: Viola-Jones (Haar cascade) vs YOLO (Unit 5).

* Haar cascade  : pretrained frontal-face detector that ships with OpenCV.
* YOLO          : pretrained Ultralytics YOLOv8n (COCO, 80 classes). The weights
                  file is downloaded automatically the first time (internet needed).

Both detectors run on: photo1, photo2 and 3 frames taken from your video
(if the video exists). Output = annotated images, side-by-side comparisons, a table.

Run:  python src/task3_object_detection.py
Optional:  python src/task3_object_detection.py --yolo-weights yolov8s.pt --conf 0.25
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common import (IMAGE_EXTS, VIDEO_EXTS, find_input, label, load_image, require_input,
                    save_image, save_metrics, task_dir)

TASK_NAME = "task3_object_recognition"


# --------------------------------------------------------------------- Haar
def load_haar():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            f"Your OpenCV ({cv2.__version__}) has no Haar cascade support (removed in OpenCV 5).\n"
            "        Fix: pip install \"opencv-python>=4.8,<5\"   (then run this task again)")
    name = "haarcascade_frontalface_default.xml"
    candidates = []
    if hasattr(cv2, "data"):
        candidates.append(Path(cv2.data.haarcascades) / name)
    candidates.append(Path(cv2.__file__).parent / "data" / name)
    for c in candidates:
        if c.exists():
            cascade = cv2.CascadeClassifier(str(c))
            if not cascade.empty():
                return cascade, c
    raise FileNotFoundError("Could not find OpenCV's haarcascade_frontalface_default.xml. "
                            "Reinstall with: pip install --force-reinstall opencv-python")


def run_haar(cascade, img):
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    t0 = time.perf_counter()
    rects, _, weights = cascade.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), outputRejectLevels=True)
    ms = (time.perf_counter() - t0) * 1000
    dets = []
    for i, (x, y, w, h) in enumerate(rects if len(rects) else []):
        dets.append({"label": "face", "conf": float(weights[i]), "box": [int(x), int(y), int(x + w), int(y + h)]})
    return dets, ms


# --------------------------------------------------------------------- YOLO
def load_yolo(weights):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Task 3] ultralytics is not installed -> run: pip install ultralytics")
        return None
    try:
        return YOLO(weights)
    except Exception as e:  # download / file problems
        print(f"[Task 3] Could not load YOLO weights '{weights}': {e}")
        return None


def run_yolo(model, img, conf):
    t0 = time.perf_counter()
    res = model.predict(img, conf=conf, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        dets.append({"label": res.names[int(b.cls[0])], "conf": float(b.conf[0]), "box": [x1, y1, x2, y2]})
    return dets, ms


# ------------------------------------------------------------------ drawing
def draw(img, dets, color, haar=False):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{d['label']} {d['conf']:.2f}" if not haar else f"face (score {d['conf']:.1f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, max(th, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def grab_video_frames(video_path, n=3, width=960):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frac in np.linspace(0.2, 0.8, n):
        idx = int(total * frac)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            h, w = fr.shape[:2]
            if w > width:
                fr = cv2.resize(fr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            frames.append((f"video_frame_{idx}", fr))
    cap.release()
    return frames


def run(yolo_weights="yolov8n.pt", conf=0.25):
    sources = []
    p1 = require_input("photo1", IMAGE_EXTS, "your original photograph")
    sources.append(("photo1", load_image(p1, 960)))
    p2 = find_input("photo2", IMAGE_EXTS)
    if p2:
        sources.append(("photo2", load_image(p2, 960)))
    vid = find_input("video", VIDEO_EXTS)
    if vid:
        sources.extend(grab_video_frames(vid))
    else:
        print("[Task 3] No video found - using photos only (add input/video.mp4 for video frames).")

    cascade, haar_file = load_haar()
    model = load_yolo(yolo_weights)

    out_haar = task_dir(TASK_NAME, "haar_results")
    out_yolo = task_dir(TASK_NAME, "yolo_results")
    out_cmp = task_dir(TASK_NAME, "side_by_side_comparison")

    rows = []
    for name, img in sources:
        h_dets, h_ms = run_haar(cascade, img)
        h_img = draw(img, h_dets, (0, 160, 0), haar=True)
        save_image(out_haar, f"{name}_haar.png", h_img)

        row = {"source": name, "haar_faces": len(h_dets), "haar_ms": round(h_ms, 1),
               "haar_scores": [round(d["conf"], 2) for d in h_dets],
               "yolo_objects": None, "yolo_persons": None, "yolo_classes": {}, "yolo_avg_conf": None,
               "yolo_ms": None, "possible_issues": []}
        if model is not None:
            y_dets, y_ms = run_yolo(model, img, conf)
            y_img = draw(img, y_dets, (200, 80, 0))
            save_image(out_yolo, f"{name}_yolo.png", y_img)
            classes = {}
            for d in y_dets:
                classes[d["label"]] = classes.get(d["label"], 0) + 1
            persons = classes.get("person", 0)
            row.update({
                "yolo_objects": len(y_dets), "yolo_persons": persons, "yolo_classes": classes,
                "yolo_avg_conf": round(float(np.mean([d["conf"] for d in y_dets])), 3) if y_dets else None,
                "yolo_ms": round(y_ms, 1)})
            # Automatic hints only - YOU must look at the images and confirm.
            if persons > 0 and len(h_dets) == 0:
                row["possible_issues"].append("YOLO sees a person but Haar found no face (Haar may have missed it, "
                                              "or the face is small/turned away).")
            if len(h_dets) > 0 and persons == 0:
                row["possible_issues"].append("Haar found a face but YOLO found no person (possible Haar false "
                                              "positive, or YOLO miss).")
            low = [d for d in y_dets if d["conf"] < 0.5]
            if low:
                row["possible_issues"].append(
                    f"{len(low)} YOLO detection(s) have confidence < 0.50 (check if they are wrong): "
                    + ", ".join(f"{d['label']} {d['conf']:.2f}" for d in low))
            cmp_img = np.hstack([label(h_img, f"Haar cascade: {len(h_dets)} face(s)"),
                                 label(y_img, f"YOLO ({yolo_weights}): {len(y_dets)} object(s)")])
        else:
            cmp_img = label(h_img, f"Haar cascade: {len(h_dets)} face(s)  (YOLO not available)")
        save_image(out_cmp, f"{name}_comparison.png", cmp_img)
        rows.append(row)
        print(f"[Task 3] {name:<18} Haar faces={row['haar_faces']}  YOLO objects={row['yolo_objects']}  "
              f"classes={row['yolo_classes']}")
        for issue in row["possible_issues"]:
            print(f"           ! {issue}")

    save_metrics("task3", {
        "haar_file": haar_file.name, "yolo_weights": yolo_weights if model is not None else None,
        "yolo_available": model is not None, "yolo_conf_threshold": conf, "rows": rows,
        "note": "possible_issues are automatic hints, not ground truth. Verify visually and write your own "
                "observation about missed / incorrect detections in the report.",
    })
    print(f"[Task 3] Done. Outputs saved in: {out_haar.parent}")
    if model is None:
        print("[Task 3] WARNING: YOLO part NOT completed (see message above). Fix and re-run this task.")
    return model is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    a = ap.parse_args()
    run(a.yolo_weights, a.conf)


if __name__ == "__main__":
    main()
