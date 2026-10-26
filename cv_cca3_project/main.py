"""Main runner for the CCA 3 Computer Vision project.

Usage (VS Code terminal, from the project folder):
    python main.py                 # check inputs, run all 4 tasks, build the report
    python main.py --tasks 1 2     # run only some tasks
    python main.py --check         # only check which input files are present
    python main.py --seg-image input/coins.jpg   # different image for watershed
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import common  # noqa: E402

REQUIRED = {
    1: ["photo1"],
    2: ["photo1", "photo2"],
    3: ["photo1"],          # photo2 and video are used too if present
    4: ["video"],
}
TASK_TITLES = {1: "Image restoration & segmentation", 2: "Feature extraction & matching",
               3: "Object recognition (Haar + YOLO)", 4: "Video motion estimation"}


def input_status():
    photos, video = common.IMAGE_EXTS, common.VIDEO_EXTS
    found = {"photo1": common.find_input("photo1", photos),
             "photo2": common.find_input("photo2", photos),
             "video": common.find_input("video", video)}
    print("\nInput check (folder: %s)" % common.INPUT_DIR)
    for key, path in found.items():
        print(f"   [{'OK' if path else 'MISSING'}] {key:<7} {path.name if path else '-> please add input/' + key + '.jpg/.mp4'}")
    print()
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    ap.add_argument("--check", action="store_true", help="only check input files")
    ap.add_argument("--seg-image", default=None, help="optional image for watershed in Task 1")
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--no-videos", action="store_true", help="Task 4: skip writing result videos")
    args = ap.parse_args()

    found = input_status()
    if args.check:
        return 0

    status = {}
    for t in args.tasks:
        missing = [k for k in REQUIRED[t] if not found[k]]
        if missing:
            status[t] = f"SKIPPED (missing input: {', '.join(missing)})"
            print(f"=== Task {t}: {TASK_TITLES[t]} -> {status[t]}\n")
            continue
        print(f"=== Task {t}: {TASK_TITLES[t]} ===")
        t0 = time.time()
        try:
            complete = True
            if t == 1:
                from task1_restoration_segmentation import run
                run(seg_image=args.seg_image)
            elif t == 2:
                from task2_feature_matching import run
                run()
            elif t == 3:
                from task3_object_detection import run
                complete = run(yolo_weights=args.yolo_weights)
            elif t == 4:
                from task4_motion_estimation import run
                run(save_videos=not args.no_videos)
            if complete:
                status[t] = f"finished in {time.time() - t0:.1f} s"
            else:
                status[t] = "PARTIAL - YOLO part not done (install ultralytics / check internet), Haar part done"
        except Exception as e:  # keep going with the other tasks
            traceback.print_exc()
            status[t] = f"FAILED: {e}"
        print()

    try:
        from generate_report import build_report
        path = build_report()
        print(f"Technical report draft written to: {path}\n")
    except Exception as e:
        print(f"Report generation failed: {e}\n")

    print("Summary")
    for t, s in status.items():
        print(f"   Task {t} ({TASK_TITLES[t]}): {s}")
    bad = [t for t, s in status.items() if not s.startswith("finished")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
