"""Task 4 - Video Motion Estimation (Unit 5).

Uses input/video.mp4 and compares three techniques frame by frame:
  1. Frame differencing        (moving regions from |frame_t - frame_t-1|)
  2. GMM background subtraction (OpenCV MOG2)
  3. Lucas-Kanade optical flow  (motion vectors of tracked corner points)

Outputs: consecutive frames, foreground masks, motion-vector images, 2x2 panels,
optional result videos, a motion-over-time plot and a comparison table.

Run:  python src/task4_motion_estimation.py
"""
import argparse
import csv
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import (VIDEO_EXTS, label, require_input, save_figure, save_image,
                    save_metrics, task_dir)

TASK_NAME = "task4_motion_estimation"
PROC_WIDTH = 640     # frames are shrunk to this width for speed
WARMUP = 30          # GMM needs time to learn the background; skip these frames in averages
FD_THRESH = 25       # frame-difference threshold
MOVE_PX = 1.0        # an optical-flow point counts as "moving" if it moves > 1 pixel
VEC_SCALE = 3        # arrows are drawn 3x longer so they are visible


def resize_width(frame, width=PROC_WIDTH):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def lucas_kanade(prev_gray, gray):
    """Track Shi-Tomasi corners from prev_gray to gray. Returns old and new points."""
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                 minDistance=10, blockSize=7)
    if p0 is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    ok = st.reshape(-1) == 1
    return p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]


def draw_vectors(frame, old, new):
    out = frame.copy()
    for (x0, y0), (x1, y1) in zip(old, new):
        dx, dy = x1 - x0, y1 - y0
        if np.hypot(dx, dy) > MOVE_PX:
            cv2.arrowedLine(out, (int(x0), int(y0)),
                            (int(x0 + VEC_SCALE * dx), int(y0 + VEC_SCALE * dy)),
                            (0, 0, 255), 2, tipLength=0.3)
            cv2.circle(out, (int(x0), int(y0)), 3, (0, 255, 0), -1)
        else:
            cv2.circle(out, (int(x0), int(y0)), 2, (255, 200, 0), -1)
    return out


def run(save_videos=True):
    video = require_input("video", VIDEO_EXTS, "your original 15-30 second video")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open the video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        total = count_frames(video)
        cap = cv2.VideoCapture(str(video))
    raw_w, raw_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps
    print(f"[Task 4] {video.name}: {raw_w}x{raw_h}, {fps:.1f} fps, {total} frames, {duration:.1f} s")
    if not 14.5 <= duration <= 31:
        print("[Task 4] WARNING: the assignment asks for a 15-30 second video; yours is "
              f"{duration:.1f} s. Consider re-recording.")
    if total < 10:
        raise ValueError("The video has too few frames.")

    out_frames = task_dir(TASK_NAME, "consecutive_frames")
    out_fd = task_dir(TASK_NAME, "frame_differencing")
    out_gmm = task_dir(TASK_NAME, "gmm_background_subtraction")
    out_lk = task_dir(TASK_NAME, "lucas_kanade_optical_flow")
    out_panel = task_dir(TASK_NAME, "comparison_panels")
    out_vid = task_dir(TASK_NAME, "result_videos")

    sample_idx = {max(1, int(total * f)) for f in (0.25, 0.5, 0.75)}
    mid = total // 2
    consecutive_idx = set(range(mid, min(mid + 6, total)))   # 6 consecutive frames

    mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    writers = {}
    stats = []
    prev_gray = prev_blur = None
    idx = 0
    saved_samples = []
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame = resize_width(raw)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ---- 2. GMM background subtraction (runs on every frame so it can learn)
        t0 = time.perf_counter()
        fg = mog.apply(frame)
        _, gmm_mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)   # drop grey "shadow" pixels
        gmm_mask = cv2.morphologyEx(gmm_mask, cv2.MORPH_OPEN, k3)
        gmm_mask = cv2.morphologyEx(gmm_mask, cv2.MORPH_CLOSE, k5)
        t_gmm = (time.perf_counter() - t0) * 1000

        if idx in consecutive_idx:
            save_image(out_frames, f"frame_{idx:04d}.png", frame)

        if prev_gray is None:
            fd_mask = np.zeros_like(gray)
            vec_img, old, new, t_fd, t_lk = frame.copy(), np.empty((0, 2)), np.empty((0, 2)), 0.0, 0.0
        else:
            # ---- 1. frame differencing
            t0 = time.perf_counter()
            diff = cv2.absdiff(blur, prev_blur)
            _, fd_mask = cv2.threshold(diff, FD_THRESH, 255, cv2.THRESH_BINARY)
            fd_mask = cv2.dilate(fd_mask, k5, iterations=2)
            t_fd = (time.perf_counter() - t0) * 1000
            # ---- 3. Lucas-Kanade optical flow
            t0 = time.perf_counter()
            old, new = lucas_kanade(prev_gray, gray)
            t_lk = (time.perf_counter() - t0) * 1000
            vec_img = draw_vectors(frame, old, new)

        disp = np.hypot(*(new - old).T) if len(old) else np.array([])
        inter = np.count_nonzero((fd_mask > 0) & (gmm_mask > 0))
        union = np.count_nonzero((fd_mask > 0) | (gmm_mask > 0))
        stats.append({
            "frame": idx,
            "fd_pct": 100 * np.count_nonzero(fd_mask) / fd_mask.size,
            "gmm_pct": 100 * np.count_nonzero(gmm_mask) / gmm_mask.size,
            "lk_points": int(len(old)),
            "lk_mean_disp": float(disp.mean()) if len(disp) else 0.0,
            "lk_moving_pct": 100 * float((disp > MOVE_PX).mean()) if len(disp) else 0.0,
            "iou_fd_gmm": inter / union if union else None,
            "t_fd": t_fd, "t_gmm": t_gmm, "t_lk": t_lk,
        })

        if idx in sample_idx:
            tiles = [label(frame, f"Original frame {idx}"),
                     label(fd_mask, "Frame differencing mask"),
                     label(gmm_mask, "GMM (MOG2) foreground mask"),
                     label(vec_img, f"Lucas-Kanade vectors (x{VEC_SCALE})")]
            save_image(out_fd, f"fd_mask_{idx:04d}.png", fd_mask)
            save_image(out_gmm, f"gmm_mask_{idx:04d}.png", gmm_mask)
            save_image(out_lk, f"lk_vectors_{idx:04d}.png", vec_img)
            panel = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            save_image(out_panel, f"panel_frame_{idx:04d}.png", panel)
            saved_samples.append(idx)

        if save_videos:
            for key, img in (("frame_differencing", cv2.cvtColor(fd_mask, cv2.COLOR_GRAY2BGR)),
                             ("gmm_foreground", cv2.cvtColor(gmm_mask, cv2.COLOR_GRAY2BGR)),
                             ("lucas_kanade", vec_img)):
                if key not in writers:
                    writers[key] = cv2.VideoWriter(str(out_vid / f"{key}.mp4"),
                                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writers[key].write(img)

        prev_gray, prev_blur = gray, blur
        idx += 1
        if idx % 100 == 0:
            print(f"[Task 4]   processed {idx}/{total} frames")
    cap.release()
    for wr in writers.values():
        wr.release()

    n = len(stats)
    if n < 5:
        raise RuntimeError("Could not read enough frames from the video.")
    start = min(WARMUP, max(1, n // 4))
    ev = stats[start:]
    mean = lambda key: float(np.mean([s[key] for s in ev]))
    ious = [s["iou_fd_gmm"] for s in ev if s["iou_fd_gmm"] is not None]
    summary = [
        {"technique": "Frame differencing", "output": "binary motion mask",
         "avg_moving_area_pct": mean("fd_pct"), "avg_time_ms_per_frame": mean("t_fd"),
         "extra": "needs previous frame only"},
        {"technique": "GMM background subtraction (MOG2)", "output": "binary foreground mask",
         "avg_moving_area_pct": mean("gmm_pct"), "avg_time_ms_per_frame": mean("t_gmm"),
         "extra": f"first ~{start} frames skipped (background learning)"},
        {"technique": "Lucas-Kanade optical flow", "output": "motion vectors at corner points",
         "avg_moving_area_pct": mean("lk_moving_pct"), "avg_time_ms_per_frame": mean("t_lk"),
         "extra": f"avg tracked points {mean('lk_points'):.0f}, avg displacement {mean('lk_mean_disp'):.2f}px; "
                  "value = % of points moving > 1px (not area)"},
    ]
    with open(out_panel.parent / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        wr.writerows(summary)

    # motion-over-time plot
    xs = [s["frame"] for s in stats]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    a1.plot(xs, [s["fd_pct"] for s in stats], label="Frame differencing (% pixels)")
    a1.plot(xs, [s["gmm_pct"] for s in stats], label="GMM foreground (% pixels)")
    a1.axvline(start, color="gray", ls="--", lw=1)
    a1.set_ylabel("moving area (%)")
    a1.legend()
    a1.set_title("Amount of detected motion per frame")
    a2.plot(xs, [s["lk_mean_disp"] for s in stats], color="tab:red", label="Lucas-Kanade mean displacement (px)")
    a2.set_ylabel("pixels")
    a2.set_xlabel("frame number")
    a2.legend()
    save_figure(fig, out_panel.parent / "motion_over_time.png")

    save_metrics("task4", {
        "video": {"file": video.name, "width": raw_w, "height": raw_h, "fps": fps,
                  "frames": n, "duration_s": n / fps},
        "processed_width": PROC_WIDTH, "warmup_frames_skipped": start,
        "thresholds": {"frame_diff": FD_THRESH, "lk_moving_px": MOVE_PX},
        "summary": summary,
        "avg_iou_framediff_vs_gmm": float(np.mean(ious)) if ious else None,
        "sample_frames": sorted(saved_samples),
        "consecutive_frames_saved": sorted(consecutive_idx),
    })
    print("[Task 4] Comparison (averages after warm-up):")
    for s in summary:
        print(f"   {s['technique']:<36} moving={s['avg_moving_area_pct']:6.2f}   {s['avg_time_ms_per_frame']:6.2f} ms/frame")
    print(f"[Task 4] Done. Outputs saved in: {out_panel.parent}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-videos", action="store_true", help="do not write result videos (faster)")
    a = ap.parse_args()
    run(save_videos=not a.no_videos)


if __name__ == "__main__":
    main()
