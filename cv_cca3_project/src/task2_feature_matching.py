"""Task 2 - Feature Extraction and Image Matching (Unit 4).

1. HOG features + visualisation for both photographs.
2. SIFT keypoints and descriptors.
3. SIFT matching between photo1 and photo2 (Lowe ratio test + RANSAC check).
4. Repeat the matching after rotating / resizing photo2.
5. Compare all matching experiments (table + chart).

Run:  python src/task2_feature_matching.py
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.feature import hog

from common import (IMAGE_EXTS, bgr2rgb, load_image, require_input, save_figure,
                    save_image, save_metrics, task_dir)

TASK_NAME = "task2_feature_extraction_matching"
RATIO = 0.75           # Lowe ratio test threshold
RANSAC_PX = 5.0        # reprojection error allowed for an "inlier"
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))


# ------------------------------------------------------------------------ HOG
def hog_features(img, out_dir, tag):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (int(w * 256 / h), 256))          # same height for both photos
    vec, hog_img = hog(gray, visualize=True, block_norm="L2-Hys", **HOG_PARAMS)
    hog_img = exposure.rescale_intensity(hog_img, out_range=(0, 255)).astype(np.uint8)
    save_image(out_dir, f"hog_{tag}.png", hog_img)
    return gray, hog_img, int(vec.shape[0])


# ----------------------------------------------------------------------- SIFT
def sift_features(img, out_dir, tag):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    vis = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save_image(out_dir, f"sift_keypoints_{tag}.png", vis)
    return kp, des, vis


def match_sift(img1, kp1, des1, img2, kp2, des2):
    """Return (good matches, inlier count, match picture, time in ms)."""
    t0 = time.perf_counter()
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], 0, None, 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(des1, des2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    inliers, mask = 0, None
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if mask is not None:
            inliers = int(mask.sum())
    ms = (time.perf_counter() - t0) * 1000
    good_sorted = sorted(good, key=lambda m: m.distance)[:80]   # draw the 80 best
    pic = cv2.drawMatches(img1, kp1, img2, kp2, good_sorted, None,
                          matchColor=(0, 255, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return good, inliers, pic, ms


def rotate_keep_all(img, angle):
    """Rotate by 'angle' degrees without cutting the corners off."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh))


def run():
    p1 = require_input("photo1", IMAGE_EXTS, "your original photograph")
    p2 = require_input("photo2", IMAGE_EXTS, "second photograph of the same object/scene")
    img1 = load_image(p1, 800)
    img2 = load_image(p2, 800)
    print(f"[Task 2] photo1 {img1.shape[1]}x{img1.shape[0]}, photo2 {img2.shape[1]}x{img2.shape[0]}")

    out_hog = task_dir(TASK_NAME, "HOG")
    out_sift = task_dir(TASK_NAME, "SIFT_keypoints")
    out_match = task_dir(TASK_NAME, "matching")

    # ---- HOG
    g1, h1, len1 = hog_features(img1, out_hog, "photo1")
    g2, h2, len2 = hog_features(img2, out_hog, "photo2")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, im, title, cmap in [(axes[0, 0], g1, "photo1 (gray)", "gray"), (axes[0, 1], h1, "photo1 HOG", "gray"),
                                (axes[1, 0], g2, "photo2 (gray)", "gray"), (axes[1, 1], h2, "photo2 HOG", "gray")]:
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Histogram of Oriented Gradients (9 orientations, 16x16 cells, 2x2 blocks)")
    save_figure(fig, out_hog / "hog_comparison.png")

    # ---- SIFT keypoints
    kp1, des1, vis1 = sift_features(img1, out_sift, "photo1")
    kp2, des2, vis2 = sift_features(img2, out_sift, "photo2")
    print(f"[Task 2] SIFT keypoints: photo1={len(kp1)}, photo2={len(kp2)}")

    # ---- Matching experiments (transform photo2, keep photo1 fixed)
    experiments = [
        ("original", "photo1 vs photo2 (no change)", img2),
        ("rotated_45", "photo2 rotated by 45 degrees", rotate_keep_all(img2, 45)),
        ("rotated_90", "photo2 rotated by 90 degrees", rotate_keep_all(img2, 90)),
        ("resized_50pct", "photo2 resized to 50 %",
         cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)),
        ("resized_25pct", "photo2 resized to 25 %",
         cv2.resize(img2, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)),
    ]
    results = []
    for key, desc, img2_t in experiments:
        if key == "original":
            k2, d2 = kp2, des2
        else:
            gray_t = cv2.cvtColor(img2_t, cv2.COLOR_BGR2GRAY)
            k2, d2 = cv2.SIFT_create().detectAndCompute(gray_t, None)
        good, inliers, pic, ms = match_sift(img1, kp1, des1, img2_t, k2, d2)
        if pic is not None:
            save_image(out_match, f"matches_{key}.png", pic)
        row = {
            "experiment": key, "description": desc,
            "keypoints_photo1": len(kp1), "keypoints_photo2_variant": len(k2),
            "good_matches": len(good), "ransac_inliers": inliers,
            "inlier_ratio_pct": round(100 * inliers / len(good), 1) if good else 0.0,
            "matching_time_ms": round(ms, 1),
        }
        results.append(row)
        print(f"[Task 2] {key:<14} good={len(good):4d}  inliers={inliers:4d}  ({ms:.0f} ms)")

    # ---- comparison chart
    names = [r["experiment"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, [r["good_matches"] for r in results], 0.4, label="good matches (ratio test)")
    ax.bar(x + 0.2, [r["ransac_inliers"] for r in results], 0.4, label="RANSAC inliers")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("number of matches")
    ax.set_title("SIFT matching quality: original vs rotated vs resized")
    ax.legend()
    save_figure(fig, out_match / "comparison_chart.png")

    save_metrics("task2", {
        "photo1": p1.name, "photo2": p2.name,
        "working_size_photo1_wh": [img1.shape[1], img1.shape[0]],
        "working_size_photo2_wh": [img2.shape[1], img2.shape[0]],
        "hog": {"params": {"orientations": 9, "pixels_per_cell": [16, 16], "cells_per_block": [2, 2]},
                "photo1_vector_length": len1, "photo2_vector_length": len2},
        "sift": {"photo1_keypoints": len(kp1), "photo2_keypoints": len(kp2),
                 "descriptor_length": int(des1.shape[1]) if des1 is not None else 0},
        "ratio_test": RATIO, "ransac_px": RANSAC_PX,
        "experiments": results,
    })
    print(f"[Task 2] Done. Outputs saved in: {out_hog.parent}")


def main():
    run()


if __name__ == "__main__":
    main()
