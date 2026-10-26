"""Task 1 - Image Restoration, Morphology and Segmentation (Unit 3).

What this file does
-------------------
A. Adds Gaussian noise and salt-and-pepper noise to your photo, restores it with
   median and Gaussian filters, and measures PSNR and SSIM against the clean photo.
B. Thresholding (binary image), erosion, dilation, opening, closing,
   Canny edge detection and watershed segmentation.

Run in VS Code terminal:
    python src/task1_restoration_segmentation.py
Optional (use a different image, e.g. one with touching coins, for watershed):
    python src/task1_restoration_segmentation.py --seg-image input/coins.jpg
"""
import argparse
import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common import (IMAGE_EXTS, bgr2rgb, load_image, require_input, save_figure,
                    save_image, save_metrics, task_dir)

TASK_NAME = "task1_restoration_segmentation"
SEED = 42            # fixed seed -> same noise every run (reproducible results)
GAUSSIAN_SIGMA = 25  # strength of Gaussian noise (0-255 scale)
SP_AMOUNT = 0.05     # 5 % of pixels become salt (white) or pepper (black)


# ---------------------------------------------------------------- Part A: noise
def add_gaussian_noise(img, sigma=GAUSSIAN_SIGMA):
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=SP_AMOUNT):
    rng = np.random.default_rng(SEED)
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < amount / 2] = 0            # pepper (black)
    out[r > 1 - amount / 2] = 255      # salt (white)
    return out


FILTERS = {
    "median": lambda im: cv2.medianBlur(im, 5),
    "gaussian": lambda im: cv2.GaussianBlur(im, (5, 5), 1.0),
}


def measure(reference, other):
    """PSNR (dB, higher = better) and SSIM (0..1, higher = better) vs the clean image."""
    p = peak_signal_noise_ratio(reference, other, data_range=255)
    s = structural_similarity(reference, other, channel_axis=2, data_range=255)
    return float(p), float(s)


def part_a_restoration(original, out_dir):
    noisy_images = {
        "gaussian_noise": add_gaussian_noise(original),
        "salt_pepper_noise": add_salt_pepper_noise(original),
    }
    table = []
    restored_images = {}
    for noise_name, noisy in noisy_images.items():
        save_image(out_dir, f"noisy_{noise_name}.png", noisy)
        p, s = measure(original, noisy)
        table.append({"noise": noise_name, "method": "none (noisy image)", "psnr": p, "ssim": s})
        for filt_name, fn in FILTERS.items():
            restored = fn(noisy)
            restored_images[(noise_name, filt_name)] = restored
            save_image(out_dir, f"restored_{noise_name}_{filt_name}.png", restored)
            p, s = measure(original, restored)
            table.append({"noise": noise_name, "method": f"{filt_name} filter 5x5", "psnr": p, "ssim": s})

    # comparison figure (2 rows x 4 columns)
    lookup = {(r["noise"], r["method"]): r for r in table}
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for row, (noise_name, noisy) in enumerate(noisy_images.items()):
        panels = [
            ("Original (clean)", original, None),
            ("Noisy", noisy, lookup[(noise_name, "none (noisy image)")]),
            ("Median 5x5", restored_images[(noise_name, "median")], lookup[(noise_name, "median filter 5x5")]),
            ("Gaussian 5x5", restored_images[(noise_name, "gaussian")], lookup[(noise_name, "gaussian filter 5x5")]),
        ]
        for col, (title, im, res) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(bgr2rgb(im))
            if res:
                title += f"\nPSNR {res['psnr']:.2f} dB | SSIM {res['ssim']:.3f}"
            ax.set_title(f"{noise_name.replace('_', ' ')}: {title}" if col == 0 else title, fontsize=10)
            ax.axis("off")
    fig.suptitle("Image restoration: original vs noisy vs restored", fontsize=14)
    save_figure(fig, out_dir / "comparison_restoration.png")

    # which filter is best for each noise type?
    best = {}
    for noise_name in noisy_images:
        rows = [r for r in table if r["noise"] == noise_name and not r["method"].startswith("none")]
        best[noise_name] = {
            "best_by_psnr": max(rows, key=lambda r: r["psnr"])["method"],
            "best_by_ssim": max(rows, key=lambda r: r["ssim"])["method"],
        }

    with open(out_dir / "psnr_ssim_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["noise", "method", "psnr", "ssim"])
        w.writeheader()
        w.writerows(table)
    return table, best


# ------------------------------------------------- Part B: morphology + edges
def white_minority(mask):
    """Make sure the 'object' pixels are white. If more than half of the mask
    is white we assume the background is white and invert it."""
    if np.count_nonzero(mask) > mask.size / 2:
        return cv2.bitwise_not(mask), True
    return mask, False


def part_b_morphology(img, out_dir):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5. thresholding -> binary images
    otsu_value, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu, inverted = white_minority(otsu)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 5)
    save_image(out_dir, "gray.png", gray)
    save_image(out_dir, "binary_otsu.png", otsu)
    save_image(out_dir, "binary_adaptive.png", adaptive)

    # 6. erosion, dilation, opening, closing (on the Otsu binary mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erosion = cv2.erode(otsu, kernel, iterations=1)
    dilation = cv2.dilate(otsu, kernel, iterations=1)
    opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    for name, im in [("erosion", erosion), ("dilation", dilation),
                     ("opening", opening), ("closing", closing)]:
        save_image(out_dir, f"morph_{name}.png", im)

    # 7. Canny edge detection (automatic thresholds from the median brightness,
    #    compared with fixed 100/200 thresholds)
    med = float(np.median(blur))
    low, high = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    canny_auto = cv2.Canny(blur, low, high)
    canny_fixed = cv2.Canny(blur, 100, 200)
    save_image(out_dir, "canny_auto.png", canny_auto)
    save_image(out_dir, "canny_fixed_100_200.png", canny_fixed)

    # overview figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    panels = [("Grayscale", gray), ("Otsu binary", otsu), ("Adaptive binary", adaptive),
              (f"Canny auto ({low},{high})", canny_auto), ("Canny fixed (100,200)", canny_fixed),
              ("Erosion", erosion), ("Dilation", dilation), ("Opening", opening),
              ("Closing", closing), ("Original", None)]
    for ax, (title, im) in zip(axes.ravel(), panels):
        if im is None:
            ax.imshow(bgr2rgb(img))
        else:
            ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Thresholding, morphology and Canny edges", fontsize=14)
    save_figure(fig, out_dir / "overview_morphology_edges.png")

    total = float(otsu.size)
    return {
        "otsu_threshold": float(otsu_value),
        "binary_was_inverted": bool(inverted),
        "otsu_white_pct": 100 * np.count_nonzero(otsu) / total,
        "erosion_white_pct": 100 * np.count_nonzero(erosion) / total,
        "dilation_white_pct": 100 * np.count_nonzero(dilation) / total,
        "opening_white_pct": 100 * np.count_nonzero(opening) / total,
        "closing_white_pct": 100 * np.count_nonzero(closing) / total,
        "canny_auto_thresholds": [low, high],
        "canny_auto_edge_pct": 100 * np.count_nonzero(canny_auto) / total,
        "canny_fixed_edge_pct": 100 * np.count_nonzero(canny_fixed) / total,
        "kernel": "5x5 ellipse",
    }


# ------------------------------------------------------------------ watershed
def part_b_watershed(img, out_dir):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary, _ = white_minority(binary)

    k = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
    sure_bg = cv2.dilate(opening, k, iterations=3)              # surely background
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)       # distance to background
    if dist.max() <= 0:
        raise RuntimeError("Watershed: no objects found after thresholding. "
                           "Use an image with clear objects (see --seg-image).")
    _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, 0)  # surely object centres
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1            # background = 1, objects = 2, 3, ...
    markers[unknown == 255] = 0      # 0 = "not decided yet", watershed decides
    markers = cv2.watershed(img.copy(), markers)

    boundaries = img.copy()
    boundaries[markers == -1] = (0, 0, 255)

    rng = np.random.default_rng(1)
    colors = rng.integers(60, 255, size=(markers.max() + 2, 3)).astype(np.uint8)
    color_layer = np.zeros_like(img)
    labels = [l for l in np.unique(markers) if l > 1]
    for lab in labels:
        color_layer[markers == lab] = colors[lab]
    regions = cv2.addWeighted(img, 0.5, color_layer, 0.5, 0)
    regions[markers == -1] = (0, 0, 255)

    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_image(out_dir, "watershed_binary.png", binary)
    save_image(out_dir, "watershed_distance_transform.png", dist_vis)
    save_image(out_dir, "watershed_sure_foreground.png", sure_fg)
    save_image(out_dir, "watershed_boundaries.png", boundaries)
    save_image(out_dir, "watershed_regions.png", regions)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (title, im, cmap) in zip(axes, [
            ("Original", bgr2rgb(img), None), ("Distance transform", dist_vis, "gray"),
            ("Boundaries (red)", bgr2rgb(boundaries), None),
            (f"Segmented regions ({len(labels)})", bgr2rgb(regions), None)]):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Watershed segmentation", fontsize=14)
    save_figure(fig, out_dir / "overview_watershed.png")
    return {"num_regions_found": len(labels)}


# ----------------------------------------------------------------------- main
def run(seg_image=None):
    photo_path = require_input("photo1", IMAGE_EXTS, "your original photograph")
    original = load_image(photo_path)
    print(f"[Task 1] Using {photo_path.name}, working size {original.shape[1]}x{original.shape[0]}")

    out_a = task_dir(TASK_NAME, "A_restoration")
    out_b = task_dir(TASK_NAME, "B_morphology_edges")
    out_w = task_dir(TASK_NAME, "B_watershed")

    table, best = part_a_restoration(original, out_a)
    print("[Task 1] PSNR / SSIM:")
    for r in table:
        print(f"   {r['noise']:<18} {r['method']:<20} PSNR {r['psnr']:6.2f} dB   SSIM {r['ssim']:.4f}")

    morph = part_b_morphology(original, out_b)

    if seg_image:
        seg_img = load_image(seg_image)
        seg_name = str(seg_image)
    else:
        seg_img, seg_name = original, photo_path.name
    ws = part_b_watershed(seg_img, out_w)
    print(f"[Task 1] Watershed found {ws['num_regions_found']} regions in {seg_name}")

    save_metrics("task1", {
        "input_file": photo_path.name,
        "working_size_wh": [original.shape[1], original.shape[0]],
        "noise": {"gaussian_sigma": GAUSSIAN_SIGMA, "salt_pepper_amount": SP_AMOUNT, "seed": SEED},
        "restoration_table": table,
        "best_filter": best,
        "morphology": morph,
        "watershed": {"image_used": seg_name, **ws},
    })
    print(f"[Task 1] Done. Outputs saved in: {out_a.parent}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-image", default=None, help="optional image for watershed (default: photo1)")
    args = ap.parse_args()
    run(seg_image=args.seg_image)


if __name__ == "__main__":
    main()
