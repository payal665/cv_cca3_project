"""Builds TECHNICAL_REPORT.md from the REAL numbers and images the tasks produced.

Nothing is invented: every table comes from outputs/metrics/*.json. Parts that only
YOU can write (your name, links, your own visual observations) are marked
with  >>> FILL IN <<<  so you do not forget them.

Run:  python src/generate_report.py      (main.py also runs it automatically)
"""
from common import OUTPUT_DIR, ROOT, load_metrics

FILL = ">>> FILL IN <<<"


def rel(*parts):
    """Path relative to the project root (used for image links) or None if missing."""
    p = OUTPUT_DIR.joinpath(*parts)
    return p.relative_to(ROOT).as_posix() if p.exists() else None


def img(path, caption):
    if path is None:
        return f"*(image not generated yet: {caption})*\n"
    return f"![{caption}]({path})\n\n*{caption}*\n"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def section_task1(m):
    s = ["## 4. Task 1 - Image Restoration and Segmentation (Unit 3)\n"]
    if not m:
        return s + ["*Task 1 has not been run yet.*\n"]
    T1 = "task1_restoration_segmentation"
    s.append("### 4.1 Image restoration\n")
    s.append(f"Gaussian noise (sigma = {m['noise']['gaussian_sigma']}) and salt-and-pepper noise "
             f"({int(m['noise']['salt_pepper_amount'] * 100)} % of pixels) were added to `{m['input_file']}`. "
             "Both noisy images were restored with a 5x5 median filter and a 5x5 Gaussian filter and compared with the "
             "clean original using PSNR (dB, higher is better) and SSIM (0 to 1, higher is better).\n")
    s.append(img(rel(T1, "A_restoration", "comparison_restoration.png"),
                 "Figure 1: Original, noisy and restored images with PSNR/SSIM"))
    s.append(table(["Noise", "Method", "PSNR (dB)", "SSIM"],
                   [[r["noise"], r["method"], f(r["psnr"]), f(r["ssim"], 4)] for r in m["restoration_table"]]))
    for noise, b in m["best_filter"].items():
        s.append(f"* **{noise}**: best PSNR -> {b['best_by_psnr']}; best SSIM -> {b['best_by_ssim']}.\n")
    s.append(f"\n**My observation:** {FILL} (look at Figure 1: which filter keeps edges sharper? which removes the "
             "white/black dots completely?)\n")
    mo = m["morphology"]
    s.append("### 4.2 Thresholding, morphology and Canny edges\n")
    s.append(f"Otsu's method chose threshold {mo['otsu_threshold']:.0f}. "
             f"{'The mask was inverted so objects are white. ' if mo['binary_was_inverted'] else ''}"
             f"Morphology used a {mo['kernel']} kernel.\n")
    s.append(img(rel(T1, "B_morphology_edges", "overview_morphology_edges.png"),
                 "Figure 2: Binary images, morphological operations and Canny edge maps"))
    s.append(table(["Result", "White pixels (%)"],
                   [["Otsu binary", f(mo["otsu_white_pct"])], ["Erosion", f(mo["erosion_white_pct"])],
                    ["Dilation", f(mo["dilation_white_pct"])], ["Opening", f(mo["opening_white_pct"])],
                    ["Closing", f(mo["closing_white_pct"])]]))
    lo, hi = mo["canny_auto_thresholds"]
    s.append(table(["Canny setting", "Edge pixels (%)"],
                   [[f"auto thresholds ({lo}, {hi})", f(mo["canny_auto_edge_pct"])],
                    ["fixed thresholds (100, 200)", f(mo["canny_fixed_edge_pct"])]]))
    s.append(f"\n**My observation:** {FILL} (erosion shrinks objects / removes small white noise; dilation grows them; "
             "opening removes small specks; closing fills small holes.)\n")
    s.append("### 4.3 Watershed segmentation\n")
    s.append(f"Image used: `{m['watershed']['image_used']}`. Number of regions found: "
             f"**{m['watershed']['num_regions_found']}**.\n")
    s.append(img(rel(T1, "B_watershed", "overview_watershed.png"), "Figure 3: Watershed segmentation"))
    s.append(f"**My observation:** {FILL} (did it separate touching objects correctly? any over/under-segmentation?)\n")
    s.append("### 4.4 Which techniques worked best?\n")
    s.append(f"{FILL} (write 3-4 lines using the tables above, e.g. best filter per noise type and why.)\n")
    return s


def section_task2(m):
    s = ["## 5. Task 2 - Feature Extraction and Image Matching (Unit 4)\n"]
    if not m:
        return s + ["*Task 2 has not been run yet.*\n"]
    T2 = "task2_feature_extraction_matching"
    h, sf = m["hog"], m["sift"]
    s.append("### 5.1 HOG features\n")
    s.append(f"HOG parameters: {h['params']}. Feature-vector length: photo1 = {h['photo1_vector_length']}, "
             f"photo2 = {h['photo2_vector_length']}.\n")
    s.append(img(rel(T2, "HOG", "hog_comparison.png"), "Figure 4: HOG visualisation of both photographs"))
    s.append("### 5.2 SIFT keypoints and descriptors\n")
    s.append(f"SIFT keypoints: photo1 = {sf['photo1_keypoints']}, photo2 = {sf['photo2_keypoints']}. "
             f"Each descriptor has {sf['descriptor_length']} values.\n")
    s.append(img(rel(T2, "SIFT_keypoints", "sift_keypoints_photo1.png"), "Figure 5: SIFT keypoints - photo1"))
    s.append(img(rel(T2, "SIFT_keypoints", "sift_keypoints_photo2.png"), "Figure 6: SIFT keypoints - photo2"))
    s.append("### 5.3 Matching and robustness test\n")
    s.append(f"Matching: brute-force L2, Lowe ratio test ({m['ratio_test']}), then RANSAC homography "
             f"(threshold {m['ransac_px']} px) to count geometrically consistent inliers.\n")
    n = 7
    for e in m["experiments"]:
        s.append(img(rel(T2, "matching", f"matches_{e['experiment']}.png"),
                     f"Figure {n}: {e['description']}"))
        n += 1
    s.append(table(["Experiment", "KP photo1", "KP photo2 variant", "Good matches", "RANSAC inliers",
                    "Inlier %", "Time (ms)"],
                   [[e["experiment"], e["keypoints_photo1"], e["keypoints_photo2_variant"], e["good_matches"],
                     e["ransac_inliers"], e["inlier_ratio_pct"], e["matching_time_ms"]] for e in m["experiments"]]))
    s.append(img(rel(T2, "matching", "comparison_chart.png"), f"Figure {n}: Matching comparison chart"))
    best = max(m["experiments"], key=lambda e: e["ransac_inliers"])
    worst = min(m["experiments"], key=lambda e: e["ransac_inliers"])
    s.append(f"From the table: most inliers in **{best['experiment']}** ({best['ransac_inliers']}), fewest in "
             f"**{worst['experiment']}** ({worst['ransac_inliers']}).\n")
    s.append(f"\n**My observation:** {FILL} (SIFT is designed to be scale/rotation invariant - do your numbers "
             "support that? why does matching drop for 25 % size?)\n")
    return s


def section_task3(m):
    s = ["## 6. Task 3 - Object Recognition: Viola-Jones vs YOLO (Unit 5)\n"]
    if not m:
        return s + ["*Task 3 has not been run yet.*\n"]
    T3 = "task3_object_recognition"
    s.append(f"Haar model: `{m['haar_file']}` (pretrained, OpenCV). YOLO model: "
             f"`{m['yolo_weights']}` (pretrained, confidence threshold {m['yolo_conf_threshold']}).\n"
             if m["yolo_available"] else
             f"Haar model: `{m['haar_file']}`. **YOLO was not available when this report was generated - re-run Task 3.**\n")
    for r in m["rows"]:
        s.append(img(rel(T3, "side_by_side_comparison", f"{r['source']}_comparison.png"),
                     f"Detection comparison on {r['source']} (left: Haar, right: YOLO)"))
    s.append(table(["Source", "Haar faces", "Haar ms", "YOLO objects", "YOLO classes", "Avg YOLO conf", "YOLO ms"],
                   [[r["source"], r["haar_faces"], r["haar_ms"], r["yolo_objects"], r["yolo_classes"],
                     r["yolo_avg_conf"], r["yolo_ms"]] for r in m["rows"]]))
    s.append("### Possible missed / incorrect detections (automatic hints - verify with your eyes)\n")
    issues = [(r["source"], i) for r in m["rows"] for i in r["possible_issues"]]
    if issues:
        for src, i in issues:
            s.append(f"* **{src}**: {i}\n")
    else:
        s.append("* No automatic hints were raised.\n")
    s.append(f"\n**My observation (required: at least one miss/wrong detection, if encountered):** {FILL}\n")
    s.append(f"\n**Comparison of the two approaches:** {FILL} (Haar: fast, only faces, no class names, many false "
             "positives; YOLO: many classes, confidence scores, more robust but heavier.)\n")
    return s


def section_task4(m):
    s = ["## 7. Task 4 - Video Motion Estimation (Unit 5)\n"]
    if not m:
        return s + ["*Task 4 has not been run yet.*\n"]
    T4 = "task4_motion_estimation"
    v = m["video"]
    s.append(f"Video `{v['file']}`: {v['width']}x{v['height']}, {v['fps']:.1f} fps, {v['frames']} frames, "
             f"{v['duration_s']:.1f} s. Frames were resized to width {m['processed_width']} px for processing.\n")
    s.append("### 7.1 Consecutive frames\n")
    for i in m["consecutive_frames_saved"]:
        p = rel(T4, "consecutive_frames", f"frame_{i:04d}.png")
        if p:
            s.append(img(p, f"Consecutive frame {i}"))
    s.append("### 7.2 Masks and motion vectors (original | frame difference / GMM mask | Lucas-Kanade vectors)\n")
    for i in m["sample_frames"]:
        s.append(img(rel(T4, "comparison_panels", f"panel_frame_{i:04d}.png"),
                     f"Frame {i}: original, frame-differencing mask, GMM foreground mask, LK vectors (arrows x3)"))
    s.append(img(rel(T4, "motion_over_time.png"), "Detected motion over time"))
    s.append("### 7.3 Comparison table (measured on your video)\n")
    s.append(table(["Technique", "Output", "Avg moving value", "Avg time (ms/frame)", "Notes"],
                   [[r["technique"], r["output"], f(r["avg_moving_area_pct"]), f(r["avg_time_ms_per_frame"]),
                     r["extra"]] for r in m["summary"]]))
    if m.get("avg_iou_framediff_vs_gmm") is not None:
        s.append(f"Average overlap (IoU) between the frame-differencing mask and the GMM mask: "
                 f"**{m['avg_iou_framediff_vs_gmm']:.2f}** (1.0 = identical masks).\n")
    s.append(f"\n**My observation:** {FILL} (look at the panels: does frame differencing only show edges of the "
             "moving object? does GMM fill the whole object? do the LK arrows point in the direction of motion?)\n")
    return s


def build_report():
    m1, m2, m3, m4 = (load_metrics(n) for n in ("task1", "task2", "task3", "task4"))
    s = ["# CCA 3 - Computer Vision Practical Assignment (Units 3, 4 and 5)\n",
         f"**Student:** {FILL}  \n**Roll no / Division / Batch:** {FILL}  \n**Course:** MCA, MIT-WPU\n",
         f"**Git repository URL:** {FILL}  \n**Project explanation video (Google Drive link):** {FILL}\n",
         "## 1. Aim\n",
         "To implement image restoration, segmentation, feature extraction, object recognition and motion estimation "
         "techniques in Python (OpenCV, NumPy, Matplotlib, scikit-image, Ultralytics YOLO) on my own photographs "
         "and self-recorded video.\n",
         "## 2. Input Data\n"]
    rows = []
    if m1:
        rows.append(["photo1 (original photograph)", m1["input_file"],
                     f"working size {m1['working_size_wh'][0]}x{m1['working_size_wh'][1]}"])
    if m2:
        rows.append(["photo2 (second photograph)", m2["photo2"],
                     f"working size {m2['working_size_photo2_wh'][0]}x{m2['working_size_photo2_wh'][1]}"])
    if m4:
        v = m4["video"]
        rows.append(["video (original)", v["file"], f"{v['width']}x{v['height']}, {v['fps']:.1f} fps, "
                                                     f"{v['duration_s']:.1f} s"])
    s.append(table(["Input", "File", "Details"], rows) if rows else "*No tasks run yet.*\n")
    s.append("Original, unedited files are stored in the `input/` folder (video also on Google Drive). "
             "Large photos are shrunk (longest side <= 1000 px) only for processing speed.\n")
    s.append(f"**Description of the photographs and video (where/what I captured):** {FILL}\n")
    s.append("## 3. Source Code\n")
    s.append(table(["Task", "File"], [
        ["Task 1 - restoration, morphology, Canny, watershed", "`src/task1_restoration_segmentation.py`"],
        ["Task 2 - HOG, SIFT, matching", "`src/task2_feature_matching.py`"],
        ["Task 3 - Haar cascade and YOLO", "`src/task3_object_detection.py`"],
        ["Task 4 - frame differencing, GMM, Lucas-Kanade", "`src/task4_motion_estimation.py`"],
        ["Runner / report", "`main.py`, `src/generate_report.py`"]]))
    s += section_task1(m1) + section_task2(m2) + section_task3(m3) + section_task4(m4)
    s += ["## 8. Conclusion\n",
          f"{FILL} (5-8 lines: what you implemented, the best filter, how SIFT behaved under rotation/resizing, "
          "Haar vs YOLO, and which motion technique suited your video, quoting numbers from the tables above.)\n",
          "## 9. Submission checklist\n",
          "- [ ] Original photo1, photo2 and video submitted\n- [ ] Git repository pushed and URL written above\n"
          "- [ ] Repository access given to rahul.inchal.inchal24@gmail.com and rahul.inchal.vf@mitwpu.edu.in\n"
          "- [ ] Explanation video (5-8 min) uploaded to the Google Drive folder and link written above\n"
          "- [ ] All `>>> FILL IN <<<` markers in this report replaced\n"]
    path = ROOT / "TECHNICAL_REPORT.md"
    path.write_text("\n".join(s), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build_report())
