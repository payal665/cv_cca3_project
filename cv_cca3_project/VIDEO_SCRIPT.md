# Project Explanation Video - Script (target 6-7 minutes)

The assignment says: explain **in your own words**, with a face/voice - not only slides or a silent screen
recording. Use this as bullet points to speak from, not to read word-for-word. Replace every
`[...]` with what you actually see in YOUR outputs. Record with any screen recorder
(Windows: `Win+G` Xbox Game Bar, or OBS; Zoom / Google Meet recording also works) and keep your
voice on.

Before recording: run `python main.py` and open `TECHNICAL_REPORT.md` preview + the `outputs/` folder.

| Time | What to show on screen | What to say (own words) |
|---|---|---|
| 0:00-0:30 | Title / report first page | Your name, course, "CCA 3 - Computer Vision, Units 3-5". Aim: restoration, segmentation, features, detection, motion estimation on my own photo and video. |
| 0:30-1:15 | `input/` folder: photo1, photo2, video | Describe what you captured: where, what object/scene, why you chose it, video length and who/what moves. Mention the files are original and unedited. |
| 1:15-1:45 | VS Code project tree, `main.py`, `requirements.txt` | Explain the structure: one file per task, `main.py` runs all, outputs saved in `outputs/`, code pushed to GitHub (show the repo page). |
| 1:45-3:00 | Task 1 code (noise + filters) then `comparison_restoration.png`, `psnr_ssim_table.csv`, morphology overview, watershed overview | Explain noise types and why median suits salt-and-pepper while Gaussian smooths Gaussian noise. Read the PSNR/SSIM numbers from the table: `[...]`. Explain erosion/dilation/opening/closing in one sentence each, Canny, and how watershed separates touching objects (distance transform -> markers). State which technique worked best on your image: `[...]`. |
| 3:00-4:00 | Task 2: HOG image, SIFT keypoints, match images, comparison chart | HOG = histogram of gradient directions in cells. SIFT = scale/rotation-invariant keypoints with 128-value descriptors. Ratio test + RANSAC. Compare numbers: original `[...]` inliers, rotated `[...]`, resized 25 % `[...]`. Explain why. |
| 4:00-5:00 | Task 3: side-by-side comparison images | Haar cascade = Viola-Jones, fast, faces only. YOLO = deep network, many classes with confidence. Show boxes/labels/scores. Point out one miss or wrong detection you actually saw: `[...]`. Compare speed numbers from the table. |
| 5:00-6:30 | Task 4: a panel image, `motion_over_time.png`, optionally play a result video | Frame differencing = subtract consecutive frames (shows edges, fast, noisy). GMM = learns background model, gives fuller foreground silhouette. Lucas-Kanade = tracks corner points, gives direction and speed arrows. Read the comparison table: `[...]`. Mention limitation (camera shake affects all three). |
| 6:30-7:00 | `TECHNICAL_REPORT.md` conclusion, GitHub page | Short conclusion: best method per task. Repo URL and report are submitted; thanks. |

## Checklist from the assignment (all must appear)

- [ ] Introduction and objectives
- [ ] Description of the original photograph and video
- [ ] Explanation of the algorithms
- [ ] Demonstration of the source code
- [ ] Demonstration of important input and output results
- [ ] Comparison of the selected techniques
- [ ] Length 5-8 minutes, you explaining in your own voice
- [ ] Uploaded to the Google Drive location given by your teacher, link added to the report
