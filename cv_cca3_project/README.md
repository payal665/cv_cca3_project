# CCA 3 - Computer Vision Practical Assignment (Units 3, 4, 5)

Python project implementing image restoration, segmentation, feature extraction, object recognition
and motion estimation on my own photographs and self-recorded video.

| Task | File | Techniques |
|---|---|---|
| 1 (Unit 3) | `src/task1_restoration_segmentation.py` | Gaussian / salt-and-pepper noise, median + Gaussian filters, PSNR/SSIM, thresholding, erosion, dilation, opening, closing, Canny, watershed |
| 2 (Unit 4) | `src/task2_feature_matching.py` | HOG, SIFT keypoints + descriptors, SIFT matching, rotation / resize robustness |
| 3 (Unit 5) | `src/task3_object_detection.py` | Haar cascade (Viola-Jones) vs pretrained YOLOv8n |
| 4 (Unit 5) | `src/task4_motion_estimation.py` | Frame differencing, GMM (MOG2), Lucas-Kanade optical flow |

## Folder structure

```
cv_cca3_project/
  main.py                  <- runs everything
  requirements.txt
  README.md
  TECHNICAL_REPORT.md      <- generated from real results (fill the marked parts)
  VIDEO_SCRIPT.md          <- 5-8 minute explanation video plan
  GITHUB_AND_SUBMISSION.md <- git + submission steps
  input/                   <- YOUR original photo1, photo2, video
  src/                     <- code (one file per task + helpers)
  outputs/                 <- all generated images, frames, tables (created automatically)
```

## Input data (you must provide)

Put these in `input/` (see `input/README.txt`):

* `photo1.jpg` - your original photograph
* `photo2.jpg` - second photograph of the same object/scene
* `video.mp4` - your original 15-30 second video

Large raw videos should go to Google Drive (they are ignored by Git through `.gitignore`).
Drive link of my input video: `<paste link here>`

## Installation (VS Code, Windows / macOS / Linux)

1. Install Python 3.10 - 3.12 (recommended) and VS Code with the *Python* extension.
2. Open the project folder in VS Code (`File > Open Folder`), then open a terminal (`Ctrl + ~`).
3. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   ```
4. Install packages:
   ```
   pip install -r requirements.txt
   ```
   (`ultralytics` installs PyTorch, so this takes a few minutes.)
5. In VS Code press `Ctrl+Shift+P` -> *Python: Select Interpreter* -> choose the `.venv` one.

## How to run

```
python main.py --check          # shows which input files are found
python main.py                  # runs all tasks + builds TECHNICAL_REPORT.md
python main.py --tasks 1 2      # only some tasks
python src/task4_motion_estimation.py     # or run one task file directly
```

Optional: `python main.py --seg-image input/coins.jpg` uses another image for watershed
(useful if photo1 has no touching objects).

The first YOLO run downloads `yolov8n.pt` (about 6 MB), so internet is needed once.

## Outputs

Everything is written to `outputs/` automatically:

* `task1_restoration_segmentation/` - noisy, restored, binary, morphology, edge, watershed images, PSNR/SSIM table
* `task2_feature_extraction_matching/` - HOG, SIFT keypoints, match images, comparison chart
* `task3_object_recognition/` - Haar results, YOLO results, side-by-side comparisons
* `task4_motion_estimation/` - consecutive frames, masks, vectors, panels, result videos, comparison table
* `metrics/` - JSON files with all measured numbers (used for the report)

## Notes

* Random noise uses a fixed seed (42) so results are reproducible.
* All numbers in the report are measured by the code on my own input files.
* Repository: `<paste Git URL here>`
