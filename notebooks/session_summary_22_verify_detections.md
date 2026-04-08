*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

# Session Summary: Building `22_verify_detections.ipynb`

**Date:** 2026-04-08  
**Branch:** `001-scaleworm-population`  
**Repository:** `specKitScience`  
**Working directory:** `scaleworm-student-lab/notebooks/`  
**Status at session end:** Notebook built and end-to-end tested on CPU. NOT YET COMMITTED.

---

## Goal

Build a student-facing Jupyter notebook that:

1. Takes the pre-trained YOLO "Mushroom Model" and runs it on CAMHD video frames
2. Presents each detection to the student in an interactive widget
3. Lets the student label each crop as **Scale Worm** / **Not a Worm** / **Skip**
4. Auto-saves progress after every click (crash-resilient)
5. Exports verified true detections as a YOLO-format training dataset (zip)

The purpose is to let students curate labeled datasets from model predictions — the "human-in-the-loop" step of a self-training pipeline.

---

## Notebook Structure — Cell by Cell

### Cell 0 — Title & Overview (markdown)

Introduces the notebook and lists the 5-step workflow. Includes AI-generated text disclosure per project conventions. Uses `<span style="font-family: 'Courier New', monospace;">` wrapping for AI-generated prose.

### Cell 1 — Section Header: "1. Setup" (markdown)

Simple header cell.

### Cell 2 — Setup & Configuration (code)

**What it does:** Imports libraries, defines all configuration constants, creates working directories.

**Key imports:**

- `ipywidgets` — interactive verification buttons
- `matplotlib.pyplot` — displaying crops at multiple zoom levels via `imshow`
- `PIL.Image` — loading and resizing crops
- `numpy` — array operations on image data
- Standard library: `json`, `re`, `shutil`, `subprocess`, `zipfile`, `pathlib`

**Configuration constants:**

| Constant | Value | Purpose |
|---|---|---|
| `MODEL_PATH` | `viame_my-analysis/notebooks/runs/outputs/model/yolo_v26/train_v1/weights/best.pt` | Path to YOLO Mushroom Model |
| `VIDEO_ROOT` | `/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301/` | CAMHD video archive on OOI SAN storage |
| `WORK_DIR` | `./verification_session` | Session working directory (all outputs go here) |
| `SCENE1_START_SEC` | 305 | Scene 1 starts 305 seconds into each video |
| `SCENE1_DURATION_SEC` | 15 | Scene 1 is 15 seconds long |
| `FPS` | 1 | Extract 1 frame per second (changed from 10 during testing) |
| `FRAME_W, FRAME_H` | 1920, 1080 | Frame dimensions for YOLO coordinate normalization |
| `CONF_THRESHOLD` | 0.1 | Low confidence to maximize recall (students verify precision) |
| `MAX_BOX_SIZE` | 300 | Reject bounding boxes larger than 300px (real worms are 20–100px) |
| `STANDARD_TIMES` | 8 times at 3-hour intervals | Filters to standard-cadence videos only (T001500, T031500, ..., T211500) |

**Directory structure created:**

```
verification_session/
├── frames/          # Extracted video frames
├── crops/           # Cropped detection images
└── export/          # YOLO-format output dataset
```

**Design decision — FPS=1 vs FPS=10:** The original pipeline (scripts 07–12) used 10 fps for population counting. For student verification, 1 fps was chosen to reduce the number of near-duplicate detections students must review (15 frames per video instead of 150). This dramatically cuts the verification burden while still sampling every second of Scene 1.

**Design decision — CONF_THRESHOLD=0.1:** A low threshold is intentional. The goal is to show students the full spectrum of model confidence — from obvious true positives (conf=0.9+) to clear false positives (conf=0.1–0.3). This teaches them what the model finds easy vs. hard.

---

### Cell 3 — Section Header: "2. Choose your date range" (markdown)

Instructs the student to set their date range.

### Cell 4 — Video Discovery (code)

**What it does:** Defines `find_videos()` and applies it to the user-specified date range.

**Function `find_videos(video_root, start_date, end_date)`:**

1. Recursively searches `VIDEO_ROOT` for files matching `CAMHDA301-*.mp4`
2. Parses the filename to extract the date and time: `CAMHDA301-YYYYMMDDTHHmmss`
3. Filters to only dates within `[start_date, end_date]` (inclusive)
4. Filters to only standard 3-hour cadence times (rejects non-standard observation windows)
5. Returns a sorted list of `Path` objects

**Student-editable parameters:**

```python
START_DATE = "2024-10-04"   # First day (YYYY-MM-DD)
END_DATE   = "2024-10-06"   # Last day (YYYY-MM-DD)
```

**Why the standard-time filter matters:** CAMHD sometimes records at non-standard times (engineering tests, special observations). These may have different zoom positions or durations. Filtering to the 8 standard times ensures Scene 1 is consistently at 305–320 seconds.

**Output:** Prints the count and filenames of matched videos.

---

### Cell 5 — Section Header: "3. Extract Scene 1 frames" (markdown)

Explains Scene 1 (305–320s), why it matters (camera zooms onto Mushroom vent chimney), and the frame count (FPS × 15s = 15 frames/video at 1 fps).

### Cell 6 — Frame Extraction (code)

**What it does:** Extracts frames from each video's Scene 1 window using `ffmpeg`.

**Function `extract_scene1_frames(video_path, output_dir)`:**

1. Creates output directory if it doesn't exist
2. **Skip check:** If the expected number of frames already exist, returns immediately (idempotent — safe to re-run)
3. Calls `ffmpeg` via `subprocess.run()` with:
   - `-ss 305` — seek to 305 seconds
   - `-t 15` — extract 15 seconds
   - `-vf fps=1` — output 1 frame per second
   - `-q:v 2` — high JPEG quality (PNG output via filename pattern)
   - Timeout: 120 seconds
4. Returns the number of frames extracted

**Frame naming convention:** `frame_0001.png`, `frame_0002.png`, etc. (ffmpeg's `%04d` pattern)

**Directory layout after extraction:**

```
verification_session/frames/
├── CAMHDA301-20241004T001500/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ... (15 frames at 1fps)
├── CAMHDA301-20241004T031500/
│   └── ...
```

**Loop:** Iterates over all videos, extracts frames, builds the `video_frame_dirs` dictionary mapping video names to frame directories. Prints progress.

---

### Cell 7 — Section Header: "4. Run the YOLO detector" (markdown)

Explains the low-confidence strategy: intentionally over-detecting so students can verify.

### Cell 8 — Install ultralytics (code)

```python
!pip install ultralytics
```

**Note:** Uses `pip install` (not `uv add`) because this notebook is designed to run in any JupyterHub environment, not just our uv-managed project. Students may run this on their own instances.

### Cell 9 — YOLO Inference (code)

**What it does:** Loads the Mushroom Model and runs inference on all extracted frames.

**Step by step:**

1. Load model: `YOLO(str(MODEL_PATH))`
2. For each video's frames, call `model.predict()` with:
   - `conf=0.1` — low threshold
   - `verbose=False` — suppress per-frame output
   - `stream=True` — generator mode to avoid loading all results into memory
3. For each frame's results, extract bounding boxes (`boxes.xyxy`) and confidences (`boxes.conf`)
4. **Size filter:** Reject any box wider or taller than `MAX_BOX_SIZE` (300px). Real scale worms are 20–100px; large boxes are chimney structures or camera artifacts.
5. Build the `all_detections` list — each entry is a dict:

```python
{
    "video": "CAMHDA301-20241004T001500",
    "frame_file": "frame_0001.png",
    "frame_path": "/full/path/to/frame_0001.png",
    "det_idx": 0,
    "x1": 542.0, "y1": 312.0,
    "x2": 587.0, "y2": 351.0,
    "conf": 0.847,
    "label": None,  # set during verification
}
```

**Output:** Prints per-video detection counts and total.

**Design decision — `stream=True`:** Without streaming, YOLO loads all frame results into memory at once. With hundreds of frames, this can use significant RAM. Streaming processes one frame at a time.

**Design decision — `.cpu().numpy()`:** YOLO returns tensors on GPU. Converting to numpy immediately avoids accumulating GPU memory across the loop.

---

### Cell 10 — Section Header: "5. Crop detections" (markdown)

Explains the 20px padding around each crop for visual context.

### Cell 11 — Crop Extraction (code)

**What it does:** Crops each detection from its source frame, saves to disk.

**Function `load_frame(frame_path)`:**

- Opens the frame image as a numpy array via PIL
- Caches up to 50 frames in `_frame_cache` (dict)
- LRU-style eviction: drops the oldest entry when cache exceeds 50
- **Why cache:** Multiple detections often come from the same frame — caching avoids re-reading the same PNG repeatedly

**Function `crop_detection(det, pad=20)`:**

- Loads the source frame (from cache if available)
- Computes padded bounding box: `[x1-pad, y1-pad, x2+pad, y2+pad]`
- Clamps to image boundaries (0 to width/height)
- Returns the cropped region as a numpy array

**Loop:**

- Iterates over `all_detections`, crops each one, saves as `crop_000000.png` through `crop_NNNNNN.png`
- Adds `"crop_path"` key to each detection dict
- Prints progress every 500 crops
- Clears `_frame_cache` at the end to free memory

---

### Cell 12 — Section Header: "6. Verify detections" (markdown)

Instructions for the student: explains the 3-panel view (1×, 2×, 4× zoom) and the three button options.

### Cell 13 — Interactive Verification Widget (code)

**This is the largest and most complex cell.** It builds the interactive labeling UI using `ipywidgets`.

**Part 1 — Resume from saved progress:**

- Checks if `verification_progress.json` exists in `WORK_DIR`
- If yes, loads it and restores all previously assigned labels to `all_detections`
- **Why:** Students can close the notebook, restart the kernel, re-run cells 1–11, and pick up exactly where they left off. Every click saves immediately.

**Part 2 — `save_progress()` function:**

- Serializes all non-None labels as `{"labels": {"0": "scale_worm", "3": "not_worm", ...}}`
- Writes to `verification_progress.json`
- Called after every single button click (crash-resilient)

**Part 3 — Widget components:**

| Component | Type | Purpose |
|---|---|---|
| `output_images` | `widgets.Output` | Displays the 3-panel matplotlib figure (1×, 2×, 4×) |
| `output_info` | `widgets.Output` | Shows detection metadata (index, confidence, video, frame, box coords, label status) |
| `output_progress` | `widgets.Output` | Progress bar with counts (worms / not-worm / skipped) |
| `btn_worm` | `Button` (green) | Label as "scale_worm" |
| `btn_not_worm` | `Button` (red) | Label as "not_worm" |
| `btn_skip` | `Button` (yellow) | Label as "skip" |
| `btn_prev` | `Button` (default) | Go back one detection |

**Part 4 — `show_detection(idx)` function:**

1. Loads the crop from `det["crop_path"]`
2. Creates a 1×3 matplotlib figure (16×5 inches)
3. Panel 1: Original crop at 1× (native pixel size)
4. Panel 2: Crop resized 2× using `Image.NEAREST` (no interpolation — preserves pixel edges)
5. Panel 3: Crop resized 4× using `Image.NEAREST`
6. Renders with `plt.show()` inside `output_images`
7. Prints metadata line in `output_info`
8. Updates progress bar

**Design decision — `Image.NEAREST`:** Nearest-neighbor interpolation preserves the actual pixel values at higher zoom. Bilinear/bicubic would blur the crop, making it harder to see the worm's diagnostic features (leg pairs, body segmentation).

**Part 5 — `advance()` function:**

- Finds the next detection where `label is None`
- Wraps around to the beginning if needed (circular search)
- If all detections are labeled, shows a completion message

**Part 6 — Button callbacks:**

- `on_worm`: sets `label = "scale_worm"`, saves, advances
- `on_not_worm`: sets `label = "not_worm"`, saves, advances
- `on_skip`: sets `label = "skip"`, saves, advances
- `on_prev`: decrements `current_idx`, re-displays (no save — just navigation)

**Part 7 — Layout:**

```
┌─────────────────────────────────────────┐
│  Progress bar (█████░░░░ 30%)           │
├─────────────────────────────────────────┤
│  [1× crop]    [2× crop]    [4× crop]   │
├─────────────────────────────────────────┤
│  Detection 47/181 | Conf: 0.847 | ...   │
├─────────────────────────────────────────┤
│  [◀ Prev] [✓ Worm] [✗ Not] [⟳ Skip]   │
└─────────────────────────────────────────┘
```

Uses `widgets.VBox` for vertical stacking and `widgets.HBox` with `justify_content="center"` for the button bar.

---

### Cell 14 — Section Header: "7. Summary" (markdown)

Run-anytime summary cell.

### Cell 15 — Summary Statistics (code)

**What it does:** Counts labeled detections by category and prints a formatted summary.

**Output example:**

```
============================================================
VERIFICATION SUMMARY
============================================================
  Total detections:     181
  ✓ Scale worm:         42 (23.2%)
  ✗ Not a worm:         130 (71.8%)
  ⟳ Skipped:            3
  ⚪ Unlabeled:          6
  False positive rate:  75.6%
============================================================
```

**Warning:** If unlabeled detections remain, prints a reminder to finish before exporting.

---

### Cell 16 — Section Header: "8. Export verified detections as YOLO dataset" (markdown)

Explains YOLO label format: `class_id center_x center_y width height` with all coordinates normalized to [0, 1].

### Cell 17 — YOLO Dataset Export (code)

**What it does:** Packages verified "scale_worm" detections into a YOLO-format training dataset zip.

**Step by step:**

1. **Filter:** Selects only detections where `label == "scale_worm"`
2. **Group by frame:** Groups detections by their source frame path (multiple worms per frame → one label file with multiple lines)
3. **Build directory structure:**

```
export/
├── images/train/
│   ├── CAMHDA301-20241004T001500_frame_0003.png
│   └── ...
├── labels/train/
│   ├── CAMHDA301-20241004T001500_frame_0003.txt
│   └── ...
└── dataset.yaml
```

4. **`pixel_to_yolo()` conversion:**
   - Input: pixel coordinates `(x1, y1, x2, y2)`
   - Output: YOLO normalized `(center_x, center_y, width, height)`
   - All values clamped to `[0, 1]`
   - Uses `FRAME_W=1920, FRAME_H=1080` for normalization

5. **Label file format:** One line per detection:
   ```
   0 0.293750 0.308333 0.023438 0.036111
   ```
   Class 0 = `scale_worm` (single-class dataset)

6. **`dataset.yaml`** — YOLO training config:
   ```yaml
   path: .
   train: images/train
   val: images/train    # same split — student reorganizes later
   nc: 1
   names:
     0: scale_worm
   ```
   Includes metadata comments (detection count, source frames, date range, threshold).

7. **Zip creation:** Compresses the entire `export/` directory into `verified_scaleworm_dataset_{START_DATE}_to_{END_DATE}.zip`

8. **Output:** Prints export summary with counts, paths, and zip size.

---

## End-to-End Test Results (2026-04-08)

Tested on CPU (CUDA driver too old on the pre-GPU instance):

| Metric | Value |
|---|---|
| Videos processed | 1 (2024-10-04, single day) |
| Frames extracted | 15 (1 fps × 15s) |
| Detections at conf≥0.1 | 181 |
| Detections after size filter | 181 (none exceeded 300px) |
| Crops saved | 181 |
| YOLO label format verified | All normalized values in [0, 1] |
| Widget test | Simulated clicks (real widget needs JupyterLab kernel) |
| Progress save/resume | Verified — labels persist across kernel restart |
| Zip export | Verified — correct directory structure, valid YAML |

---

## Dependencies

| Package | Purpose | Install method |
|---|---|---|
| `ultralytics` | YOLO model loading and inference | `pip install` (in-notebook) |
| `ipywidgets` | Interactive verification buttons | Pre-installed in JupyterHub |
| `matplotlib` | Displaying crops at multiple zoom levels | Pre-installed |
| `Pillow` (PIL) | Image loading, resizing, saving | Pre-installed |
| `numpy` | Array operations | Pre-installed |
| `ffmpeg` | Frame extraction from video | System binary (pre-installed) |

---

## Key Design Decisions Summary

1. **FPS=1** (not 10): Reduces verification burden for students. 15 frames/video instead of 150.
2. **CONF_THRESHOLD=0.1**: Intentionally low to show students the full confidence spectrum.
3. **MAX_BOX_SIZE=300**: Filters out chimney-scale false positives without rejecting any real worms (20–100px).
4. **NEAREST interpolation for zoom**: Preserves pixel-level detail rather than blurring.
5. **Save after every click**: Crash-resilient. Students never lose work.
6. **Circular advance**: After labeling, jumps to next unlabeled detection. Wraps around so students can revisit skipped items.
7. **Single-class YOLO export**: `scale_worm` is class 0. `val` points to `train` — students can split later if needed.
8. **`pip install` not `uv add`**: Notebook is designed for any JupyterHub, not just our uv project.

---

## Files Created

| File | Location | Purpose |
|---|---|---|
| `22_verify_detections.ipynb` | `scaleworm-student-lab/notebooks/` | The notebook itself |

**Runtime outputs** (created when notebook is executed, not tracked in git):

| File/Directory | Location | Purpose |
|---|---|---|
| `verification_session/frames/` | Notebook working dir | Extracted Scene 1 frames |
| `verification_session/crops/` | Notebook working dir | Cropped detection images |
| `verification_session/verification_progress.json` | Notebook working dir | Saved labels (auto-save) |
| `verification_session/export/` | Notebook working dir | YOLO dataset structure |
| `verification_session/verified_scaleworm_dataset_*.zip` | Notebook working dir | Final export zip |

---

## Resume Checklist

1. **Commit** `22_verify_detections.ipynb` to `001-scaleworm-population` branch
2. **Test widget interactively** in JupyterLab with GPU (simulated clicks during testing — real widget needs a live kernel)
3. **Verify GPU inference speed** — the L40S should run YOLO at ~119 fps vs. CPU's ~2 fps
4. **Discuss refinements** before distributing to students
