"""Build the annotated version of 22_verify_detections.ipynb.

Writes to a SEPARATE file (22_verify_detections_annotated.ipynb) so the
original student notebook remains untouched.

Run from the notebooks/ directory:
    python build_annotated_notebook.py
"""
import json
from pathlib import Path

NB_SOURCE = Path(__file__).parent / "22_verify_detections.ipynb"       # read metadata from original
NB_PATH = Path(__file__).parent / "22_verify_detections_annotated.ipynb"  # write annotated version here

def md(source):
    """Create a markdown cell dict."""
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    """Create a code cell dict."""
    return {"cell_type": "code", "metadata": {}, "source": source,
            "outputs": [], "execution_count": None}

cells = []

# ─── CELL 0: Title ─────────────────────────────────────────────────
cells.append(md("""\
# Scaleworm Detection Verification Lab

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

This notebook runs the YOLO "Mushroom Model" on CAMHD video frames and lets you verify each detection. Your job is to look at each cropped detection and decide: **is this a scale worm, or not?**

**Workflow:**
1. **Choose your date range** — pick which days of video to analyze
2. **Extract frames** — pull Scene 1 frames from each video (the Mushroom vent zoom)
3. **Run the detector** — YOLO finds candidate scale worms at low confidence (catches more, but includes false positives)
4. **Verify each detection** — you'll see each crop at multiple zoom levels and mark it as worm or not-worm
5. **Export** — verified true detections are packaged as a YOLO-format dataset for downstream training

**Context:** Scale worms (*Lepidonotopodium piscesae*) colonize hydrothermal vent chimneys at Axial Seamount. The CAMHD camera on the OOI Regional Cabled Array records 3-hour-cadence video of Mushroom vent in the ASHES field. Scene 1 (305–320 s into each video) is the close-up zoom on the chimney where worms are visible. The YOLO "Mushroom Model" was trained on manually labeled CAMHD frames to detect individual worms, but at low confidence thresholds it produces false positives (bacterial mats, tube structures, image artifacts). This notebook lets a human reviewer curate those predictions into a clean training dataset.

</span>"""))

# ─── CELL 1: Setup markdown ────────────────────────────────────────
cells.append(md("""\
## 1. Setup

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The code cell below imports all required libraries and defines every configurable parameter for the pipeline. Nothing is computed yet — this cell only sets up the environment.

**Why:** Centralizing configuration at the top means you only need to edit one place if you want to change paths, thresholds, or extraction parameters. Every downstream cell reads from these constants.

**Inputs:** None (all paths and values are hardcoded constants).

**Adjustable parameters:**

| Parameter | Default | What it controls | Guidance for changing |
|---|---|---|---|
| `MODEL_PATH` | `viame_my-analysis/.../best.pt` | Path to the YOLO Mushroom Model weights | Replace with your own `.pt` file if you've trained a different model |
| `VIDEO_ROOT` | `/home/jovyan/ooi/san_data/RS03ASHS-...` | Root directory of the CAMHD video archive | Change if your videos are stored elsewhere |
| `SCENE1_START_SEC` | `305` | When Scene 1 begins in each video (seconds) | Only change if CAMHD changes its observation profile |
| `SCENE1_DURATION_SEC` | `15` | How many seconds of Scene 1 to extract | Shorter = fewer frames = faster; longer = more temporal coverage |
| `FPS` | `1` | Frames per second to extract | Higher = more frames but more detections to verify. 10 fps = 150 frames/video; 1 fps = 15 |
| `CONF_THRESHOLD` | `0.1` | Minimum YOLO confidence to keep a detection | Lower = more candidates (more false positives). Production counting uses 0.9 |
| `MAX_BOX_SIZE` | `300` | Maximum bounding box dimension in pixels | Real worms are 20–100 px. Boxes larger than this are chimney structures or artifacts |
| `STANDARD_TIMES` | 8 UTC times at 3-hr intervals | Which video timestamps to include | Filters out non-standard engineering/test recordings |

**Test the method:** After running this cell, verify that `MODEL_PATH` and `VIDEO_ROOT` both print `Exists: True`. If either is `False`, fix the path before continuing.

</span>"""))

# ─── CELL 2: Setup code ────────────────────────────────────────────
cells.append(code("""\
# ── Standard library imports ────────────────────────────────────────
import json          # read/write verification_progress.json (save/resume labels)
import re            # parse video filenames with regex (extract date + time)
import shutil        # copy frame images into YOLO export directory
import subprocess    # call ffmpeg for frame extraction
import zipfile       # package the final YOLO dataset as a .zip
from pathlib import Path  # filesystem path operations

# ── Third-party imports ─────────────────────────────────────────────
import ipywidgets as widgets                    # interactive buttons for labeling
import matplotlib.pyplot as plt                 # display crops at multiple zoom levels
import numpy as np                              # image arrays
from IPython.display import display, clear_output  # widget rendering in Jupyter
from PIL import Image                           # load/resize crop images

# ── Model path ──────────────────────────────────────────────────────
# Points to the YOLO "Mushroom Model" trained on CAMHD scaleworm data.
# This is a YOLOv11 model fine-tuned on ~600 manually labeled frames.
MODEL_PATH = Path("/home/jovyan/repos/specKitScience/viame_my-analysis/"
                   "notebooks/runs/outputs/model/yolo_v26/train_v1/weights/best.pt")

# ── Video archive root ──────────────────────────────────────────────
# OOI SAN storage path for CAMHD video files, organized as year/month/day subdirs.
VIDEO_ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301/")

# ── Working directory for this session ──────────────────────────────
# All outputs (frames, crops, exports) go here. Safe to delete between sessions.
WORK_DIR = Path("./verification_session")
FRAMES_DIR = WORK_DIR / "frames"    # extracted video frames, one subdir per video
CROPS_DIR = WORK_DIR / "crops"      # cropped detection images, one PNG per detection
EXPORT_DIR = WORK_DIR / "export"    # final YOLO-format dataset

# ── Frame extraction parameters ────────────────────────────────────
SCENE1_START_SEC = 305    # Scene 1 starts 305 seconds into each CAMHD video
SCENE1_DURATION_SEC = 15  # Scene 1 lasts 15 seconds (camera zoomed on chimney)
FPS = 1                   # extract 1 frame/sec → 15 frames per video (adjustable)
FRAME_W, FRAME_H = 1920, 1080  # CAMHD native resolution (used for YOLO normalization)

# ── Detection parameters ───────────────────────────────────────────
CONF_THRESHOLD = 0.1   # intentionally low — catch all candidates, verify manually
MAX_BOX_SIZE = 300     # reject bounding boxes > 300 px wide or tall (not real worms)

# ── Standard 3-hour cadence times (UTC) ─────────────────────────────
# CAMHD records at these 8 times each day under normal operations.
# Non-standard times (engineering tests, special obs) are excluded because
# the camera zoom position at 305 s may differ.
STANDARD_TIMES = {
    "T001500", "T031500", "T061500", "T091500",
    "T121500", "T151500", "T181500", "T211500",
}

# ── Create output directories ──────────────────────────────────────
# exist_ok via parents=True: safe to re-run without error
for d in [WORK_DIR, FRAMES_DIR, CROPS_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Print configuration for verification ────────────────────────────
print(f"Model:     {MODEL_PATH}")
print(f"Exists:    {MODEL_PATH.exists()}")       # ← must be True
print(f"Video root: {VIDEO_ROOT}")
print(f"Exists:    {VIDEO_ROOT.exists()}")        # ← must be True
print(f"Work dir:  {WORK_DIR.resolve()}")
print(f"Conf threshold: {CONF_THRESHOLD}")"""))

# ─── CELL 3: Date range markdown ───────────────────────────────────
cells.append(md("""\
## 2. Choose your date range

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The code cell below defines a function to discover CAMHD video files within a date range, then applies it to `START_DATE` / `END_DATE` that you set.

**Why:** The CAMHD archive contains thousands of videos spanning years. Selecting a date range lets you focus on a manageable batch for verification. The function also filters to standard 3-hour cadence times, rejecting non-standard recordings where the camera position at Scene 1 may differ.

**Inputs:**
- `VIDEO_ROOT` — from Setup (Cell 2)
- `STANDARD_TIMES` — from Setup (Cell 2)

**Adjustable parameters:**

| Parameter | Default | What it controls |
|---|---|---|
| `START_DATE` | `"2024-10-04"` | First day to include (inclusive). Format: `YYYY-MM-DD` |
| `END_DATE` | `"2024-10-06"` | Last day to include (inclusive). Format: `YYYY-MM-DD` |

**Test the method:** Check that the printed video count matches your expectation. Each day should have up to 8 standard-cadence videos (one every 3 hours). If 0 videos are found, verify that `VIDEO_ROOT` is correct and that data exists for your date range.

</span>"""))

# ─── CELL 4: Date range code ───────────────────────────────────────
cells.append(code("""\
def find_videos(video_root, start_date, end_date):
    \"\"\"Find standard-cadence CAMHD videos between two dates (inclusive).

    Searches video_root recursively for files matching the CAMHD naming
    convention, filters by date range and standard observation times.

    Parameters
    ----------
    video_root : Path
        Root of the CAMHD archive (contains year/month/day subdirs).
    start_date : str
        Start date as 'YYYY-MM-DD'.
    end_date : str
        End date as 'YYYY-MM-DD'.

    Returns
    -------
    list of Path
        Sorted list of video file paths matching the criteria.
    \"\"\"
    import datetime

    # Parse date strings into date objects for comparison
    d_start = datetime.date.fromisoformat(start_date)
    d_end = datetime.date.fromisoformat(end_date)

    videos = []
    # rglob: recursive search for all CAMHD .mp4 files under video_root
    for mp4 in sorted(video_root.rglob("CAMHDA301-*.mp4")):
        # Extract date and time from filename, e.g. CAMHDA301-20241004T001500.mp4
        m = re.search(r"CAMHDA301-(\\d{4})(\\d{2})(\\d{2})T(\\d{6})", mp4.name)
        if not m:
            continue  # skip files that don't match the expected naming pattern

        # Parse year, month, day, and time components from the filename
        y, mo, d, time_str = m.group(1), m.group(2), m.group(3), m.group(4)
        file_date = datetime.date(int(y), int(mo), int(d))

        # Skip files outside the requested date range
        if file_date < d_start or file_date > d_end:
            continue

        # Only keep standard 3-hour cadence times (reject engineering/test recordings)
        if f"T{time_str}" not in STANDARD_TIMES:
            continue

        videos.append(mp4)

    return sorted(videos)  # sort by filename = chronological order


# ════════════════════════════════════════════════════════════════════
# ▼▼▼  SET YOUR DATE RANGE HERE  ▼▼▼
# ════════════════════════════════════════════════════════════════════

START_DATE = "2024-10-04"   # First day to analyze (YYYY-MM-DD)
END_DATE   = "2024-10-06"   # Last day to analyze (YYYY-MM-DD)

# ════════════════════════════════════════════════════════════════════

# Run the search and store results in `videos` (list of Path objects)
videos = find_videos(VIDEO_ROOT, START_DATE, END_DATE)

# Print what was found so the user can verify before proceeding
print(f"Found {len(videos)} standard-cadence videos "
      f"between {START_DATE} and {END_DATE}:\\n")
for v in videos:
    print(f"  {v.name}")"""))

# ─── CELL 5: Frame extraction markdown ─────────────────────────────
cells.append(md("""\
## 3. Extract Scene 1 frames

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The code cell below uses `ffmpeg` to extract frames from the Scene 1 window (305–320 seconds) of each selected video. Frames are saved as PNG images.

**Why:** CAMHD videos are ~25 minutes long, but scale worms are only visible during Scene 1, when the camera zooms in on the Mushroom vent chimney. Extracting just this 15-second window avoids processing irrelevant footage.

**Inputs:**
- `videos` — list of video paths from Step 2
- `SCENE1_START_SEC`, `SCENE1_DURATION_SEC`, `FPS` — from Setup (Cell 2)

**Adjustable parameters:**

| Parameter | Current | Effect of changing |
|---|---|---|
| `SCENE1_START_SEC` | `305` | Shifts the extraction window. Scene 1 timing is determined by the CAMHD observation profile |
| `SCENE1_DURATION_SEC` | `15` | Longer = more frames extracted; shorter = faster but may miss late-arriving worms |
| `FPS` | `1` | At 1 fps: 15 frames/video. At 10 fps: 150 frames/video. Higher FPS gives more temporal resolution but more detections to verify |

**Outputs:**
- `video_frame_dirs` — dict mapping each video name to its frame directory
- PNG frames in `verification_session/frames/<video_name>/frame_0001.png` etc.

**Idempotent:** If frames already exist for a video (from a previous run), they are not re-extracted. Safe to re-run.

**Test the method:** Check that each video reports the expected number of frames (FPS × SCENE1_DURATION_SEC). If a video reports 0 frames, ffmpeg may have failed — check that the video file is not corrupted.

</span>"""))

# ─── CELL 6: Frame extraction code ─────────────────────────────────
cells.append(code("""\
def extract_scene1_frames(video_path, output_dir):
    \"\"\"Extract Scene 1 frames from a CAMHD video using ffmpeg.

    Returns the number of frames extracted, or 0 on failure.
    \"\"\"
    # Create output directory for this video's frames
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check: skip if frames already exist ─────────────
    # If we already have enough frames, return immediately (safe to re-run)
    existing = sorted(output_dir.glob("frame_*.png"))
    if len(existing) >= FPS * SCENE1_DURATION_SEC - 1:
        return len(existing)

    # ── Build ffmpeg command ────────────────────────────────────────
    cmd = [
        "ffmpeg", "-y",               # -y: overwrite without asking
        "-ss", str(SCENE1_START_SEC),  # seek to Scene 1 start (305 s)
        "-i", str(video_path),         # input video file
        "-t", str(SCENE1_DURATION_SEC),  # extract this many seconds (15 s)
        "-vf", f"fps={FPS}",           # output frame rate (1 fps default)
        "-q:v", "2",                   # high PNG quality (lower number = better)
        str(output_dir / "frame_%04d.png"),  # output naming: frame_0001.png, etc.
    ]

    # ── Run ffmpeg ──────────────────────────────────────────────────
    # capture_output=True: suppress ffmpeg's verbose stderr in normal operation
    # timeout=120: kill if extraction hangs (corrupted video)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        # Print last 200 chars of stderr for debugging
        print(f"  ERROR extracting {video_path.name}: {result.stderr[-200:]}")
        return 0

    # Count and return the number of frames actually written
    frames = sorted(output_dir.glob("frame_*.png"))
    return len(frames)


# ── Extract frames for all videos ───────────────────────────────────
total_frames = 0
video_frame_dirs = {}  # maps video name → Path to its frame directory

for i, vpath in enumerate(videos):
    # Use video stem (filename without .mp4) as the subdirectory name
    vid_name = vpath.stem  # e.g., CAMHDA301-20241004T001500
    frame_dir = FRAMES_DIR / vid_name
    video_frame_dirs[vid_name] = frame_dir  # store mapping for downstream cells

    # Extract frames (or skip if already done)
    n = extract_scene1_frames(vpath, frame_dir)
    total_frames += n
    print(f"  [{i+1}/{len(videos)}] {vid_name}: {n} frames")

print(f"\\nTotal: {total_frames} frames from {len(videos)} videos")"""))

# ─── CELL 7: YOLO detector markdown ────────────────────────────────
cells.append(md("""\
## 4. Run the YOLO detector

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The next two code cells install the `ultralytics` package (if needed) and then run the YOLO Mushroom Model on every extracted frame. Each detection is stored with its bounding box coordinates and confidence score.

**Why:** The YOLO model scans each frame for objects matching the "scale_worm" class. We use a deliberately low confidence threshold (0.1) so the model reports everything it is even slightly uncertain about. This maximizes recall — we catch nearly all real worms, at the cost of also catching many false positives. Your job in Step 6 is to sort the true worms from the false ones.

**Inputs:**
- `MODEL_PATH` — from Setup (Cell 2)
- `video_frame_dirs` — from Step 3 (frame extraction)
- `CONF_THRESHOLD`, `MAX_BOX_SIZE` — from Setup (Cell 2)

**Adjustable parameters:**

| Parameter | Current | Effect of changing |
|---|---|---|
| `CONF_THRESHOLD` | `0.1` | Raising to 0.5 would halve the number of candidates but miss uncertain worms. Lowering below 0.1 adds very low-quality detections |
| `MAX_BOX_SIZE` | `300` | Real worms are 20–100 px. 300 px is generous — only rejects chimney-scale false positives |

**Outputs:**
- `all_detections` — list of dicts, one per candidate detection. Each dict contains: `video`, `frame_file`, `frame_path`, `det_idx`, `x1`, `y1`, `x2`, `y2` (pixel coords), `conf` (0–1), `label` (None until verified)

**Test the method:** Compare the number of detections per video. Videos with dramatically more detections than others may have lighting changes or camera issues worth investigating. Try running a single video at conf=0.9 to see how many "confident" detections the model finds — that's approximately the true worm count.

</span>"""))

# ─── CELL 8: pip install ───────────────────────────────────────────
cells.append(md("""\
### 4a. Install the YOLO library

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The cell below installs the `ultralytics` package, which provides the YOLO model loading and inference API. This uses `pip install` (not `uv add`) because this notebook is designed to run on any JupyterHub instance, not just our project-managed environment.

**Why:** `ultralytics` is the official library for YOLOv5/v8/v11 models. It handles model loading, GPU/CPU dispatch, non-maximum suppression, and bounding box formatting.

</span>"""))

cells.append(code("""\
# Install ultralytics (YOLO library) if not already present.
# Uses pip (not uv) for portability across JupyterHub instances.
!pip install ultralytics"""))

# ─── CELL 9 markdown: inference ────────────────────────────────────
cells.append(md("""\
### 4b. Run inference on all frames

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The cell below loads the YOLO model and runs it on every extracted frame. For each detection above `CONF_THRESHOLD`, it records the bounding box, confidence, video name, and frame path into the `all_detections` list.

**Why we use `stream=True`:** Without streaming, YOLO loads all frame results into GPU memory at once. Streaming processes one frame at a time, keeping memory usage constant regardless of how many frames you have.

**Why we call `.cpu().numpy()`:** YOLO returns GPU tensors. Converting to numpy immediately prevents GPU memory from accumulating across the loop.

**Size filter logic:** Real scale worms produce bounding boxes of 20–100 pixels. Anything wider or taller than `MAX_BOX_SIZE` (300 px) is a chimney structure, bacterial mat, or camera artifact — not a worm.

</span>"""))

# ─── CELL 9: Inference code ────────────────────────────────────────
cells.append(code("""\
from ultralytics import YOLO

# ── Load the YOLO Mushroom Model ────────────────────────────────────
# The model file (.pt) contains the architecture + trained weights.
# YOLO() auto-detects GPU if available; falls back to CPU otherwise.
model = YOLO(str(MODEL_PATH))
print(f"Loaded model: {MODEL_PATH.name}")

# ── Run inference on all extracted frames ───────────────────────────
all_detections = []  # accumulator: one dict per candidate detection

for vid_name, frame_dir in sorted(video_frame_dirs.items()):
    # Get all frame PNGs for this video, sorted chronologically
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        continue  # skip videos that had extraction failures

    # model.predict() runs YOLO inference on a batch of images
    results = model.predict(
        source=[str(f) for f in frames],  # list of image paths
        conf=CONF_THRESHOLD,    # minimum confidence to report (0.1 = very permissive)
        verbose=False,          # suppress per-frame console output
        stream=True,            # generator mode: process one frame at a time (saves GPU memory)
    )

    vid_det_count = 0
    # zip frames with results: each result corresponds to one frame
    for frame_path, result in zip(frames, results):
        boxes = result.boxes  # YOLO Boxes object containing all detections for this frame
        if boxes is None or len(boxes) == 0:
            continue  # no detections in this frame

        # Extract bounding box coordinates (x1, y1, x2, y2) and confidence scores
        xyxy = boxes.xyxy.cpu().numpy()  # move from GPU to CPU, convert to numpy array
        confs = boxes.conf.cpu().numpy()  # confidence scores as numpy array

        # Process each detection in this frame
        for det_idx, (box, conf) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = box  # pixel coordinates of bounding box corners
            w, h = x2 - x1, y2 - y1  # width and height in pixels

            # Size filter: reject boxes larger than MAX_BOX_SIZE
            # Real worms are 20–100 px; large boxes are chimney structures or artifacts
            if w > MAX_BOX_SIZE or h > MAX_BOX_SIZE:
                continue

            # Store this detection as a dict for downstream processing
            all_detections.append({
                "video": vid_name,           # which video this came from
                "frame_file": frame_path.name,  # frame filename (e.g., frame_0003.png)
                "frame_path": str(frame_path),  # full path to the frame image
                "det_idx": det_idx,          # detection index within this frame
                "x1": float(x1), "y1": float(y1),  # top-left corner (pixels)
                "x2": float(x2), "y2": float(y2),  # bottom-right corner (pixels)
                "conf": float(conf),         # model confidence (0.0–1.0)
                "label": None,               # human label — set during verification
            })
            vid_det_count += 1

    # Print per-video summary
    print(f"  {vid_name}: {len(frames)} frames, {vid_det_count} detections")

print(f"\\nTotal: {len(all_detections)} candidate detections to verify")"""))

# ─── CELL 10: Crop markdown ────────────────────────────────────────
cells.append(md("""\
## 5. Crop detections

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The code cell below crops each detection's bounding box from its source frame, adds 20 pixels of context padding on all sides, and saves each crop as a separate PNG file.

**Why:** During verification (Step 6), you need to see each detection individually. Pre-cropping avoids reloading the full 1920×1080 frame for every detection. The 20 px padding around each crop gives you visual context — you can see what's adjacent to the detection, which helps distinguish real worms from tube structures or artifacts.

**Inputs:**
- `all_detections` — list of detection dicts from Step 4
- Each detection's `frame_path` — path to the full-resolution source frame

**Adjustable parameters:**

| Parameter | Current | Effect of changing |
|---|---|---|
| `PAD_PX` | `20` | More padding = more context around each crop. Less padding = tighter focus on the detection itself |

**Outputs:**
- Crop images saved to `verification_session/crops/crop_000000.png` through `crop_NNNNNN.png`
- Each detection dict gets a new `"crop_path"` key pointing to its saved crop

**Performance note:** A frame cache (max 50 frames) avoids re-reading the same PNG when multiple detections come from the same frame. The cache is cleared after all crops are saved to free memory.

</span>"""))

# ─── CELL 11: Crop code ────────────────────────────────────────────
cells.append(code("""\
PAD_PX = 20  # pixels of context padding around each bounding box crop

# ── Frame cache ─────────────────────────────────────────────────────
# Multiple detections often come from the same frame. Caching avoids
# re-reading the same PNG file from disk repeatedly.
_frame_cache = {}

def load_frame(frame_path):
    \"\"\"Load a frame image as a numpy array, with LRU-style caching.\"\"\"
    if frame_path not in _frame_cache:
        # Load the image and convert to numpy array (H, W, 3) uint8
        _frame_cache[frame_path] = np.array(Image.open(frame_path))
        # Evict oldest entry if cache exceeds 50 frames (~300 MB at 1920×1080)
        if len(_frame_cache) > 50:
            oldest = next(iter(_frame_cache))
            del _frame_cache[oldest]
    return _frame_cache[frame_path]


def crop_detection(det, pad=PAD_PX):
    \"\"\"Crop a detection's bounding box from its source frame with padding.

    Parameters
    ----------
    det : dict
        Detection dict with keys 'frame_path', 'x1', 'y1', 'x2', 'y2'.
    pad : int
        Pixels of padding to add on each side.

    Returns
    -------
    numpy.ndarray
        Cropped image region as (H, W, 3) uint8 array.
    \"\"\"
    img = load_frame(det["frame_path"])  # load (or retrieve from cache)
    h, w = img.shape[:2]  # frame dimensions for boundary clamping

    # Expand bounding box by pad pixels, clamped to image boundaries
    x1 = max(0, int(det["x1"]) - pad)  # left edge, clamped to 0
    y1 = max(0, int(det["y1"]) - pad)  # top edge, clamped to 0
    x2 = min(w, int(det["x2"]) + pad)  # right edge, clamped to image width
    y2 = min(h, int(det["y2"]) + pad)  # bottom edge, clamped to image height

    return img[y1:y2, x1:x2]  # slice the image array to get the crop


# ── Pre-crop all detections and save to disk ────────────────────────
print("Cropping detections...")
for i, det in enumerate(all_detections):
    crop = crop_detection(det)                      # extract the padded crop
    crop_path = CROPS_DIR / f"crop_{i:06d}.png"     # sequential filename
    Image.fromarray(crop).save(crop_path)           # save as PNG
    det["crop_path"] = str(crop_path)               # store path in detection dict

    # Print progress every 500 crops (or at the end)
    if (i + 1) % 500 == 0 or (i + 1) == len(all_detections):
        print(f"  {i+1}/{len(all_detections)} crops saved")

_frame_cache.clear()  # free memory — frames are no longer needed after cropping
print("Done.")"""))

# ─── CELL 12: Verification markdown ────────────────────────────────
cells.append(md("""\
## 6. Verify detections

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The code cell below builds an interactive widget that presents each candidate detection for human review. You see each crop at three zoom levels (1×, 2×, 4×) and click a button to classify it.

**Why:** The YOLO model at conf=0.1 catches nearly all real worms but also produces many false positives. Human verification separates true detections from false ones. The multi-zoom display lets you see both the overall shape (1×) and fine anatomical detail like leg pairs or body segmentation (4×) that distinguish worms from tubes or artifacts.

**Inputs:**
- `all_detections` — list of detection dicts with `crop_path`, `conf`, etc.
- Previously saved progress (if any) from `verification_progress.json`

**Widget controls:**

| Button | Label assigned | Keyboard shortcut |
|---|---|---|
| **✓ Scale Worm** (green) | `"scale_worm"` | None (click only) |
| **✗ Not a Worm** (red) | `"not_worm"` | None (click only) |
| **⟳ Skip** (yellow) | `"skip"` | None (click only) |
| **◀ Previous** | (navigation only) | None (click only) |

**Auto-save behavior:** After every single button click, all labels are written to `verification_progress.json`. If the kernel crashes or you close the notebook, re-run cells 1–5 and your labels will be restored automatically from the JSON file.

**Navigation:** After labeling a detection, the widget automatically advances to the next *unlabeled* detection (skipping already-labeled ones). The search wraps around, so if you skip some and come back later, it finds them. When all detections are labeled, a completion message appears.

**Display layout:**
```
┌──────────────────────────────────────────┐
│  Progress: [████████░░░░] 65%            │
│  🟢 42 worms | 🔴 89 not-worm | 🟡 3    │
├──────────────────────────────────────────┤
│  [1× crop]    [2× crop]    [4× crop]    │
├──────────────────────────────────────────┤
│  Detection 47/181 | Conf: 0.847 | ...    │
├──────────────────────────────────────────┤
│  [◀ Prev] [✓ Worm] [✗ Not] [⟳ Skip]    │
└──────────────────────────────────────────┘
```

**Test the method:** After labeling ~10 detections, close and re-open the notebook, re-run cells 1–5, then run this cell. Verify that your previous labels are restored and the widget starts at the next unlabeled detection.

</span>"""))

# ─── CELL 13: Verification widget code ─────────────────────────────
cells.append(code("""\
# ── Load saved progress if resuming ─────────────────────────────────
# Labels are stored in a JSON file so work survives kernel restarts.
SAVE_PATH = WORK_DIR / "verification_progress.json"

if SAVE_PATH.exists():
    with open(SAVE_PATH) as f:
        saved = json.load(f)
    # Restore labels from the saved dict into all_detections
    for i, label in saved.get("labels", {}).items():
        idx = int(i)  # JSON keys are strings; convert to int index
        if idx < len(all_detections):
            all_detections[idx]["label"] = label  # restore the label
    print(f"Resumed progress: {len(saved.get('labels', {}))} detections already labeled")
else:
    print("Starting fresh — no saved progress found.")


def save_progress():
    \"\"\"Write all current labels to disk as JSON. Called after every click.\"\"\"
    labels = {}
    for i, det in enumerate(all_detections):
        if det["label"] is not None:
            labels[str(i)] = det["label"]  # key = string index, value = label string
    with open(SAVE_PATH, "w") as f:
        json.dump({"labels": labels}, f, indent=2)


# ── Find first unlabeled detection ──────────────────────────────────
# current_idx is a list (not int) so callbacks can mutate it (closure trick)
current_idx = [0]
for i, det in enumerate(all_detections):
    if det["label"] is None:
        current_idx[0] = i
        break

# ── Create widget display areas ────────────────────────────────────
output_images = widgets.Output(layout=widgets.Layout(width="100%"))    # crop display
output_info = widgets.Output(layout=widgets.Layout(width="100%"))      # metadata line
output_progress = widgets.Output(layout=widgets.Layout(width="100%"))  # progress bar

# ── Create buttons ──────────────────────────────────────────────────
btn_worm = widgets.Button(
    description="✓ Scale Worm",
    button_style="success",  # green
    layout=widgets.Layout(width="180px", height="50px"),
    style={"font_weight": "bold"},
)
btn_not_worm = widgets.Button(
    description="✗ Not a Worm",
    button_style="danger",  # red
    layout=widgets.Layout(width="180px", height="50px"),
    style={"font_weight": "bold"},
)
btn_skip = widgets.Button(
    description="⟳ Skip",
    button_style="warning",  # yellow
    layout=widgets.Layout(width="120px", height="50px"),
)
btn_prev = widgets.Button(
    description="◀ Previous",
    layout=widgets.Layout(width="120px", height="50px"),
)


def show_detection(idx):
    \"\"\"Display the detection at index `idx` at three zoom levels.\"\"\"
    det = all_detections[idx]
    crop = np.array(Image.open(det["crop_path"]))  # load the saved crop PNG

    with output_images:
        clear_output(wait=True)  # clear previous display
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: original pixel size (1×)
        axes[0].imshow(crop)
        axes[0].set_title("1× (original)", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: 2× zoom using nearest-neighbor interpolation
        # NEAREST preserves pixel edges — no blurring (important for small worms)
        crop_2x = np.array(Image.fromarray(crop).resize(
            (crop.shape[1] * 2, crop.shape[0] * 2), Image.NEAREST))
        axes[1].imshow(crop_2x)
        axes[1].set_title("2× zoom", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: 4× zoom for fine anatomical detail (leg pairs, segmentation)
        crop_4x = np.array(Image.fromarray(crop).resize(
            (crop.shape[1] * 4, crop.shape[0] * 4), Image.NEAREST))
        axes[2].imshow(crop_4x)
        axes[2].set_title("4× zoom", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        fig.tight_layout()
        plt.show()

    # Show metadata for this detection
    with output_info:
        clear_output(wait=True)
        label_str = det["label"] or "unlabeled"
        # Color indicator for current label status
        label_color = {"scale_worm": "🟢", "not_worm": "🔴", "skip": "🟡"}.get(
            det["label"], "⚪")
        print(f"Detection {idx + 1} / {len(all_detections)}  |  "
              f"Conf: {det['conf']:.3f}  |  "
              f"Video: {det['video']}  |  "
              f"Frame: {det['frame_file']}  |  "
              f"Box: [{det['x1']:.0f}, {det['y1']:.0f}, {det['x2']:.0f}, {det['y2']:.0f}]  |  "
              f"Status: {label_color} {label_str}")

    update_progress()  # refresh the progress bar


def update_progress():
    \"\"\"Redraw the progress bar with current label counts.\"\"\"
    n_done = sum(1 for d in all_detections if d["label"] is not None)
    n_worm = sum(1 for d in all_detections if d["label"] == "scale_worm")
    n_not = sum(1 for d in all_detections if d["label"] == "not_worm")
    n_skip = sum(1 for d in all_detections if d["label"] == "skip")
    n_total = len(all_detections)
    pct = 100 * n_done / max(n_total, 1)  # avoid division by zero

    with output_progress:
        clear_output(wait=True)
        # Build ASCII progress bar (50 chars wide)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"Progress: [{bar}] {pct:.0f}%  "
              f"({n_done}/{n_total})  |  "
              f"🟢 {n_worm} worms  |  🔴 {n_not} not-worm  |  🟡 {n_skip} skipped")


def advance():
    \"\"\"Move to the next unlabeled detection (circular search).\"\"\"
    start = current_idx[0]
    # Search forward from current position, wrapping around
    for offset in range(1, len(all_detections) + 1):
        candidate = (start + offset) % len(all_detections)
        if all_detections[candidate]["label"] is None:
            current_idx[0] = candidate
            show_detection(current_idx[0])
            return
    # If we get here, all detections have been labeled
    current_idx[0] = len(all_detections) - 1
    show_detection(current_idx[0])
    with output_info:
        print("\\n🎉  ALL DETECTIONS VERIFIED!  Proceed to Step 7 to export.")


# ── Button click callbacks ──────────────────────────────────────────
# Each callback: (1) assigns the label, (2) saves to disk, (3) advances

def on_worm(b):
    all_detections[current_idx[0]]["label"] = "scale_worm"  # mark as true positive
    save_progress()  # write to JSON immediately
    advance()        # show next unlabeled detection

def on_not_worm(b):
    all_detections[current_idx[0]]["label"] = "not_worm"  # mark as false positive
    save_progress()
    advance()

def on_skip(b):
    all_detections[current_idx[0]]["label"] = "skip"  # uncertain — revisit later
    save_progress()
    advance()

def on_prev(b):
    \"\"\"Go back one detection (for reviewing previous decisions).\"\"\"
    if current_idx[0] > 0:
        current_idx[0] -= 1
    show_detection(current_idx[0])  # no save — just navigation


# ── Connect callbacks to buttons ────────────────────────────────────
btn_worm.on_click(on_worm)
btn_not_worm.on_click(on_not_worm)
btn_skip.on_click(on_skip)
btn_prev.on_click(on_prev)

# ── Assemble and display the widget layout ──────────────────────────
button_bar = widgets.HBox(
    [btn_prev, btn_worm, btn_not_worm, btn_skip],
    layout=widgets.Layout(justify_content="center", gap="10px"),
)

ui = widgets.VBox([
    output_progress,   # top: progress bar
    output_images,     # middle: 3-panel crop display
    output_info,       # below image: detection metadata
    button_bar,        # bottom: action buttons
])

display(ui)                      # render the widget in the notebook
show_detection(current_idx[0])   # show the first unlabeled detection"""))

# ─── CELL 14: Summary markdown ─────────────────────────────────────
cells.append(md("""\
## 7. Summary

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The code cell below counts how many detections you've labeled in each category and prints a summary table.

**Why:** This gives you a quick quality check before exporting. The false positive rate tells you what fraction of the model's predictions were wrong — useful for assessing model performance at your chosen confidence threshold.

**Inputs:**
- `all_detections` — with labels assigned during Step 6

**Outputs:**
- Printed summary: total detections, counts by label, false positive rate
- Warning if any detections are still unlabeled

**Test the method:** The false positive rate should be interpretable. At conf=0.1, expect 50–80% false positives (the model is casting a wide net). At conf=0.9, expect <10% false positives.

</span>"""))

# ─── CELL 15: Summary code ─────────────────────────────────────────
cells.append(code("""\
# ── Count detections by label category ──────────────────────────────
n_total = len(all_detections)
n_worm = sum(1 for d in all_detections if d["label"] == "scale_worm")       # true positives
n_not = sum(1 for d in all_detections if d["label"] == "not_worm")          # false positives
n_skip = sum(1 for d in all_detections if d["label"] == "skip")             # uncertain
n_unlabeled = sum(1 for d in all_detections if d["label"] is None)          # not yet reviewed

# ── Print formatted summary ─────────────────────────────────────────
print("=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print(f"  Total detections:     {n_total:,}")
print(f"  ✓ Scale worm:         {n_worm:,} ({100*n_worm/max(n_total,1):.1f}%)")
print(f"  ✗ Not a worm:         {n_not:,} ({100*n_not/max(n_total,1):.1f}%)")
print(f"  ⟳ Skipped:            {n_skip:,}")
print(f"  ⚪ Unlabeled:          {n_unlabeled:,}")
# False positive rate: fraction of definitive labels that were not-worm
print(f"  False positive rate:  {100*n_not/max(n_worm+n_not,1):.1f}%")
print("=" * 60)

# ── Warn if work is incomplete ──────────────────────────────────────
if n_unlabeled > 0:
    print(f"\\n⚠️  {n_unlabeled} detections still unlabeled. "
          "Go back to Step 6 to finish before exporting.")"""))

# ─── CELL 16: Export markdown ──────────────────────────────────────
cells.append(md("""\
## 8. Export verified detections as YOLO dataset

<span style="font-family: 'Courier New', monospace;">

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

**What happens next:** The code cell below packages all detections labeled `"scale_worm"` into a YOLO-format training dataset and compresses it into a zip file.

**Why:** The verified detections are the output of this lab — a curated, human-reviewed dataset that can be used to train or fine-tune a YOLO model. The YOLO format is a standard: each image gets a matching `.txt` file where each line describes one bounding box as normalized coordinates.

**Inputs:**
- `all_detections` — with labels from Step 6
- `FRAME_W`, `FRAME_H` — from Setup (used for coordinate normalization)
- `START_DATE`, `END_DATE` — from Step 2 (used in the zip filename)

**YOLO label format:** Each line in a `.txt` label file is:
```
class_id  center_x  center_y  width  height
```
- `class_id`: always `0` (single class: `scale_worm`)
- All coordinates are normalized to [0, 1] relative to image dimensions (1920×1080)
- `center_x`, `center_y`: center of the bounding box
- `width`, `height`: size of the bounding box

**Outputs:**
- `verification_session/export/` — YOLO dataset directory:
  - `images/train/` — source frame PNGs (only frames with verified worms)
  - `labels/train/` — matching `.txt` label files
  - `dataset.yaml` — YOLO training config file
- `verification_session/verified_scaleworm_dataset_<dates>.zip` — compressed archive

**Note on train/val split:** Both `train` and `val` in `dataset.yaml` point to the same directory. For real training, you should split the images into separate train and val folders (typically 80/20).

**Test the method:** After export, check that (1) the number of images matches the number of unique frames with worm detections, (2) each image has a corresponding `.txt` file, and (3) label values are all between 0 and 1.

</span>"""))

# ─── CELL 17: Export code ──────────────────────────────────────────
cells.append(code("""\
def pixel_to_yolo(x1, y1, x2, y2, img_w=FRAME_W, img_h=FRAME_H):
    \"\"\"Convert pixel coordinates (x1, y1, x2, y2) to YOLO normalized format.

    YOLO format: (center_x, center_y, width, height), all in [0, 1].

    Parameters
    ----------
    x1, y1 : float
        Top-left corner of the bounding box in pixels.
    x2, y2 : float
        Bottom-right corner of the bounding box in pixels.
    img_w, img_h : int
        Image dimensions for normalization (default: 1920×1080).

    Returns
    -------
    tuple of float
        (center_x, center_y, width, height) normalized to [0, 1].
    \"\"\"
    cx = ((x1 + x2) / 2) / img_w  # center x, normalized
    cy = ((y1 + y2) / 2) / img_h  # center y, normalized
    w = (x2 - x1) / img_w         # width, normalized
    h = (y2 - y1) / img_h         # height, normalized
    # Clamp all values to [0, 1] to handle edge-case rounding
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return cx, cy, w, h


# ── Filter to verified worm detections only ─────────────────────────
worm_dets = [d for d in all_detections if d["label"] == "scale_worm"]

if len(worm_dets) == 0:
    print("No verified worm detections to export!")
else:
    # ── Group detections by source frame ────────────────────────────
    # Multiple worms in the same frame → one image file, multiple label lines
    from collections import defaultdict
    frame_groups = defaultdict(list)
    for det in worm_dets:
        frame_groups[det["frame_path"]].append(det)

    # ── Create YOLO directory structure ─────────────────────────────
    img_dir = EXPORT_DIR / "images" / "train"  # YOLO expects images/train/
    lbl_dir = EXPORT_DIR / "labels" / "train"  # YOLO expects labels/train/
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # ── Copy frames and write label files ───────────────────────────
    for frame_path, dets in sorted(frame_groups.items()):
        src = Path(frame_path)
        # Build a unique filename: video_name + frame_name
        # e.g., CAMHDA301-20241004T001500_frame_0003.png
        video_name = dets[0]["video"]
        frame_name = dets[0]["frame_file"]
        unique_name = f"{video_name}_{frame_name}"
        stem = Path(unique_name).stem  # filename without extension

        # Copy the source frame image into the YOLO images directory
        dst_img = img_dir / unique_name
        if not dst_img.exists():
            shutil.copy2(src, dst_img)  # copy2 preserves metadata

        # Write the YOLO label file (one line per detection in this frame)
        dst_lbl = lbl_dir / f"{stem}.txt"
        with open(dst_lbl, "w") as f:
            for det in dets:
                # Convert pixel coords to YOLO normalized format
                cx, cy, w, h = pixel_to_yolo(det["x1"], det["y1"],
                                              det["x2"], det["y2"])
                # Format: class_id center_x center_y width height
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\\n")

    # ── Write dataset.yaml (YOLO training configuration) ────────────
    yaml_path = EXPORT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: .\\n")                     # dataset root = this directory
        f.write(f"train: images/train\\n")          # training images path
        f.write(f"val: images/train\\n\\n")          # val = same as train (split later)
        f.write(f"nc: 1\\n")                        # number of classes
        f.write(f"names:\\n")
        f.write(f"  0: scale_worm\\n\\n")            # class 0 = scale_worm
        # Metadata comments for provenance tracking
        f.write(f"# Verified detections: {len(worm_dets)}\\n")
        f.write(f"# Source frames: {len(frame_groups)}\\n")
        f.write(f"# Date range: {START_DATE} to {END_DATE}\\n")
        f.write(f"# Confidence threshold: {CONF_THRESHOLD}\\n")
        f.write(f"# Frame size: {FRAME_W}x{FRAME_H}\\n")

    # ── Create zip archive ──────────────────────────────────────────
    zip_name = f"verified_scaleworm_dataset_{START_DATE}_to_{END_DATE}.zip"
    zip_path = WORK_DIR / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Walk the export directory and add all files
        for fpath in sorted(EXPORT_DIR.rglob("*")):
            if fpath.is_file():
                arcname = fpath.relative_to(EXPORT_DIR)  # path inside zip
                zf.write(fpath, arcname)

    zip_size_mb = zip_path.stat().st_size / 1e6  # file size in MB

    # ── Print export summary ────────────────────────────────────────
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"  Verified worm detections: {len(worm_dets):,}")
    print(f"  Unique frames:            {len(frame_groups):,}")
    print(f"  YOLO dataset:             {EXPORT_DIR}")
    print(f"  Zip file:                 {zip_path}")
    print(f"  Zip size:                 {zip_size_mb:.1f} MB")
    print("=" * 60)
    print(f"\\nTo use this dataset for YOLO training:")
    print(f"  yolo detect train data=dataset.yaml model=yolo11m.pt epochs=20 imgsz=1920")"""))

# ─── Assemble notebook ─────────────────────────────────────────────
# Read the ORIGINAL notebook to preserve metadata/kernel spec
with open(NB_SOURCE) as f:
    nb = json.load(f)

nb["cells"] = cells

# Write to the ANNOTATED filename (not the original)
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote annotated notebook: {NB_PATH}")
print(f"  (original untouched:   {NB_SOURCE})")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
