#!/usr/bin/env python3
"""Generate the student scaleworm pipeline notebook (20_student_scaleworm_pipeline.ipynb).

This script builds the notebook programmatically because it's large and needs
precise cell structure. Run with: uv run python scripts/build_student_notebook.py
"""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "notebooks" / "20_student_scaleworm_pipeline.ipynb"


def md(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
        "outputs": [],
        "execution_count": None,
    }


def build_notebook():
    cells = []

    # =========================================================================
    # HEADER + OVERVIEW
    # =========================================================================

    cells.append(md("""
# Scale Worm Detection Pipeline — Student Analyst Notebook

**Instrument**: OOI Cabled Array HD Camera (CAMHDA301) at Axial Seamount
**Location**: Mushroom hydrothermal vent, ASHES vent field (45.93°N, 130.01°W, ~1540 m depth)
**Detector**: Faster R-CNN v2 (ResNet50-FPN), fine-tuned on 2023 CAMHD scale worm annotations

This notebook guides you through a complete **human-in-the-loop object detection** workflow
for counting scale worms (*Lepidonotopodium* sp.) using video from the OOI CAMHD camera.

**What you will do:**
1. **Phase 1** — Run a pretrained detector on your assigned month of video (~30 min)
2. **Phase 2** — Review model detections and correct mistakes (~2-3 hours)
3. **Phase 3** — Retrain the model with your corrections and compare results (~1 hour)
4. **Phase 4** — Export final counts and a quality report card

**What are scale worms?** Polynoid scale worms are among the most abundant macrofauna at
hydrothermal vents. They are typically 2-5 cm long, pale/translucent, and congregate
on sulfide structures near vent orifices. In CAMHD imagery they appear as small
elongated bright objects against the dark vent substrate.

**What is CAMHD?** The Cabled Array HD Camera (CAMHD) at Axial Seamount records
8 videos per day at fixed times. Each video pans through multiple scenes; we analyze
**Scene 1** (the Mushroom vent zoom view, 305-320 seconds into each video).

**Important:** Run cells in order from top to bottom. If you need to restart, you can
re-run any cell safely — the notebook checks for existing outputs and skips completed work.

---
"""))

    # =========================================================================
    # GIT PRIMER
    # =========================================================================

    cells.append(md("""
## Git Quick Start — Working on Your Own Copy

You cloned this repository to get the notebook and environment files. Now you need to
set up **your own remote** so that your changes (corrections, models, results) are saved
to your own GitHub account — not pushed back to the original class repo.

### Step-by-step (run these in a terminal, not in this notebook):

```bash
# 1. You already cloned the repo (if not, do this first):
git clone https://github.com/daxsoule/scaleworm-student-lab.git
cd scaleworm-student-lab

# 2. Create your own repository on GitHub:
#    Go to github.com → "+" (top right) → "New repository"
#    Name it something like "scaleworm-jsmith" (use your name)
#    Do NOT initialize with README (you already have one)

# 3. Rename the original remote to "upstream" (read-only reference)
#    and add YOUR repo as "origin":
git remote rename origin upstream
git remote add origin https://github.com/YOUR_USERNAME/scaleworm-jsmith.git

# 4. Push to YOUR repo:
git push -u origin main

# 5. From now on, "git push" sends to YOUR repo.
#    To get updates from the class repo later:
git pull upstream main
```

### Key git commands you'll use:

| Command | What it does |
|---------|-------------|
| `git status` | See what files you've changed |
| `git add <file>` | Stage a file for commit |
| `git commit -m "message"` | Save a snapshot of your changes |
| `git push` | Upload your commits to GitHub |
| `git pull upstream main` | Get updates from the class repo |

### Tips:
- **Commit early and often** — each time you complete a phase or have results worth saving
- Write commit messages that describe *what you found*, not just *what you did*
  (e.g., "Round 1: 12% FP rate, mean count 8.3" is better than "ran notebook")
- The `.gitignore` excludes large files (models, videos, images) from git — this is intentional
- Your corrections JSON files and counts parquet files ARE tracked, so commit those

---
"""))

    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================

    cells.append(md("""
## Environment Setup

Before running this notebook, you need the correct Python environment with PyTorch,
torchvision, and GPU support. There are two ways to set this up:

### Option A: Conda (recommended if starting fresh)

```bash
# From the scaleworm-student-lab directory:
conda env create -f environment.yml
conda activate scaleworm
python -m ipykernel install --user --name scaleworm --display-name "Python (scaleworm)"
```

Then select the **"Python (scaleworm)"** kernel in JupyterHub (top right of this notebook).

### Option B: Already on JupyterHub with GPU

If your JupyterHub already has PyTorch and torchvision installed (as ours does),
you can skip the conda step. Just make sure you're using a kernel with GPU access.

The cell below checks that everything is in place. **All three checks must pass**
before you continue.
"""))

    cells.append(code("""
# ── Environment Check ──
import subprocess, sys, shutil

# GPU
import torch
gpu_ok = torch.cuda.is_available()
print(f"PyTorch {torch.__version__}")
print(f"GPU available: {gpu_ok}")
if gpu_ok:
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  Memory: {mem:.1f} GB")
else:
    print("  WARNING: No GPU detected. Training will be very slow.")

# ffmpeg
ffmpeg_ok = shutil.which("ffmpeg") is not None
print(f"ffmpeg available: {ffmpeg_ok}")
if not ffmpeg_ok:
    print("  ERROR: ffmpeg is required. Contact your supervisor.")

# Video data
from pathlib import Path
video_root = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
video_ok = video_root.exists()
print(f"Video archive: {'FOUND' if video_ok else 'NOT FOUND'}")
if not video_ok:
    print(f"  ERROR: Expected video data at {video_root}")

if gpu_ok and ffmpeg_ok and video_ok:
    print("\\nAll checks passed. Ready to proceed.")
else:
    print("\\nSome checks failed. Fix the issues above before continuing.")
"""))

    cells.append(code("""
# ── Imports ──
import csv
import json
import random
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, clear_output

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.dpi"] = 120
"""))

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    cells.append(md("""
## Configuration

**Edit the cell below** with your name and assigned month.

There are **three variables** you need to change. Everything else in this notebook
runs automatically based on these values.

| Variable | What to enter | Example |
|----------|--------------|---------|
| `STUDENT_NAME` | Your name, lowercase, no spaces | `"jsmith"` |
| `ASSIGNED_MONTH` | Your month in YYYY/MM format | `"2024/09"` |
| `CURRENT_ROUND` | Which improvement round (start at 1) | `1` |

### Student Assignments

| Student | Primary Month | Secondary Month |
|---------|--------------|-----------------|
| A | 2024/09 | 2024/10 |
| B | 2024/10 | 2024/11 |
| C | 2024/11 | 2024/12 |
| D | 2024/12 | 2024/09 |

**How CURRENT_ROUND works:** You start at round 1. After completing Phases 1-3, the
convergence check (end of Phase 3) will tell you whether to proceed to Phase 4 or to
increment `CURRENT_ROUND` to 2 and repeat Phases 2-3 with your improved model.
Most students need 2-3 rounds to converge.
"""))

    cells.append(code("""
# ╔══════════════════════════════════════════════╗
# ║   EDIT THESE THREE VALUES                    ║
# ╚══════════════════════════════════════════════╝

STUDENT_NAME = "your_name"       # e.g., "jsmith" — lowercase, no spaces
ASSIGNED_MONTH = "2024/10"       # e.g., "2024/09" — YYYY/MM format
CURRENT_ROUND = 1                # Start at 1. Increment for each improvement round.

# ──────────────────────────────────────────────
assert STUDENT_NAME != "your_name", "Please enter your name above!"
assert re.match(r"\\d{4}/\\d{2}$", ASSIGNED_MONTH), "Month must be YYYY/MM format"
print(f"Student: {STUDENT_NAME}")
print(f"Month:   {ASSIGNED_MONTH}")
print(f"Round:   {CURRENT_ROUND}")
"""))

    # =========================================================================
    # STARTER PACKAGE
    # =========================================================================

    cells.append(md("""
## Starter Package Setup

Before running this notebook for the first time, you need the **starter package** — a
3 GB ZIP file containing the pretrained model and 2023 training data.

### What to do:

1. **Download** `scaleworm_starter_package.zip` from the shared drive
2. **Upload** it to your JupyterHub home directory (`/home/jovyan/`)
3. **Run the cell below** to unzip it (this only needs to happen once)

### What's inside the starter package:

| Item | Description | Size |
|------|-------------|------|
| `annotations/*.zip` | 2023 video clips + VIAME CSV annotations (4 months) | ~2.5 GB |
| `model/best_model_v3.pth` | Pretrained Faster R-CNN v2 detector | ~160 MB |
| `baseline/scaleworm_counts.parquet` | 2023 ground truth counts (for comparison) | ~50 KB |
| `baseline/annotations_scene1/*.csv` | Parsed scene 1 annotations | ~2 MB |
| `labels.txt` | Class names (`scale_worm`) | 11 bytes |

**Why do we need 2023 training data?** When you retrain the model in Phase 3, your
corrections are combined with the original 2023 annotations. This prevents the model
from "forgetting" what it learned during initial training — a common problem in
machine learning called **catastrophic forgetting**.
"""))

    cells.append(code("""
# ── Unzip starter package (run once) ──
import zipfile

STARTER_ZIP = Path.home() / "scaleworm_starter_package.zip"
STARTER_DIR = Path.home() / "scaleworm_starter"

if STARTER_DIR.exists():
    print(f"Starter directory already exists: {STARTER_DIR}")
    print("  Skipping unzip.")
elif STARTER_ZIP.exists():
    print(f"Unzipping {STARTER_ZIP} ...")
    with zipfile.ZipFile(STARTER_ZIP) as zf:
        zf.extractall(Path.home())
    print(f"Done. Contents at {STARTER_DIR}")
else:
    print(f"ERROR: Upload scaleworm_starter_package.zip to {Path.home()}")
    print("  Then re-run this cell.")
"""))

    # =========================================================================
    # BUILD TRAINING DATA
    # =========================================================================

    cells.append(md("""
## Build Training Data from Starter Package

This cell extracts video frames from the 2023 annotation clips and creates the
train/val split used for model training in Phase 3.

**What it does:**
1. Unzips the 4 monthly annotation archives (March, April, June, August 2023)
2. Extracts frames from each scene 1 clip at 10 fps using ffmpeg
3. Splits the data 80/20 (stratified by month) into train and validation sets
4. Rewrites VIAME CSV annotations to reference extracted frame filenames

**This takes ~10 minutes** on first run (extracting frames from ~150 video clips).
Once done, the training data is saved and this cell will skip on subsequent runs.

**You do not need to modify anything in this cell.**
"""))

    cells.append(code("""
# ── Build training data from starter package (run once, ~10 min) ──
# This extracts video frames from the 2023 annotation clips and creates
# the train/val split used for model training.

STARTER_DIR = Path.home() / "scaleworm_starter"
TRAINING_DIR = Path.home() / "scaleworm_training"

if (TRAINING_DIR / "train").exists() and (TRAINING_DIR / "val").exists():
    n_train = len(list((TRAINING_DIR / "train").iterdir()))
    n_val = len(list((TRAINING_DIR / "val").iterdir()))
    print(f"Training data already exists: {n_train} train, {n_val} val sequences")
    print("  Skipping extraction.")
else:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    annotation_dir = STARTER_DIR / "annotations"
    months = ["March", "April", "June", "August"]

    # Unzip annotation archives and collect scene1 clips
    all_clips = []  # (month_name, clip_path, csv_path)
    tmp_anno = TRAINING_DIR / "_tmp_annotations"
    tmp_anno.mkdir(exist_ok=True)

    for month in months:
        zip_path = annotation_dir / f"{month}.zip"
        if not zip_path.exists():
            print(f"  WARNING: {zip_path} not found, skipping")
            continue
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_anno)
        month_dir = tmp_anno / month
        if not month_dir.exists():
            continue
        for csv_file in sorted(month_dir.glob("*_scene1.csv")):
            mp4_file = csv_file.with_suffix(".mp4")
            if mp4_file.exists():
                all_clips.append((month, mp4_file, csv_file))

    print(f"Found {len(all_clips)} scene 1 clips across {len(months)} months")

    # Stratified 80/20 split by month
    random.seed(42)
    train_clips, val_clips = [], []
    for month in months:
        month_clips = [c for c in all_clips if c[0] == month]
        random.shuffle(month_clips)
        split_idx = max(1, int(len(month_clips) * 0.8))
        train_clips.extend(month_clips[:split_idx])
        val_clips.extend(month_clips[split_idx:])

    print(f"Split: {len(train_clips)} train, {len(val_clips)} val")

    def extract_sequence(clip_info, output_root):
        month, mp4_path, csv_path = clip_info
        # Derive sequence name from filename
        stem = mp4_path.stem.replace("_scene1", "")
        seq_dir = output_root / stem
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames at 10fps
        pattern = str(seq_dir / "frame_%04d.png")
        subprocess.run(
            ["ffmpeg", "-i", str(mp4_path), "-vf", "fps=10", "-q:v", "2", pattern, "-y"],
            capture_output=True,
        )
        n_frames = len(list(seq_dir.glob("frame_*.png")))

        # Rewrite VIAME CSV: replace timestamp column with frame filename
        gt_out = seq_dir / "groundtruth.csv"
        with open(csv_path) as f_in, open(gt_out, "w") as f_out:
            for line in f_in:
                if line.startswith("#"):
                    f_out.write(line)
                    continue
                parts = line.strip().split(",")
                if len(parts) < 10:
                    continue
                frame_idx = int(parts[2])
                if frame_idx < 1 or frame_idx > n_frames:
                    continue
                parts[1] = f"frame_{frame_idx:04d}.png"
                f_out.write(",".join(parts) + "\\n")

        return n_frames

    # Extract all sequences
    for split_name, clips, output_root in [
        ("train", train_clips, TRAINING_DIR / "train"),
        ("val", val_clips, TRAINING_DIR / "val"),
    ]:
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"\\nExtracting {split_name} set ({len(clips)} sequences)...")
        for i, clip_info in enumerate(clips):
            n = extract_sequence(clip_info, output_root)
            if (i + 1) % 5 == 0 or (i + 1) == len(clips):
                print(f"  {i+1}/{len(clips)} done")

    # Write labels.txt
    (TRAINING_DIR / "labels.txt").write_text("scale_worm\\n")

    # Clean up temp
    import shutil
    shutil.rmtree(tmp_anno, ignore_errors=True)

    n_train = len(list((TRAINING_DIR / "train").iterdir()))
    n_val = len(list((TRAINING_DIR / "val").iterdir()))
    print(f"\\nTraining data ready: {n_train} train, {n_val} val sequences")
    print(f"  Location: {TRAINING_DIR}")
"""))

    # =========================================================================
    # PATH SETUP
    # =========================================================================

    cells.append(md("""
## Directory Structure

The cell below sets up your working directories. All your outputs are organized by
student name, month, and round number:

```
~/scaleworm_students/
    jsmith/
        2024_09/
            scene1_clips/          ← extracted 15-second video clips
            round_1/
                inference/         ← model detection CSVs
                labeling/          ← sampled frames + your corrections
                model/             ← retrained model weights
                counts.parquet     ← scale worm counts (before retraining)
                counts_new.parquet ← counts (after retraining)
                comparison.png     ← before/after figure
            round_2/
                ...
            final/                 ← your deliverables
```

**You do not need to modify anything in this cell.**
"""))

    cells.append(code("""
# ── Path setup and shared constants ──
STARTER_DIR = Path.home() / "scaleworm_starter"
TRAINING_DIR = Path.home() / "scaleworm_training"
VIDEO_ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
BASELINE_COUNTS = STARTER_DIR / "baseline" / "scaleworm_counts.parquet"
MODEL_V3_PATH = STARTER_DIR / "model" / "best_model_v3.pth"

# Student output directories
year, month = ASSIGNED_MONTH.split("/")
month_str = f"{year}_{month}"
STUDENT_DIR = Path.home() / "scaleworm_students" / STUDENT_NAME / month_str
ROUND_DIR = STUDENT_DIR / f"round_{CURRENT_ROUND}"
CLIPS_DIR = STUDENT_DIR / "scene1_clips"
INFERENCE_DIR = ROUND_DIR / "inference"
LABELING_DIR = ROUND_DIR / "labeling"
MODEL_OUT_DIR = ROUND_DIR / "model"

for d in [CLIPS_DIR, INFERENCE_DIR, LABELING_DIR, MODEL_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Constants
SCENE1_START_SEC = 305
SCENE1_DURATION_SEC = 15
STANDARD_TIMES = {"T001500", "T031500", "T061500", "T091500",
                  "T121500", "T151500", "T181500", "T211500"}
NUM_CLASSES = 2
EXTRACT_FPS = 10
CONF_THRESHOLD = 0.5
MAX_BBOX_DIM = 300
SEED = 42

print(f"Student directory: {STUDENT_DIR}")
print(f"Round directory:   {ROUND_DIR}")
print(f"Video source:      {VIDEO_ROOT / year / month}")
"""))

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================

    cells.append(md("""
## Utility Functions

The cell below defines helper functions used throughout this notebook. You don't need
to understand every line, but here's what each function does:

| Function | Purpose |
|----------|---------|
| `find_standard_cadence_videos()` | Finds the 8 daily videos, skipping high-cadence experiment days |
| `extract_scene1_clip()` | Cuts a 15-second Scene 1 clip from a full CAMHD video |
| `load_detector()` | Loads the Faster R-CNN model from a checkpoint file |
| `extract_frames_from_clip()` | Pulls individual frames from an MP4 clip |
| `run_inference_on_frames()` | Runs the detector on a batch of frames |
| `preds_to_viame_csv()` | Saves predictions in VIAME CSV format |
| `count_from_csv()` | Counts worms: max concurrent detections in any frame |
| `extract_timestamp()` | Parses the UTC timestamp from CAMHD filenames |
| `draw_detections()` | Draws numbered, color-coded bounding boxes on a frame |

**You do not need to modify anything in this cell.** Just run it once.
"""))

    cells.append(code("""
# ── Utility Functions ──
# These are used throughout the notebook. Run this cell once.

def find_standard_cadence_videos(video_root, year, month):
    \"\"\"Find standard 3-hour cadence videos for a given month.
    Automatically detects and skips high-cadence days (>8 recordings/day).\"\"\"
    month_dir = video_root / year / month
    if not month_dir.exists():
        print(f"ERROR: {month_dir} does not exist")
        return []

    # Collect all MP4s by day
    by_day = defaultdict(list)
    for day_dir in sorted(month_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        for mp4 in sorted(day_dir.glob("CAMHDA301-*.mp4")):
            by_day[day_dir.name].append(mp4)

    videos = []
    skipped_days = []
    for day, day_videos in sorted(by_day.items()):
        # Check if this is a high-cadence day
        standard = [v for v in day_videos
                    if any(v.name.endswith(f"{t}.mp4") or f"{t}" in v.name
                           for t in STANDARD_TIMES)]

        # Heuristic: if >8 standard-time videos exist, it's high cadence
        # Just keep the 8 standard ones
        times_seen = set()
        for v in day_videos:
            m = re.search(r"T(\\d{6})", v.name)
            if m:
                times_seen.add(m.group(1))

        if len(day_videos) > 10:  # high-cadence day
            skipped_days.append(day)
            # Still include the 8 standard times
            for v in day_videos:
                m = re.search(r"T(\\d{6})", v.name)
                if m and f"T{m.group(1)}" in STANDARD_TIMES:
                    videos.append(v)
        else:
            videos.extend(day_videos)

    if skipped_days:
        print(f"  High-cadence days detected (using standard times only): {skipped_days}")

    return sorted(videos)


def extract_scene1_clip(video_path, output_path):
    \"\"\"Extract scene 1 (305-320s) from a full CAMHD video.\"\"\"
    if output_path.exists() and output_path.stat().st_size > 10000:
        return True
    subprocess.run([
        "ffmpeg", "-ss", str(SCENE1_START_SEC),
        "-i", str(video_path),
        "-t", str(SCENE1_DURATION_SEC),
        "-c:v", "libx264", "-crf", "18",
        "-an", "-y", str(output_path),
    ], capture_output=True)
    return output_path.exists()


def load_detector(model_path, device):
    \"\"\"Load Faster R-CNN v2 detector.\"\"\"
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def extract_frames_from_clip(clip_path, fps=10):
    \"\"\"Extract frames from an MP4 clip, return as list of PIL images.\"\"\"
    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = f"{tmpdir}/frame_%04d.png"
        subprocess.run(
            ["ffmpeg", "-i", str(clip_path), "-vf", f"fps={fps}",
             "-q:v", "2", pattern, "-y"],
            capture_output=True,
        )
        frame_files = sorted(Path(tmpdir).glob("frame_*.png"))
        return [Image.open(f).convert("RGB") for f in frame_files]


def run_inference_on_frames(model, frames, device):
    \"\"\"Run detector on a list of PIL images. Returns list of prediction dicts.\"\"\"
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    all_preds = []
    for frame in frames:
        img_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)[0]
        all_preds.append({
            "boxes": pred["boxes"].cpu(),
            "labels": pred["labels"].cpu(),
            "scores": pred["scores"].cpu(),
        })
    return all_preds


def preds_to_viame_csv(preds, output_path, conf_threshold=0.0):
    \"\"\"Write predictions as VIAME CSV.\"\"\"
    with open(output_path, "w") as f:
        f.write("# track_id,image,frame_idx,x1,y1,x2,y2,conf,length,label,label_conf\\n")
        track_id = 0
        for frame_idx, pred in enumerate(preds):
            for j in range(len(pred["scores"])):
                score = pred["scores"][j].item()
                if score < conf_threshold:
                    continue
                if pred["labels"][j].item() != 1:
                    continue
                box = pred["boxes"][j]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                f.write(f"{track_id},frame_{frame_idx+1:04d}.png,{frame_idx},"
                        f"{x1},{y1},{x2},{y2},{score:.6f},-1,scale_worm,{score:.6f}\\n")
                track_id += 1


def parse_viame_csv(csv_path, conf_threshold=0.0):
    \"\"\"Parse VIAME CSV, return list of dicts.\"\"\"
    rows = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            try:
                score = float(parts[7])
            except (ValueError, IndexError):
                continue
            if score < conf_threshold or parts[9].strip() != "scale_worm":
                continue
            rows.append({
                "track_id": int(parts[0]),
                "frame": parts[1],
                "frame_idx": int(parts[2]),
                "x1": int(parts[3]), "y1": int(parts[4]),
                "x2": int(parts[5]), "y2": int(parts[6]),
                "confidence": score,
            })
    return rows


def count_from_csv(csv_path, conf_threshold=0.5):
    \"\"\"Count scale worms: max concurrent detections in any frame.\"\"\"
    dets = parse_viame_csv(csv_path, conf_threshold)
    if not dets:
        return 0
    by_frame = defaultdict(int)
    for d in dets:
        by_frame[d["frame_idx"]] += 1
    return max(by_frame.values())


def extract_timestamp(filename):
    \"\"\"Extract UTC timestamp from CAMHD filename.\"\"\"
    m = re.search(r"CAMHDA301-(\\d{4})(\\d{2})(\\d{2})T(\\d{2})(\\d{2})(\\d{2})", filename)
    if m:
        y, mo, d, h, mi, s = m.groups()
        return pd.Timestamp(f"{y}-{mo}-{d}T{h}:{mi}:{s}", tz="UTC")
    return None


def draw_detections(img, detections, conf_threshold=0.5):
    \"\"\"Draw numbered bounding boxes on a PIL image. Returns annotated copy.\"\"\"
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    filtered = [d for d in detections if d["confidence"] >= conf_threshold]
    for i, det in enumerate(filtered):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]
        # Color by confidence: green (high) → yellow (med) → red (low)
        if conf >= 0.9:
            color = "lime"
        elif conf >= 0.7:
            color = "yellow"
        else:
            color = "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 16), f"{i}:{conf:.2f}", fill=color, font=font)
    return img_copy


print("Utility functions loaded.")
"""))

    # =========================================================================
    # PHASE 1: EXTRACT AND RUN BASELINE
    # =========================================================================

    cells.append(md("""
---
# Phase 1: Extract Video and Run Baseline Detection

In this phase you will:
1. **Find videos** — locate all standard-cadence videos for your assigned month
2. **Extract clips** — cut the Scene 1 segment (305-320 seconds) from each video
3. **Run the detector** — process every clip through the pretrained model
4. **Count and visualize** — build a time series of scale worm counts

**What to expect:**
- Your month should have ~240 videos (8 per day × 30 days)
- Scene 1 extraction takes ~1-2 minutes per 100 videos
- Model inference takes ~10-20 minutes depending on GPU speed
- Typical scale worm counts: 5-20 per video (varies by month and vent activity)

**You do not need to modify any cells in this phase.** Just run them in order.
"""))

    cells.append(code("""
# ── Find videos for your month ──
year, month = ASSIGNED_MONTH.split("/")
videos = find_standard_cadence_videos(VIDEO_ROOT, year, month)
print(f"Found {len(videos)} standard-cadence videos for {ASSIGNED_MONTH}")

if len(videos) == 0:
    print("ERROR: No videos found. Check that ASSIGNED_MONTH is correct.")
else:
    dates = set()
    for v in videos:
        ts = extract_timestamp(v.name)
        if ts:
            dates.add(ts.date())
    print(f"  Date range: {min(dates)} to {max(dates)} ({len(dates)} days)")
    print(f"  ~{len(videos)/len(dates):.1f} videos per day")
"""))

    cells.append(code("""
# ── Extract Scene 1 clips ──
# This takes ~1-2 minutes per 100 videos.

n_extracted = 0
n_skipped = 0
for i, video_path in enumerate(videos):
    clip_name = video_path.stem + "_scene1.mp4"
    clip_path = CLIPS_DIR / clip_name
    if clip_path.exists() and clip_path.stat().st_size > 10000:
        n_skipped += 1
        continue
    extract_scene1_clip(video_path, clip_path)
    n_extracted += 1
    if (n_extracted) % 20 == 0:
        print(f"  Extracted {n_extracted} clips...")

clips = sorted([c for c in CLIPS_DIR.glob("*_scene1.mp4") if c.stat().st_size > 10000])
print(f"\\nScene 1 clips ready: {len(clips)} ({n_extracted} new, {n_skipped} existing)")
"""))

    cells.append(md("""
## Run Model Inference

The detector is a **Faster R-CNN v2** (ResNet50 backbone with Feature Pyramid Network),
pretrained on ImageNet/COCO and fine-tuned on 2023 CAMHD scale worm annotations.

**How it works:** The model processes each video frame independently and outputs
bounding boxes with confidence scores for each detected scale worm. We save all
detections (even low-confidence ones) so you can review them in Phase 2.

**Confidence scores** range from 0 to 1:
- **> 0.9** — the model is very confident this is a scale worm
- **0.7 - 0.9** — probably a scale worm, but worth checking
- **0.5 - 0.7** — uncertain — could be a worm, tube worm, or artifact
- **< 0.5** — probably not a scale worm (we still save these for analysis)
"""))

    cells.append(code("""
# ── Determine which model to use ──
if CURRENT_ROUND == 1:
    model_path = MODEL_V3_PATH
    print(f"Round 1: Using pretrained v3 model")
else:
    prev_model = STUDENT_DIR / f"round_{CURRENT_ROUND - 1}" / "model" / "best_model.pth"
    if prev_model.exists():
        model_path = prev_model
        print(f"Round {CURRENT_ROUND}: Using your model from round {CURRENT_ROUND - 1}")
    else:
        model_path = MODEL_V3_PATH
        print(f"Round {CURRENT_ROUND}: Previous model not found, using v3")

print(f"  Model: {model_path}")
assert model_path.exists(), f"Model not found at {model_path}"
"""))

    cells.append(code("""
# ── Run inference on all clips ──
# This takes ~10-20 minutes depending on GPU and number of clips.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_detector(model_path, device)

clips = sorted([c for c in CLIPS_DIR.glob("*_scene1.mp4") if c.stat().st_size > 10000])
n_processed = 0
n_skipped = 0
t0 = time.time()

for i, clip in enumerate(clips):
    out_csv = INFERENCE_DIR / (clip.stem + ".csv")
    if out_csv.exists():
        n_skipped += 1
        continue

    frames = extract_frames_from_clip(clip, fps=EXTRACT_FPS)
    if not frames:
        print(f"  WARNING: No frames from {clip.name}")
        continue

    preds = run_inference_on_frames(model, frames, device)
    preds_to_viame_csv(preds, out_csv, conf_threshold=0.0)
    n_processed += 1

    if (n_processed) % 20 == 0:
        elapsed = time.time() - t0
        rate = n_processed / elapsed * 60
        print(f"  {n_processed} processed ({rate:.0f}/min), {n_skipped} skipped")

# Free GPU memory
del model
torch.cuda.empty_cache()

total_csvs = len(list(INFERENCE_DIR.glob("*.csv")))
elapsed = time.time() - t0
print(f"\\nInference complete: {total_csvs} videos ({n_processed} new in {elapsed:.0f}s)")
"""))

    cells.append(code("""
# ── Count scale worms and build time series ──
csv_files = sorted(INFERENCE_DIR.glob("*.csv"))
rows = []
for csv_path in csv_files:
    count = count_from_csv(csv_path, CONF_THRESHOLD)
    ts = extract_timestamp(csv_path.name)
    rows.append({
        "filename": csv_path.stem,
        "timestamp": ts,
        "scaleworm_count": count,
        "method": "automated",
        "round": CURRENT_ROUND,
        "student": STUDENT_NAME,
    })

counts_df = pd.DataFrame(rows)
counts_df.to_parquet(ROUND_DIR / "counts.parquet", index=False)

print(f"Counted {len(counts_df)} videos")
print(f"  Mean:   {counts_df['scaleworm_count'].mean():.1f}")
print(f"  Median: {counts_df['scaleworm_count'].median():.0f}")
print(f"  Std:    {counts_df['scaleworm_count'].std():.1f}")
print(f"  Range:  {counts_df['scaleworm_count'].min()} -- {counts_df['scaleworm_count'].max()}")
"""))

    cells.append(md("""
## Phase 1 Results

The figures below show three views of your baseline detection results:

1. **Daily time series** (left) — the average scale worm count per day across your month.
   Look for day-to-day patterns and any suspicious gaps or outliers.
2. **Count distribution** (center) — a histogram of per-video counts. Is it bell-shaped
   or skewed? A long right tail suggests occasional high-activity events.
3. **Baseline comparison** (right) — your automated counts (orange) compared to the
   2023 manual counts by month (blue). The 2023 counts averaged 10-15 worms per video.

**What to look for:**
- If your automated counts are **much lower** than the 2023 baseline (< 5), the model
  is probably missing many worms — your corrections in Phase 2 will be especially valuable.
- If your counts are **much higher** (> 25), there may be many false positives to correct.
- Counts of 0 for some videos are normal — sometimes the camera view is occluded or
  the vent is less active.
"""))

    cells.append(code("""
# ── Phase 1 Figures ──
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Daily time series
counts_df["date"] = counts_df["timestamp"].dt.date
daily = counts_df.groupby("date")["scaleworm_count"].mean()
axes[0].plot(daily.index, daily.values, "o-", markersize=3, linewidth=1)
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Scale worms (daily mean)")
axes[0].set_title(f"Daily Counts -- {ASSIGNED_MONTH}")
axes[0].tick_params(axis="x", rotation=45)

# Count distribution
axes[1].hist(counts_df["scaleworm_count"], bins=range(0, counts_df["scaleworm_count"].max() + 2),
             edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Scale worm count")
axes[1].set_ylabel("Number of videos")
axes[1].set_title("Count Distribution")
axes[1].axvline(counts_df["scaleworm_count"].mean(), color="red", linestyle="--",
                label=f"Mean: {counts_df['scaleworm_count'].mean():.1f}")
axes[1].legend()

# Compare with 2023 baseline
if BASELINE_COUNTS.exists():
    baseline = pd.read_parquet(BASELINE_COUNTS)
    baseline_monthly = baseline.groupby(
        baseline["video_timestamp"].dt.to_period("M")
    )["scaleworm_count"].agg(["mean", "std"]).reset_index()

    months_2023 = baseline_monthly["video_timestamp"].astype(str).values
    means_2023 = baseline_monthly["mean"].values

    x = np.arange(len(months_2023) + 1)
    labels = list(months_2023) + [ASSIGNED_MONTH]
    means = list(means_2023) + [counts_df["scaleworm_count"].mean()]
    colors = ["steelblue"] * len(months_2023) + ["darkorange"]

    axes[2].bar(x, means, color=colors, edgecolor="black", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    axes[2].set_ylabel("Mean count")
    axes[2].set_title("Your Month vs. 2023 Baseline")
else:
    axes[2].text(0.5, 0.5, "Baseline data\\nnot found", ha="center", va="center",
                 transform=axes[2].transAxes)

plt.tight_layout()
fig.savefig(ROUND_DIR / "phase1_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {ROUND_DIR / 'phase1_overview.png'}")
"""))

    cells.append(md("""
### Self-Evaluation Checkpoint

Before moving on, take a moment to assess your Phase 1 results:

- **Daily time series**: Do you see reasonable day-to-day variation? Any suspicious gaps or outliers?
- **Count distribution**: Is the distribution roughly bell-shaped or skewed?
- **Baseline comparison**: How does your month compare to the 2023 data?

If everything looks reasonable, proceed to Phase 2. If something looks wrong (e.g., all
counts are 0, or counts are in the hundreds), check that the starter package unzipped
correctly and that you're using the right month.

**Good time to commit your work:**
```bash
git add -A && git commit -m "Phase 1 complete: baseline counts for [your month]"
```
"""))

    # =========================================================================
    # PHASE 2: HUMAN IN THE LOOP
    # =========================================================================

    cells.append(md("""
---
# Phase 2: Review Model Detections

This is **the most important phase** of the entire workflow. Your careful corrections
are what make the model better. The quality of the science depends on this step.

**How it works:**
1. We sample ~100 frames, stratified across confidence levels to focus your effort
2. You view each frame with numbered, color-coded bounding boxes
3. For each detection, you decide: **correct** (scale worm) or **wrong** (not a worm)
4. You also note any **missed** worms the model didn't detect

### What to mark as a false positive (NOT a scale worm):

- **Tube worms** — elongated but attached to substrate, don't move between frames
- **Bacterial mats** — bright patches but diffuse, not discrete organisms
- **Shadows/artifacts** — edges, reflections, camera lens artifacts
- **Other fauna** — limpets, crabs, anemones (different shape and size)

### What IS a scale worm:

- Small (2-5 cm), elongated, pale/translucent body
- Often partially curled or moving
- Congregates near vent orifices on sulfide structures
- May be partially hidden under ledges or behind other organisms

### Color coding on the frames:

- **Green boxes** (confidence >= 0.9): Usually correct. Only mark as FP if clearly wrong.
- **Yellow boxes** (confidence 0.7-0.9): Review carefully — these are where your input matters most.
- **Red boxes** (confidence 0.5-0.7): Often wrong. Only keep if clearly a scale worm.
"""))

    cells.append(code("""
# ── Sample frames for review ──
# Stratified by confidence: 25 frames from each of 4 bins

random.seed(SEED + CURRENT_ROUND)

# Collect all detections across all inference CSVs
all_detections = []  # (csv_path, frame_idx, detections_on_frame)
for csv_path in sorted(INFERENCE_DIR.glob("*.csv")):
    dets = parse_viame_csv(csv_path, conf_threshold=0.3)
    by_frame = defaultdict(list)
    for d in dets:
        w, h = d["x2"] - d["x1"], d["y2"] - d["y1"]
        if w > MAX_BBOX_DIM or h > MAX_BBOX_DIM or w <= 0 or h <= 0:
            continue
        by_frame[d["frame_idx"]].append(d)
    for frame_idx, frame_dets in by_frame.items():
        max_conf = max(d["confidence"] for d in frame_dets)
        all_detections.append({
            "csv": csv_path.name,
            "clip": csv_path.stem,
            "frame_idx": frame_idx,
            "detections": frame_dets,
            "max_conf": max_conf,
            "n_dets": len(frame_dets),
        })

print(f"Total frames with detections: {len(all_detections)}")

# Stratified sample
CONF_BINS = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
N_PER_BIN = 25
sampled = []
for lo, hi in CONF_BINS:
    bin_frames = [f for f in all_detections if lo <= f["max_conf"] < hi]
    n = min(N_PER_BIN, len(bin_frames))
    sampled.extend(random.sample(bin_frames, n))
    print(f"  Confidence [{lo:.1f}, {hi:.1f}): {len(bin_frames)} available, sampled {n}")

random.shuffle(sampled)
print(f"\\nTotal frames to review: {len(sampled)}")
"""))

    cells.append(md("""
## Extract Frames for Labeling

The cell below extracts the sampled frames from the video clips and creates annotated
versions with numbered bounding boxes. This takes a few minutes.

**You do not need to modify anything in this cell.**
"""))

    cells.append(code("""
# ── Extract sampled frames from clips and create annotated versions ──
frames_dir = LABELING_DIR / "frames"
annotated_dir = LABELING_DIR / "frames_annotated"
frames_dir.mkdir(exist_ok=True)
annotated_dir.mkdir(exist_ok=True)

frame_index = []
for i, entry in enumerate(sampled):
    frame_id = f"frame_{i:04d}"
    clip_path = CLIPS_DIR / (entry["clip"] + ".mp4")
    if not clip_path.exists():
        # Try without _scene1 suffix duplication
        alt = entry["clip"].replace("_scene1", "") + "_scene1.mp4"
        clip_path = CLIPS_DIR / alt

    if not clip_path.exists():
        continue

    # Extract specific frame
    frame_file = frames_dir / f"{frame_id}.png"
    if not frame_file.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = f"{tmpdir}/frame_%04d.png"
            subprocess.run(
                ["ffmpeg", "-i", str(clip_path), "-vf", f"fps={EXTRACT_FPS}",
                 "-q:v", "2", pattern, "-y"],
                capture_output=True,
            )
            src = Path(tmpdir) / f"frame_{entry['frame_idx']+1:04d}.png"
            if src.exists():
                import shutil
                shutil.copy(src, frame_file)

    if not frame_file.exists():
        continue

    # Draw detections on frame
    img = Image.open(frame_file)
    annotated = draw_detections(img, entry["detections"], conf_threshold=0.3)
    annotated.save(annotated_dir / f"{frame_id}.png")

    frame_index.append({
        "id": frame_id,
        "file": f"{frame_id}.png",
        "clip": entry["clip"],
        "frame_idx": entry["frame_idx"],
        "detections": [
            {"idx": j, "bbox": [d["x1"], d["y1"], d["x2"], d["y2"]],
             "confidence": d["confidence"]}
            for j, d in enumerate(entry["detections"])
        ],
    })

    if (i + 1) % 25 == 0:
        print(f"  Extracted {i+1}/{len(sampled)} frames")

# Save frame index
with open(LABELING_DIR / "frame_index.json", "w") as f:
    json.dump(frame_index, f, indent=2)

print(f"\\nReady for labeling: {len(frame_index)} frames")
print(f"  Frames: {frames_dir}")
print(f"  Annotated: {annotated_dir}")
"""))

    cells.append(md("""
## Labeling Interface

Below you will review frames **in batches of 6**. For each frame:

1. **Look at the annotated image** — each detection has a numbered box with confidence score
2. **Identify false positives** — note the box numbers of things that are NOT scale worms
3. **Count missed worms** — estimate how many worms the model missed in each frame

### Workflow for each batch:

1. Run the `show_batch(...)` cell to display 6 frames
2. Examine each frame and its detection list printed below the images
3. Scroll down to the **"Enter your corrections"** cell
4. Edit the `batch_corrections` dictionary with your findings
5. Run the corrections cell to save
6. Change `BATCH_NUMBER` and repeat

### Tips:
- You don't need to correct every frame — if all detections look correct, skip it
- Focus your attention on **yellow** (medium confidence) boxes — these are where your
  corrections help the model the most
- When in doubt, label as `unsure` — it's better to skip uncertain cases than to
  introduce wrong labels
- It's normal to find 10-30% false positives in the initial model predictions
"""))

    cells.append(code("""
# ── Labeling: process frames in batches ──
# Load frame index
with open(LABELING_DIR / "frame_index.json") as f:
    frame_index = json.load(f)

# Load existing corrections (allows resuming)
corrections_path = LABELING_DIR / "corrections.json"
if corrections_path.exists():
    with open(corrections_path) as f:
        corrections = json.load(f)
else:
    corrections = {}

BATCH_SIZE_LABEL = 6  # frames per batch

def show_batch(start_idx):
    \"\"\"Display a batch of frames for labeling.\"\"\"
    batch = frame_index[start_idx:start_idx + BATCH_SIZE_LABEL]
    n = len(batch)
    if n == 0:
        print("No more frames to review.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, entry in enumerate(batch):
        img_path = LABELING_DIR / "frames_annotated" / entry["file"]
        if img_path.exists():
            img = Image.open(img_path)
            axes[i].imshow(img)
        axes[i].set_title(f"Frame {entry['id']} ({len(entry['detections'])} dets)", fontsize=10)
        axes[i].axis("off")

        # Print detection list below
        det_text = []
        for det in entry["detections"]:
            conf = det["confidence"]
            default = "worm" if conf >= 0.7 else "not_worm" if conf < 0.5 else "check"
            det_text.append(f"  #{det['idx']}: conf={conf:.2f} [{default}]")

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Print detection details for each frame
    print("\\nDetection details:")
    print("=" * 70)
    for entry in batch:
        print(f"\\n{entry['id']} (clip: {entry['clip']}, frame {entry['frame_idx']}):")
        for det in entry["detections"]:
            conf = det["confidence"]
            marker = "+" if conf >= 0.7 else "?" if conf >= 0.5 else "-"
            print(f"  {marker} Box #{det['idx']}: confidence {conf:.3f}")

# Show first batch
print("Showing first batch. Scroll down to enter corrections.\\n")
show_batch(0)
"""))

    cells.append(md("""
### Enter Corrections

For each frame you reviewed above, list the box numbers that are **false positives**
(not scale worms) and how many worms the model **missed** (false negatives).

**How to fill in the dictionary below:**

```python
batch_corrections = {
    "frame_0000": {"false_positives": [2, 4], "missed_count": 1},
    #              ^^^ box #2 and #4 are       ^^^ 1 worm the model
    #                  NOT scale worms              didn't detect
    "frame_0001": {"false_positives": [], "missed_count": 0},
    #              ^^^ all detections are correct, no missed worms
}
```

**Only include frames where you have corrections to make.** If all detections in a frame
look correct and no worms were missed, you can skip that frame entirely.

Edit the dictionary below, then **run the cell** to save your corrections.
"""))

    cells.append(code("""
# ── Enter your corrections here ──
# Format: "frame_id": {"false_positives": [box_numbers], "missed_count": N}
#
# Example:
#   "frame_0000": {"false_positives": [2, 4], "missed_count": 1},
#   "frame_0001": {"false_positives": [], "missed_count": 0},
#
# Only include frames where you have corrections to make.
# High-confidence (green) boxes default to correct.
# Low-confidence (red) boxes default to incorrect.

batch_corrections = {
    # "frame_0000": {"false_positives": [], "missed_count": 0},
}

# Save corrections
corrections.update(batch_corrections)
with open(corrections_path, "w") as f:
    json.dump(corrections, f, indent=2)
print(f"Saved {len(corrections)} frame corrections to {corrections_path}")
"""))

    cells.append(md("""
### View More Batches

Change `BATCH_NUMBER` below and re-run to see the next set of frames. Each batch shows
6 frames. With ~100 frames total, you'll have about 17 batches.

After viewing each batch, scroll back up to the corrections cell, add your findings,
and run it to save.

| Batch | Frames |
|-------|--------|
| 0 | 0-5 |
| 1 | 6-11 |
| 2 | 12-17 |
| ... | ... |
| 16 | 96-99 |
"""))

    cells.append(code("""
# ── Show next batch ──
# Change the start index below to see different batches.
# Batch 0: frames 0-5, Batch 1: frames 6-11, etc.

BATCH_NUMBER = 1  # Change this: 0, 1, 2, 3, ... up to len(frame_index)//6
show_batch(BATCH_NUMBER * BATCH_SIZE_LABEL)
"""))

    cells.append(md("""
### Labeling Summary

Once you've reviewed all batches and entered your corrections, run the cell below
to see a summary of your labeling work. This will also apply default labels:
- High-confidence detections (>= 0.7) you didn't correct are assumed correct
- Low-confidence detections (< 0.5) you didn't correct are assumed wrong

The summary will tell you your false positive and false negative rates. Typical
values for a first round:
- **FP rate 10-30%** — normal, the model makes mistakes
- **FP rate < 5%** — suspiciously low, you may be approving boxes too readily
- **FP rate > 40%** — unusually high, but possible if the model struggles with your month

**Good time to commit after this step:**
```bash
git add -A && git commit -m "Phase 2 round [N]: reviewed [X] frames, [Y]% FP rate"
```
"""))

    cells.append(code("""
# ── Labeling Summary ──
with open(corrections_path) as f:
    corrections = json.load(f)

# Apply default labels for uncorrected frames
all_labels = {}  # frame_id -> {det_idx -> label}
n_total_dets = 0
n_fp = 0
n_confirmed = 0
n_missed = 0

for entry in frame_index:
    fid = entry["id"]
    corr = corrections.get(fid, {})
    fp_set = set(corr.get("false_positives", []))
    n_missed += corr.get("missed_count", 0)

    frame_labels = {}
    for det in entry["detections"]:
        idx = det["idx"]
        if idx in fp_set:
            frame_labels[idx] = "not_worm"
            n_fp += 1
        elif det["confidence"] >= 0.7:
            frame_labels[idx] = "scale_worm"
            n_confirmed += 1
        elif det["confidence"] < 0.5:
            frame_labels[idx] = "not_worm"
            n_fp += 1
        else:
            frame_labels[idx] = "unsure"
        n_total_dets += 1

    all_labels[fid] = frame_labels

# Save full labels
with open(LABELING_DIR / "frame_labels.json", "w") as f:
    json.dump(all_labels, f, indent=2)

fp_rate = n_fp / max(n_total_dets, 1) * 100
fn_rate_approx = n_missed / max(n_confirmed + n_missed, 1) * 100

print("=== Labeling Summary ===")
print(f"Frames reviewed:        {len(frame_index)}")
print(f"Frames with corrections: {len(corrections)}")
print(f"Total detections:       {n_total_dets}")
print(f"  Confirmed worms:      {n_confirmed} ({n_confirmed/max(n_total_dets,1)*100:.0f}%)")
print(f"  False positives:      {n_fp} ({fp_rate:.0f}%)")
print(f"  Missed worms (FN):    {n_missed}")
print(f"\\nFP rate: {fp_rate:.1f}%")
print(f"Approx FN rate: {fn_rate_approx:.1f}%")

if fp_rate > 30:
    print("\\nYour FP rate is high (>30%). Are you being too strict?")
    print("  Scale worms can be hard to see -- when in doubt, label as 'unsure'.")
elif fp_rate < 5:
    print("\\nYour FP rate is very low (<5%). Are you reviewing carefully?")
    print("  Check low-confidence (red) boxes especially -- many should be false positives.")
else:
    print("\\nFP rate looks reasonable. Proceed to Phase 3.")
"""))

    # =========================================================================
    # PHASE 3: RETRAIN AND COMPARE
    # =========================================================================

    cells.append(md("""
---
# Phase 3: Retrain the Model with Your Corrections

Now the payoff — your corrections teach the model to do better. Here's what happens:

1. **Combine datasets** — your labeled frames are merged with the original 2023 training data
2. **Fine-tune the model** — the detector is retrained with a lower learning rate (0.0003)
   so it learns from your corrections without forgetting its original training
3. **Re-run inference** — the improved model processes all your clips again
4. **Compare** — side-by-side plots show how counts changed

**What to expect:**
- Training takes ~15-30 minutes on GPU (20 epochs with early stopping)
- The model should make fewer false positives and catch more missed worms
- Count changes are usually small (1-3 worms per video) but systematic

**You do not need to modify any cells in this phase.** Just run them in order.
"""))

    cells.append(code("""
# ── Prepare combined training dataset ──
from torch.utils.data import Dataset, DataLoader

class OriginalDataset(Dataset):
    \"\"\"2023 VIAME-format training data (frames + groundtruth.csv).\"\"\"
    def __init__(self, root_dir):
        self.samples = []
        seq_dirs = sorted(
            [d for d in Path(root_dir).iterdir()
             if d.is_dir() and d.name.startswith("CAMHDA301")]
        )
        for seq_dir in seq_dirs:
            gt_file = seq_dir / "groundtruth.csv"
            if not gt_file.exists():
                continue
            frame_annots = {}
            with open(gt_file) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split(",")
                    if len(parts) < 10:
                        continue
                    frame_name = parts[1]
                    x1, y1 = int(parts[3]), int(parts[4])
                    x2, y2 = int(parts[5]), int(parts[6])
                    if parts[9].strip() != "scale_worm" or x2 <= x1 or y2 <= y1:
                        continue
                    frame_annots.setdefault(frame_name, []).append(
                        {"box": [x1, y1, x2, y2], "label": 1}
                    )
            for frame_name, annots in frame_annots.items():
                img_path = seq_dir / frame_name
                if img_path.exists():
                    self.samples.append((str(img_path), annots))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, annots = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self._to_tensors(img, annots, idx)

    @staticmethod
    def _to_tensors(img, annots, idx):
        img = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(img)
        boxes = (torch.tensor([a["box"] for a in annots], dtype=torch.float32)
                 if annots else torch.zeros((0, 4), dtype=torch.float32))
        labels = (torch.tensor([a["label"] for a in annots], dtype=torch.int64)
                  if annots else torch.zeros(0, dtype=torch.int64))
        area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                if len(boxes) > 0 else torch.zeros(0, dtype=torch.float32))
        return img, {
            "boxes": boxes, "labels": labels, "area": area,
            "iscrowd": torch.zeros(len(annots), dtype=torch.int64),
            "image_id": torch.tensor(idx),
        }


class StudentCorrectionDataset(Dataset):
    \"\"\"Frames with student corrections (confirmed worms + hard negatives).\"\"\"
    def __init__(self, labeling_dir, frame_index_path):
        with open(frame_index_path) as f:
            frame_index = json.load(f)
        labels_path = labeling_dir / "frame_labels.json"
        with open(labels_path) as f:
            all_labels = json.load(f)

        self.samples = []
        for entry in frame_index:
            fid = entry["id"]
            labels = all_labels.get(fid, {})
            if not labels:
                continue

            img_path = labeling_dir / "frames" / entry["file"]
            if not img_path.exists():
                continue

            # Collect confirmed worm boxes
            worm_boxes = []
            has_not_worm = False
            for det in entry["detections"]:
                label = labels.get(str(det["idx"]), "unsure")
                if label == "scale_worm":
                    worm_boxes.append({"box": det["bbox"], "label": 1})
                elif label == "not_worm":
                    has_not_worm = True

            if worm_boxes:
                self.samples.append((str(img_path), worm_boxes))
            elif has_not_worm:
                self.samples.append((str(img_path), []))  # hard negative

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, annots = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return OriginalDataset._to_tensors(img, annots, idx + 100000)


# Build datasets
original_train = OriginalDataset(TRAINING_DIR / "train")
original_val = OriginalDataset(TRAINING_DIR / "val")
student_data = StudentCorrectionDataset(LABELING_DIR, LABELING_DIR / "frame_index.json")

from torch.utils.data import ConcatDataset
combined_train = ConcatDataset([original_train, student_data])

n_pos = sum(1 for s in student_data.samples if s[1])
n_neg = sum(1 for s in student_data.samples if not s[1])
print(f"Training data:")
print(f"  Original 2023:       {len(original_train)} frames")
print(f"  Your corrections:    {len(student_data)} frames ({n_pos} positive, {n_neg} negative)")
print(f"  Combined:            {len(combined_train)} frames")
print(f"  Validation:          {len(original_val)} frames")
"""))

    cells.append(md("""
## Train the Model

The cell below fine-tunes the detector using your combined dataset. Key settings:

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 0.0003 | Low LR preserves original training while learning corrections |
| Epochs | 20 (max) | With early stopping after 5 epochs of no improvement |
| Optimizer | SGD + cosine annealing | Standard for fine-tuning object detectors |
| Batch size | 4 | Fits in GPU memory alongside the model |

**What to watch:** The training loss should decrease steadily. The validation loss
should decrease initially then flatten. If validation loss starts increasing, the
model is overfitting and early stopping will kick in.

**This takes 15-30 minutes on GPU.** You can watch the progress or go get coffee.
"""))

    cells.append(code("""
# ── Train the model ──
# Fine-tune from previous model at lower learning rate.
# This takes ~15-30 minutes on GPU.

def collate_fn(batch):
    return tuple(zip(*batch))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model = fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.to(device)

# Training config
EPOCHS = 20
LR = 0.0003
PATIENCE = 5

train_loader = DataLoader(combined_train, batch_size=4, shuffle=True,
                          num_workers=2, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(original_val, batch_size=4, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Training loop
training_log = []
best_val_loss = float("inf")
patience_counter = 0
t0 = time.time()

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    n_batch = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += losses.item()
        n_batch += 1

    # Validate
    model.train()  # Faster RCNN needs train mode for loss computation
    val_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            val_loss += sum(l.item() for l in loss_dict.values())
            n_val += 1

    scheduler.step()
    avg_train = train_loss / max(n_batch, 1)
    avg_val = val_loss / max(n_val, 1)

    improved = ""
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_OUT_DIR / "best_model.pth")
        improved = " *"
    else:
        patience_counter += 1

    training_log.append({"epoch": epoch+1, "train_loss": round(avg_train, 4),
                         "val_loss": round(avg_val, 4)})
    print(f"  Epoch {epoch+1}/{EPOCHS}: train={avg_train:.4f} val={avg_val:.4f}{improved}")

    if patience_counter >= PATIENCE:
        print(f"\\n  Early stopping (no improvement for {PATIENCE} epochs)")
        break

elapsed = time.time() - t0
print(f"\\nTraining complete in {elapsed/60:.1f} minutes")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Model saved: {MODEL_OUT_DIR / 'best_model.pth'}")

# Save training log
with open(MODEL_OUT_DIR / "training_log.json", "w") as f:
    json.dump(training_log, f, indent=2)

del model
torch.cuda.empty_cache()
"""))

    cells.append(md("""
## Re-Run Inference and Compare

Now we run your improved model on the same clips and compare the results.

The comparison plot has 4 panels:
- **(a) Scatter plot** — old vs. new counts per video. Points on the diagonal = no change.
  Points below = model finds fewer worms (reduced FPs). Points above = model finds more.
- **(b) Delta histogram** — distribution of count changes. Centered at 0 = stable.
- **(c) Daily time series** — before and after overlaid. Should track closely.
- **(d) Training curve** — loss vs. epoch. Look for smooth convergence.
"""))

    cells.append(code("""
# ── Re-run inference with your model ──
new_model_path = MODEL_OUT_DIR / "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_detector(new_model_path, device)

# Create new inference directory for this round
new_inference_dir = ROUND_DIR / "inference_new"
new_inference_dir.mkdir(exist_ok=True)

clips = sorted([c for c in CLIPS_DIR.glob("*_scene1.mp4") if c.stat().st_size > 10000])
t0 = time.time()
for i, clip in enumerate(clips):
    out_csv = new_inference_dir / (clip.stem + ".csv")
    if out_csv.exists():
        continue
    frames = extract_frames_from_clip(clip, fps=EXTRACT_FPS)
    if not frames:
        continue
    preds = run_inference_on_frames(model, frames, device)
    preds_to_viame_csv(preds, out_csv, conf_threshold=0.0)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(clips)}")

del model
torch.cuda.empty_cache()
print(f"Re-inference complete in {(time.time()-t0)/60:.1f} min")

# Count with new model
new_rows = []
for csv_path in sorted(new_inference_dir.glob("*.csv")):
    count = count_from_csv(csv_path, CONF_THRESHOLD)
    ts = extract_timestamp(csv_path.name)
    new_rows.append({"filename": csv_path.stem, "timestamp": ts, "scaleworm_count": count})

new_counts = pd.DataFrame(new_rows)
new_counts.to_parquet(ROUND_DIR / "counts_new.parquet", index=False)
print(f"\\nNew counts -- Mean: {new_counts['scaleworm_count'].mean():.1f}, "
      f"Median: {new_counts['scaleworm_count'].median():.0f}")
"""))

    cells.append(code("""
# ── Comparison Plots ──
old_counts = pd.read_parquet(ROUND_DIR / "counts.parquet")
new_counts = pd.read_parquet(ROUND_DIR / "counts_new.parquet")

# Merge on filename
merged = old_counts[["filename", "scaleworm_count"]].merge(
    new_counts[["filename", "scaleworm_count"]],
    on="filename", suffixes=("_old", "_new"))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Scatter -- old vs new counts
axes[0, 0].scatter(merged["scaleworm_count_old"], merged["scaleworm_count_new"],
                   alpha=0.3, s=15)
max_val = max(merged["scaleworm_count_old"].max(), merged["scaleworm_count_new"].max())
axes[0, 0].plot([0, max_val], [0, max_val], "r--", linewidth=1)
axes[0, 0].set_xlabel("Before (this round)")
axes[0, 0].set_ylabel("After (retrained)")
axes[0, 0].set_title("(a) Count Comparison per Video")

# Panel B: Delta histogram
delta = merged["scaleworm_count_new"] - merged["scaleworm_count_old"]
axes[0, 1].hist(delta, bins=range(int(delta.min())-1, int(delta.max())+2),
                edgecolor="black", alpha=0.7)
axes[0, 1].axvline(0, color="red", linestyle="--")
axes[0, 1].set_xlabel("Count change (new - old)")
axes[0, 1].set_ylabel("Number of videos")
axes[0, 1].set_title(f"(b) Count Change (mean: {delta.mean():+.2f})")

# Panel C: Daily time series overlay
old_counts["date"] = pd.to_datetime(old_counts["timestamp"]).dt.date
new_counts["date"] = pd.to_datetime(new_counts["timestamp"]).dt.date
daily_old = old_counts.groupby("date")["scaleworm_count"].mean()
daily_new = new_counts.groupby("date")["scaleworm_count"].mean()
axes[1, 0].plot(daily_old.index, daily_old.values, "o-", markersize=3, label="Before", alpha=0.7)
axes[1, 0].plot(daily_new.index, daily_new.values, "s-", markersize=3, label="After", alpha=0.7)
axes[1, 0].legend()
axes[1, 0].set_xlabel("Date")
axes[1, 0].set_ylabel("Daily mean count")
axes[1, 0].set_title("(c) Daily Time Series")
axes[1, 0].tick_params(axis="x", rotation=45)

# Panel D: Training loss curve
with open(MODEL_OUT_DIR / "training_log.json") as f:
    log = json.load(f)
epochs = [e["epoch"] for e in log]
axes[1, 1].plot(epochs, [e["train_loss"] for e in log], "o-", label="Train")
axes[1, 1].plot(epochs, [e["val_loss"] for e in log], "s-", label="Val")
axes[1, 1].legend()
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].set_title("(d) Training Curve")

plt.suptitle(f"{STUDENT_NAME} -- Round {CURRENT_ROUND} Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(ROUND_DIR / "comparison.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

    cells.append(md("""
## Convergence Check

The cell below checks whether your model has converged — meaning your corrections
have been successfully incorporated and further rounds would not significantly change
the results.

### Convergence criteria:

| Criterion | Threshold | What it measures |
|-----------|-----------|-----------------|
| MAE between rounds | < 0.5 | Stability: counts aren't changing much between rounds |
| FP rate | < 15% | Precision: model doesn't produce too many false detections |
| FN rate | < 10% | Recall: model isn't missing too many worms |
| Count RMSD | < 5% | Overall count accuracy |
| Frames reviewed | >= 50 | Minimum labeling effort for reliable training |
| At least 2 rounds | >= 2 | Ensures the model was actually retrained and tested |

**If all criteria pass:** Proceed to Phase 4 to export your final results.

**If some criteria fail:** Go back to the **Configuration** cell at the top,
change `CURRENT_ROUND` to `CURRENT_ROUND + 1`, and re-run from Phase 2 onward.
Your improved model from this round will be used as the starting point.
"""))

    cells.append(code("""
# ── Convergence Check ──
mae_between = abs(merged["scaleworm_count_new"] - merged["scaleworm_count_old"]).mean()
rmsd = np.sqrt(((merged["scaleworm_count_new"] - merged["scaleworm_count_old"]) ** 2).mean())
mean_count = merged["scaleworm_count_new"].mean()
rmsd_pct = rmsd / max(mean_count, 1) * 100

# Reload labeling stats
with open(LABELING_DIR / "frame_labels.json") as f:
    all_labels = json.load(f)
total_dets = sum(len(v) for v in all_labels.values())
total_fp = sum(1 for v in all_labels.values() for label in v.values() if label == "not_worm")
total_fn = sum(corrections.get(fid, {}).get("missed_count", 0) for fid in corrections)
fp_rate = total_fp / max(total_dets, 1) * 100
fn_rate = total_fn / max(total_dets - total_fp + total_fn, 1) * 100
frames_reviewed = len(frame_index)

# Check criteria
checks = {
    "MAE between rounds < 0.5":    mae_between < 0.5,
    "FP rate < 15%":               fp_rate < 15,
    "FN rate < 10%":               fn_rate < 10,
    "Count RMSD < 5%":             rmsd_pct < 5,
    "Frames reviewed >= 50":       frames_reviewed >= 50,
    "At least 2 rounds":           CURRENT_ROUND >= 2,
}

print("=" * 60)
print(f"  CONVERGENCE CHECK -- Round {CURRENT_ROUND}")
print("=" * 60)
for criterion, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {status}  {criterion}")

print(f"\\n  MAE between rounds:  {mae_between:.2f}")
print(f"  FP rate:             {fp_rate:.1f}%")
print(f"  FN rate:             {fn_rate:.1f}%")
print(f"  Count RMSD:          {rmsd_pct:.1f}%")
print(f"  Frames reviewed:     {frames_reviewed}")
print(f"  Current round:       {CURRENT_ROUND}")

all_passed = all(checks.values())
print("\\n" + "=" * 60)
if all_passed:
    print("  ALL CRITERIA MET -- Proceed to Phase 4 (Export)")
else:
    print("  NOT CONVERGED -- Increment CURRENT_ROUND and re-run from Phase 2")
    print("  (Change CURRENT_ROUND at the top of the notebook)")
print("=" * 60)

# Save stats
stats = {
    "round": CURRENT_ROUND, "student": STUDENT_NAME, "month": ASSIGNED_MONTH,
    "mae_between_rounds": round(mae_between, 3),
    "fp_rate": round(fp_rate, 1), "fn_rate": round(fn_rate, 1),
    "rmsd_pct": round(rmsd_pct, 1), "frames_reviewed": frames_reviewed,
    "mean_count_old": round(old_counts["scaleworm_count"].mean(), 1),
    "mean_count_new": round(new_counts["scaleworm_count"].mean(), 1),
    "converged": all_passed,
}
with open(ROUND_DIR / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)
"""))

    # =========================================================================
    # PHASE 4: EXPORT AND REPORT CARD
    # =========================================================================

    cells.append(md("""
---
# Phase 4: Export Final Results

Run this phase **only when all convergence criteria pass.** This creates your final
deliverables:

| File | Description |
|------|-------------|
| `final/counts.parquet` | Your best scale worm counts for every video |
| `final/model.pth` | Your fine-tuned detector weights |
| `final/report.json` | Machine-readable quality report card |
| `final/report_card.png` | Visual summary of your multi-round improvement |

**After running Phase 4, commit and push your work:**

```bash
git add -A
git commit -m "Final export: [your month], [N] rounds, mean count [X]"
git push
```

Then notify your supervisor that your results are ready.
"""))

    cells.append(code("""
# ── Export final results ──
final_dir = STUDENT_DIR / "final"
final_dir.mkdir(exist_ok=True)

# Determine which counts to use (new if available, else original)
final_counts_path = ROUND_DIR / "counts_new.parquet"
if not final_counts_path.exists():
    final_counts_path = ROUND_DIR / "counts.parquet"

import shutil
shutil.copy(final_counts_path, final_dir / "counts.parquet")
shutil.copy(MODEL_OUT_DIR / "best_model.pth", final_dir / "model.pth")

# Collect all round stats
all_stats = []
for rd in sorted(STUDENT_DIR.glob("round_*")):
    stats_file = rd / "stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            all_stats.append(json.load(f))

# Build report card
final_counts = pd.read_parquet(final_dir / "counts.parquet")
report = {
    "student": STUDENT_NAME,
    "month": ASSIGNED_MONTH,
    "rounds_completed": len(all_stats),
    "final_mean_count": round(final_counts["scaleworm_count"].mean(), 1),
    "final_median_count": int(final_counts["scaleworm_count"].median()),
    "final_std": round(final_counts["scaleworm_count"].std(), 1),
    "n_videos": len(final_counts),
    "rounds": all_stats,
}

with open(final_dir / "report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"Final results exported to: {final_dir}")
print(f"  counts.parquet:  {len(final_counts)} videos")
print(f"  model.pth:       trained detector")
print(f"  report.json:     quality report card")
"""))

    cells.append(md("""
## Report Card

The figure below summarizes your multi-round improvement journey:

- **(a) Error rates** — FP and FN rates should decrease across rounds
- **(b) Count stability (MAE)** — how much counts change between consecutive rounds;
  should decrease toward 0 as the model stabilizes
- **(c) Mean count evolution** — how the average count changes; should converge
- **(d) Review effort** — how many frames you reviewed per round
"""))

    cells.append(code("""
# ── Report Card Figure ──
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

rounds = [s["round"] for s in all_stats]

# Panel A: FP/FN rate per round
axes[0, 0].plot(rounds, [s["fp_rate"] for s in all_stats], "o-", label="FP rate")
axes[0, 0].plot(rounds, [s["fn_rate"] for s in all_stats], "s-", label="FN rate")
axes[0, 0].axhline(15, color="red", linestyle=":", alpha=0.5, label="FP threshold")
axes[0, 0].axhline(10, color="orange", linestyle=":", alpha=0.5, label="FN threshold")
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_xlabel("Round")
axes[0, 0].set_ylabel("Rate (%)")
axes[0, 0].set_title("(a) Error Rates")

# Panel B: MAE between rounds
if len(all_stats) > 1:
    axes[0, 1].plot(rounds[1:], [s["mae_between_rounds"] for s in all_stats[1:]], "o-")
    axes[0, 1].axhline(0.5, color="red", linestyle=":", alpha=0.5, label="Threshold")
    axes[0, 1].legend()
axes[0, 1].set_xlabel("Round")
axes[0, 1].set_ylabel("MAE")
axes[0, 1].set_title("(b) Count Stability (MAE)")

# Panel C: Mean count per round
axes[1, 0].plot(rounds, [s["mean_count_new"] for s in all_stats], "o-")
axes[1, 0].set_xlabel("Round")
axes[1, 0].set_ylabel("Mean count")
axes[1, 0].set_title("(c) Mean Count Evolution")

# Panel D: Frames reviewed
axes[1, 1].bar(rounds, [s["frames_reviewed"] for s in all_stats])
axes[1, 1].axhline(50, color="red", linestyle=":", alpha=0.5, label="Minimum")
axes[1, 1].legend()
axes[1, 1].set_xlabel("Round")
axes[1, 1].set_ylabel("Frames")
axes[1, 1].set_title("(d) Review Effort")

plt.suptitle(f"Report Card -- {STUDENT_NAME} ({ASSIGNED_MONTH})", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(final_dir / "report_card.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\\nReport card saved: {final_dir / 'report_card.png'}")
print(f"\\n{'='*60}")
print(f"  ANALYSIS COMPLETE")
print(f"  Submit the contents of {final_dir} to your supervisor.")
print(f"{'='*60}")
"""))

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================

    cells.append(md("""
---

## Summary

| Phase | What you did | Key output |
|-------|-------------|------------|
| 1 | Ran pretrained detector on your month | `counts.parquet`, `phase1_overview.png` |
| 2 | Reviewed ~100 frames, corrected FP/FN | `corrections.json`, `frame_labels.json` |
| 3 | Retrained model, compared before/after | `best_model.pth`, `comparison.png` |
| 4 | Exported final counts and report card | `final/` directory |

| Variable | Where to change | When to change |
|----------|----------------|----------------|
| `STUDENT_NAME` | Configuration cell | Once, at the start |
| `ASSIGNED_MONTH` | Configuration cell | When starting secondary month |
| `CURRENT_ROUND` | Configuration cell | After each non-converged round |
| `BATCH_NUMBER` | Phase 2, "Show next batch" cell | To cycle through labeling batches |
| `batch_corrections` | Phase 2, "Enter corrections" cell | After viewing each batch |

**To start a new round:** Change `CURRENT_ROUND` in the Configuration cell and re-run
from Phase 2 onward (**Kernel -> Restart & Run All** is fine — completed work is skipped).

**To start your secondary month:** Change `ASSIGNED_MONTH` and reset `CURRENT_ROUND = 1`.
"""))

    # =========================================================================
    # ASSEMBLE NOTEBOOK
    # =========================================================================

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "cells": cells,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(notebook, f, indent=1)

    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"Notebook written: {OUTPUT}")
    print(f"  {len(cells)} cells ({n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    build_notebook()
