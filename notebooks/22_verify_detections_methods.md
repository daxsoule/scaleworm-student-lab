*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

# Methods: Detection Verification (Step 22)

## §22.1 Purpose

<span style="font-family: 'Courier New', monospace;">

This notebook implements a human-in-the-loop verification workflow for YOLO-based scale worm (*Lepidonotopodium piscesae*) detections from OOI CAMHD video of Mushroom vent (ASHES hydrothermal field, Axial Seamount). The purpose is to convert raw model predictions into a curated training dataset by having a human reviewer classify each candidate detection as a true scale worm or a false positive.

The notebook is designed as a student lab exercise: students learn to evaluate object detection outputs, understand the precision–recall tradeoff, and produce labeled datasets suitable for downstream model training.

</span>

---

## §22.2 Input Data

<span style="font-family: 'Courier New', monospace;">

### §22.2.1 CAMHD Video Archive

| Property | Value |
|---|---|
| Instrument | OOI CAMHD (RS03ASHS-PN03B-06-CAMHDA301) |
| Location | Mushroom vent, ASHES hydrothermal field, Axial Seamount |
| Depth | ~1520 m |
| Cadence | 8 videos per day at 3-hour intervals (UTC: 00:15, 03:15, 06:15, 09:15, 12:15, 15:15, 18:15, 21:15) |
| Duration | ~25 minutes per video |
| Resolution | 1920 × 1080 pixels |
| Format | MP4 (H.264) |
| Archive path | `/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301/` |
| Filename convention | `CAMHDA301-YYYYMMDDTHHmmss.mp4` |

### §22.2.2 Scene 1 Definition

Each CAMHD video follows a programmed observation profile with multiple scenes (zoom positions). **Scene 1** (305–320 seconds into each video) is the close-up view of the Mushroom vent chimney, where scale worms are visible. This 15-second window is the only portion of each video used for worm detection.

### §22.2.3 YOLO Mushroom Model

| Property | Value |
|---|---|
| Architecture | YOLOv11 (via ultralytics) |
| Training data | ~600 manually labeled CAMHD frames |
| Classes | 1 (scale_worm) |
| Weights file | `viame_my-analysis/notebooks/runs/outputs/model/yolo_v26/train_v1/weights/best.pt` |
| Inference speed | ~119 fps on GPU (NVIDIA L40S), ~2 fps on CPU |

</span>

---

## §22.3 Algorithm

<span style="font-family: 'Courier New', monospace;">

The verification pipeline has 6 sequential stages:

### §22.3.1 Video Discovery

Given a user-specified date range (`START_DATE`, `END_DATE`), the pipeline searches the CAMHD archive recursively for video files matching the naming convention `CAMHDA301-YYYYMMDDTHHmmss.mp4`. Two filters are applied:

1. **Date filter:** Only videos within the inclusive date range are kept.
2. **Cadence filter:** Only videos recorded at the 8 standard 3-hour UTC times are kept. Non-standard recordings (engineering tests, special observations) are excluded because the camera zoom position at 305 seconds may differ from the standard observation profile.

### §22.3.2 Frame Extraction

For each selected video, `ffmpeg` extracts frames from the Scene 1 window:

- **Seek position:** 305 seconds (`-ss 305`)
- **Duration:** 15 seconds (`-t 15`)
- **Frame rate:** Configurable via `FPS` (default: 1 fps → 15 frames per video)
- **Output format:** PNG (`frame_0001.png`, `frame_0002.png`, etc.)

The extraction is **idempotent**: if frames already exist for a video (from a previous run), they are not re-extracted.

**Choice of FPS:** The default of 1 fps (15 frames/video) balances temporal coverage against verification burden. At 10 fps (150 frames/video), near-duplicate detections across adjacent frames dramatically increase the number of candidates to review without proportional information gain. For student use, 1 fps is recommended.

### §22.3.3 YOLO Inference

The YOLO Mushroom Model runs on all extracted frames with two key parameter choices:

1. **Confidence threshold = 0.1:** Intentionally low to maximize recall. At this threshold, the model reports any detection it is ≥10% confident about. This catches nearly all real worms but also produces many false positives (bacterial mats, tube structures, lighting artifacts). The human reviewer separates true from false detections in Step 6.

2. **Size filter (MAX_BOX_SIZE = 300 px):** Bounding boxes wider or taller than 300 pixels are rejected. Real scale worms produce boxes of 20–100 pixels. Large boxes correspond to chimney structures or whole-frame artifacts.

Inference uses `stream=True` (generator mode) to process one frame at a time, preventing GPU memory accumulation. Bounding box coordinates and confidence scores are converted from GPU tensors to CPU numpy arrays immediately after each frame.

### §22.3.4 Crop Extraction

Each detection's bounding box is extracted from its source frame with 20 pixels of context padding on all sides (clamped to image boundaries). Crops are saved as individual PNG files for efficient random access during verification.

A frame cache (maximum 50 entries, LRU eviction) avoids redundant disk reads when multiple detections come from the same frame. The cache is cleared after all crops are saved.

### §22.3.5 Interactive Verification

An `ipywidgets`-based interface presents each candidate detection at three zoom levels:

| Panel | Scale | Purpose |
|---|---|---|
| Left | 1× (native pixels) | Overall shape and context |
| Center | 2× (nearest-neighbor) | Medium detail |
| Right | 4× (nearest-neighbor) | Fine anatomical features (leg pairs, segmentation) |

**Nearest-neighbor interpolation** is used for zooming to preserve pixel-level detail without blurring. This is critical for small worms (20–40 px) where diagnostic features span only a few pixels.

The reviewer assigns one of three labels per detection:

| Label | Stored value | Meaning |
|---|---|---|
| ✓ Scale Worm | `"scale_worm"` | True positive — included in export |
| ✗ Not a Worm | `"not_worm"` | False positive — excluded from export |
| ⟳ Skip | `"skip"` | Uncertain — excluded from export, revisit later |

**Auto-save:** After every label assignment, all labels are serialized to `verification_progress.json`. This provides crash resilience — if the kernel dies, labels are not lost. On restart, re-running the notebook restores all previously assigned labels from this file.

**Navigation:** After labeling, the widget advances to the next unlabeled detection using circular search (wraps from end to beginning). The "Previous" button allows reviewing and correcting earlier decisions.

### §22.3.6 YOLO Dataset Export

Verified `"scale_worm"` detections are packaged into a YOLO-format training dataset:

**Coordinate conversion:** Pixel bounding boxes (x1, y1, x2, y2) are converted to YOLO normalized format (center_x, center_y, width, height), where all values are in [0, 1] relative to the 1920×1080 frame dimensions.

**Directory structure:**

```
export/
├── images/train/           # source frame PNGs (only frames with verified worms)
│   └── CAMHDA301-..._frame_0003.png
├── labels/train/           # one .txt per image, one line per worm detection
│   └── CAMHDA301-..._frame_0003.txt
└── dataset.yaml            # YOLO training configuration
```

**Label file format:** Each line: `0 center_x center_y width height` (class 0 = scale_worm).

**dataset.yaml** specifies paths, number of classes (1), and class names. Both `train` and `val` point to `images/train` — the user should split images into separate directories for proper training/validation.

The export directory is compressed into a zip file named `verified_scaleworm_dataset_<START_DATE>_to_<END_DATE>.zip`.

</span>

---

## §22.4 Parameters Reference

<span style="font-family: 'Courier New', monospace;">

| Parameter | Value | Defined in | Rationale |
|---|---|---|---|
| `SCENE1_START_SEC` | 305 | Cell 2 (Setup) | CAMHD observation profile places Scene 1 at this offset |
| `SCENE1_DURATION_SEC` | 15 | Cell 2 (Setup) | Full duration of the chimney close-up view |
| `FPS` | 1 | Cell 2 (Setup) | Balances temporal coverage vs. verification burden |
| `CONF_THRESHOLD` | 0.1 | Cell 2 (Setup) | Maximizes recall — students verify precision manually |
| `MAX_BOX_SIZE` | 300 | Cell 2 (Setup) | Real worms are 20–100 px; rejects chimney-scale artifacts |
| `PAD_PX` | 20 | Cell 13 (Crop) | Context padding around each crop for visual reference |
| `STANDARD_TIMES` | 8 UTC times | Cell 2 (Setup) | Standard 3-hour cadence; rejects non-standard recordings |
| `FRAME_W × FRAME_H` | 1920 × 1080 | Cell 2 (Setup) | CAMHD native resolution; used for YOLO normalization |

</span>

---

## §22.5 Outputs

<span style="font-family: 'Courier New', monospace;">

| Output | Path | Description |
|---|---|---|
| Extracted frames | `verification_session/frames/<video>/frame_NNNN.png` | Scene 1 frames at configured FPS |
| Detection crops | `verification_session/crops/crop_NNNNNN.png` | Padded crops of each candidate detection |
| Verification progress | `verification_session/verification_progress.json` | Labels assigned during review (auto-saved) |
| YOLO dataset | `verification_session/export/` | images/, labels/, dataset.yaml |
| Dataset zip | `verification_session/verified_scaleworm_dataset_<dates>.zip` | Compressed archive for distribution |

</span>

---

## §22.6 Dependencies

<span style="font-family: 'Courier New', monospace;">

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | latest | YOLO model loading and inference |
| `ipywidgets` | ≥7.0 | Interactive verification buttons |
| `matplotlib` | ≥3.0 | Crop display via `imshow` |
| `Pillow` | ≥9.0 | Image loading, resizing, saving |
| `numpy` | ≥1.20 | Array operations on image data |
| `ffmpeg` | system binary | Frame extraction from video files |

</span>

---

## §22.7 Validation

<span style="font-family: 'Courier New', monospace;">

### §22.7.1 End-to-End Test (2026-04-08)

The notebook was tested on 1 video (2024-10-04), producing 15 frames at 1 fps and 181 detections at conf≥0.1. All YOLO label coordinates were verified to be normalized values in [0, 1]. The save/resume mechanism was tested by simulating a kernel restart and confirming label persistence.

### §22.7.2 Recommended Validation Steps

1. **Path check:** After running Cell 2, confirm both `MODEL_PATH` and `VIDEO_ROOT` report `Exists: True`.
2. **Frame count:** Each video should yield approximately `FPS × SCENE1_DURATION_SEC` frames.
3. **Detection sanity:** Compare per-video detection counts. Large variation may indicate lighting or camera issues.
4. **Confidence distribution:** At conf=0.1, expect 50–80% false positives. At conf=0.9, expect <10%.
5. **Resume test:** After labeling ~10 detections, restart the kernel, re-run cells 1–5, and verify labels are restored.
6. **Export check:** Verify that the number of exported images matches the number of unique frames with worm detections, and that all label file values are in [0, 1].

</span>
