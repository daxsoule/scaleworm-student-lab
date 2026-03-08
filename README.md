# Scale Worm Student Lab

Human-in-the-loop object detection for counting scale worms (*Lepidonotopodium* sp.) at the Mushroom hydrothermal vent on Axial Seamount, using OOI CAMHD video.

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/specKitScience/scaleworm-student-lab.git
cd scaleworm-student-lab
```

### 2. Download the starter package

Download `scaleworm_starter_package.zip` (~3 GB) from the shared drive and place it in your home directory:

```
~/scaleworm_starter_package.zip
```

The notebook will unzip it for you on first run.

### 3. Install dependencies

```bash
uv sync
```

### 4. Open the notebook

Open `notebooks/20_student_scaleworm_pipeline.ipynb` in JupyterHub and follow the instructions inside.

## What's in the starter package?

| Item | Description |
|------|-------------|
| `annotations/*.zip` | MP4 video clips + VIAME CSV annotations for March, April, June, August 2023 |
| `model/best_model_v3.pth` | Pretrained Faster R-CNN v2 detector |
| `baseline/scaleworm_counts.parquet` | 2023 ground truth counts for comparison |
| `baseline/annotations_scene1/*.csv` | Parsed scene 1 annotation CSVs |
| `labels.txt` | Class names (`scale_worm`) |

## Workflow

1. **Phase 1** — Run the pretrained detector on your assigned month (~30 min)
2. **Phase 2** — Review detections and correct mistakes (~2-3 hours)
3. **Phase 3** — Retrain the model with your corrections (~1 hour)
4. **Phase 4** — Export final counts and quality report

## Requirements

- JupyterHub with GPU access
- `ffmpeg` installed
- Video archive at `/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301/`

## For instructors

`notebooks/21_cross_student_comparison.ipynb` compares results across all students after they complete their work.
