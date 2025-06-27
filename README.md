# Enhanced DeepSORT: Modern Detection & ReID Integration

## Overview

This repository contains an enhanced implementation of **DeepSORT** that integrates modern, efficient person detection and ReID algorithms. It builds upon the original DeepSORT codebase, extending it with support for multiple detectors, ReID models, and optional segmentation, while preserving the original commit history. This solution achieves better performance than the unmodified DeepSORT on MOT Challenge videos, balancing tracking accuracy and real-time speed fileciteturn0file0.

## Features

* **Multiple Detection Models**: Easily switch between at least three detectors (YOLOv5, YOLOv8, SSD-Lite) via configuration.
* **Multiple ReID Models**: Support for at least three ReID backbones (original DeepSORT, ResNet18, TorchReID) with dynamic selection.
* **Optional Segmentation**: Toggle between detection and segmentation-based tracking pipelines.
* **Real-Time Performance**: Achieves ≥5 FPS in Colab on at least one model combination, outperforming the original implementation in HOTA metric.
* **Batch Processing**: Process all MOT Challenge videos in one command with configurable parameters.
* **Evaluation & Visualization**: Built-in scripts for computing HOTA, generating tracking overlays, and exporting result videos.
* **Colab Notebook**: Fully operational Colab with execution instructions available in `/notebooks`.

## Repository Structure

```text
├── configs/                 # Detector and tracker configurations (.yaml)
│   ├── ssdlite320_mbv3_resnet18.yaml
│   ├── yolov5_torchreid.yaml
│   └── yolov8_mobilenetv2.yaml
├── detectors/               # Detector implementations
├── reid_models/             # ReID backbones
├── application_util/        # Preprocessing & visualization utilities
├── tools/                   # Helper scripts (generate_detections, freeze_model)
├── utils/                   # Config parsing, drawing, MOT export
├── videos/                  # Raw MOT Challenge videos
├── data/                    # MOT dataset directories
├── outputs/                 # Tracking outputs per config
├── trackers/                # Formatted tracker results for MOT Challenge
├── notebooks/               # Colab notebooks with tutorials
├── run_batch.py             # Batch tracker execution script
├── evaluate_motchallenge.py # Compute HOTA on MOT datasets
├── generate_videos.py       # Build overlay videos of tracking results
├── convert_sequences.py     # Convert image folders to .mp4 videos
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Setup

1. **Clone the repository**:

   ```powershell
   PS> git clone --branch develop --single-branch https://github.com/LeinMS/deep_sort.git
   ```
2. **Create a Python environment**:

   ```powershell
   PS> python -m venv .venv
   PS> .\.venv\Scripts\Activate.ps1
   ```
3. **Install dependencies**:

   ```powershell
   PS> pip install -r requirements.txt
   ```

## Data Preparation

1. **Download MOT Challenge datasets**:

   * Place sequences under `data/{SequenceName}` with subfolders `img1/`, `gt/`, and `det/`.
2. **Convert image sequences to videos** (optional):

   ```powershell
   PS> python convert_sequences.py
   ```

## Generating Detections

Use one of the supported detectors to generate bounding-box proposals:

```powershell
PS> python tools\generate_detections.py `
    --model resources\models\<model>.pt `
    --mot_dir data\MOT16-09 `
    --output_dir resources\detections\MOT16-09
```

## Configuring and Running the Tracker

Select a detector+ReID combo via one of the YAML configs in `configs/`:

```powershell
PS> .\.venv\Scripts\Activate.ps1  # ensure env is active
PS> python run_batch.py configs\yolov5_torchreid.yaml
```

This processes the standard MOT set: TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1, MOT16-09, MOT16-11.

### Single-Sequence Run

To track a single sequence:

```powershell
PS> python deep_sort_app.py `
    --sequence_dir data\MOT16-09 `
    --detection_file resources\detections\MOT16-09.npy `
    --output_file outputs\MOT16-09.txt `
    --min_confidence 0.3 `
    --max_cosine_distance 0.2 `
    --nn_budget 100 `
    --display False
```

## Evaluation

Compute the HOTA metric for all sequences:

```powershell
PS> python .\scripts\run_mot_challenge.py `
    --BENCHMARK MOT16 `
    --SPLIT_TO_EVAL train `
    --TRACKERS_TO_EVAL yolov5_torchreid_tracker `
    --METRICS HOTA `
    --USE_PARALLEL False `
    --NUM_PARALLEL_CORES 1 `
    --SEQMAP_FILE .\seqmaps\seqmaps_mot16.txt `
    --SKIP_SPLIT_FOL True
```

Results (per-sequence and average HOTA) are saved in `evaluation_results/`.

## Visualization

Generate overlay videos of tracking results:

```powershell
PS> python generate_videos.py `
    --mot_dir data `
    --result_dir trackers\mot_challenge\<config_name>\data `
    --output_dir videos\results
```

## 🚀 Google Colab Notebook

We provide a ready-to-run Colab notebook that includes full setup, tracking execution, and result visualization.

[Open in Google Colab](https://github.com/LeinMS/deep_sort/blob/develop/_new_deepsort.ipynb)


## Report

See `report/Enhanced_DeepSORT_Report.pdf` for a detailed description of model choices, parameter tuning, and performance analysis.

## License

This project is released under the MIT License.

## References

* Original DeepSORT paper: Nicolai Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric" fileciteturn0file1
* Deep Cosine Metric Learning: Alex Bewley et al., WACV 2018.
