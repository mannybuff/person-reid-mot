# Person Re-Identification and Multi-Object Tracking

This repository contains a cleaned, portfolio-ready version of a deep learning project
that combines:

- **Person re-identification (ReID)** on the **Market-1501** dataset using a Siamese network
  with triplet loss, and
- A **Faster R-CNN** based detector for pedestrian localization on **MOT-style** sequences,
  suitable for building a simple multi-object tracking (MOT) pipeline.

Originally developed as a Colab-based homework project, this repo restructures the code into
reusable Python modules and documents the experiments in a way that is easier to read and adapt.

---

## 1. Overview

The project is split into two main components:

1. `src/train_reid_market1501.py` – Train an embedding model for person ReID on Market-1501
2. `src/train_detector_fasterrcnn.py` – Fine-tune a Faster R-CNN detector on a MOT-style pedestrian dataset

These pieces can be used independently:

- The ReID model can be applied to re-identify people across cameras or over time.
- The detector can be used for generic pedestrian detection.

Or together:

- Use the detector to extract tracklets / detections from a video sequence.
- Use the ReID embeddings to match detections across frames or cameras.

---

## 2. Repository Structure

Suggested structure for the local repo:

```text
person-reid-mot/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  ├─ train_reid_market1501.py      # ReID training script (triplet loss)
│  └─ train_detector_fasterrcnn.py  # Pedestrian detector training script
└─ data/
   ├─ market1501/                   # Market-1501 dataset (not tracked in git)
   └─ mot/                          # MOT-style dataset (train/val/test splits)
```

The `data/` folder is not included in the repository; you are expected to download the datasets
from their official sources and organize them locally.

---

## 3. Component 1 – Person ReID on Market-1501

### 3.1 Task

Given an image of a person from one camera, retrieve images of the same person from other cameras
and frames. This requires learning a feature embedding where:

- Samples from the **same identity (PID)** are close together.
- Samples from **different identities** are far apart.

### 3.2 Approach

The training script:

- Uses the **Market-1501** dataset with its standard directory structure.
- Implements a **triplet sampling dataset** that yields `(anchor, positive, negative)` triplets.
- Uses a **Siamese-style network** with:
  - A convolutional backbone (e.g., ResNet-50)
  - A projection head that outputs L2-normalized embeddings
- Optimizes with a **triplet margin loss**.

The resulting model can be used to compute embeddings for query and gallery sets and perform
nearest-neighbor retrieval or more advanced matching schemes.

---

## 4. Component 2 – Pedestrian Detection with Faster R-CNN

### 4.1 Task

Detect pedestrians in MOT-style video frames (e.g., sequences from MOT16/MOT17 or similar),
producing bounding boxes and class scores for each frame.

### 4.2 Approach

The detector training script:

- Uses a directory with images and MOT-style annotations (or a compatible format).
- Wraps the dataset into a `torchvision`-compatible `Dataset`.
- Fine-tunes **Faster R-CNN** (ResNet-50 FPN backbone) for a single `person` class.
- Reports training/validation losses and simple metrics.

The core idea is to produce a robust per-frame detector that can later be combined with
ReID embeddings and a simple association strategy (e.g., Hungarian matching over embedding + IOU
distance) to construct a MOT system.

---

## 5. Installation

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Usage

### 6.1 Training the ReID model

1. Download Market-1501 and place it under `data/market1501/` with the usual structure
   (`bounding_box_train`, `bounding_box_test`, `query`, etc.).
2. Edit the `DATA_ROOT` path in `src/train_reid_market1501.py` if needed.
3. Run:

```bash
python src/train_reid_market1501.py
```

This will:

- Build the triplet dataset and data loader.
- Train the embedding network for a configurable number of epochs.
- Save the trained weights and (optionally) embeddings.

### 6.2 Training the pedestrian detector

1. Prepare a MOT-style dataset under `data/mot/`, organized into train/val/test splits.
2. Update the dataset paths in `src/train_detector_fasterrcnn.py` as needed.
3. Run:

```bash
python src/train_detector_fasterrcnn.py
```

This will:

- Fine-tune Faster R-CNN for the `person` class.
- Save model checkpoints.
- Report basic validation metrics (loss, detection stats).

---

## 7. Requirements

See `requirements.txt` for a concise list of dependencies. The main ones are:

- `torch`, `torchvision`
- `numpy`
- `Pillow`
- `tqdm`
- `scikit-learn` (for evaluation helpers)
- `matplotlib` (optional for visualizations)

A GPU is strongly recommended for training both components.

---

## 8. License

If this project is hosted on GitHub with a **MIT License** turned on at repo creation time,
the canonical license text will be in the `LICENSE` file generated by GitHub.
