# ReStraV: AI-Generated Video Detection via Perceptual Straightening

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/virtual/2025/poster/118520)
[![arXiv](https://img.shields.io/badge/arXiv-24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2507.00583)

Official implementation of the paper **"AI-Generated Video Detection via Perceptual Straightening"**, accepted at NeurIPS 2025.

![ReStraV Method Pipeline](./assets/pipeline.png)
*Figure 1: The ReStraV method. Video frames are processed by a self-supervised encoder (DINOv2) to get embeddings. In this representation space, natural videos trace "straighter" paths than AI-generated ones. The trajectory's geometry, especially its curvature, serves as a powerful signal for a lightweight classifier to distinguish real from fake.*

> **Important (local setup knobs):** several scripts include **hard-coded values** for `device` (e.g. `cuda:1`), `batch_size`, `num_workers`, paths, and download worker counts.  
> **You will likely need to open the files and change these values** to match your machine (GPU index, RAM/VRAM, CPU cores, filesystem layout).

---

## What this repo does

**Core idea:**  
1. Sample a short clip from each video (default: ~2 seconds, 24 frames).  
2. Encode frames with a pretrained vision backbone (**DINOv2 ViT-S/14** via `torch.hub`).  
3. Treat the per-frame embeddings as a trajectory in representation space.  
4. Compute **temporal geometry features**: stepwise distances and curvature/turning angles across time.  
5. Train a lightweight classifier (an MLP) on a **21-D feature vector** per video.  
6. Use the trained model to predict whether a new video is **REAL** or **FAKE**.

---

## Repository layout (high level)

- `dinov2_features.py` — video decoding + DINOv2 embedding extraction + 21-D feature computation
- `train.py` — trains the MLP classifier; saves `model.pt`, `mean.npy`, `std.npy`, `best_tau.npy`
- `demo.py` — Gradio demo (upload video or paste URL; uses `yt-dlp` to download)
- `DATA/` — data + helper scripts (download/extract features) and generated artifacts

---

## Method details (the 21-D feature vector)

The feature builder in `dinov2_features.py` computes:

- **7** early stepwise distances: `d[0:7]`
- **6** early turning angles: `theta[0:6]`
- **8** summary statistics (mean/min/max/variance) for distances and angles:
  - `μ_d, min_d, max_d, var_d`
  - `μ_θ, min_θ, max_θ, var_θ`

Total: `7 + 6 + 8 = 21` features per video.

---

## Setup

### 1) Clone

```bash
git clone https://github.com/ChristianInterno/ReStraV.git
cd ReStraV
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Data (training)

* **REAL videos**: pulled from the Video Similarity Challenge URL list, filtered by a local reference list file
* **FAKE videos**: pulled from **VidProM** (often the `example/` subset from Hugging Face)

---

## Step-by-step pipeline

### Step A — Download training videos

```bash
python DATA/download_training_data.py
```

* Downloads a subset of REAL mp4s by matching filenames from a `ref_file_paths.txt` list
* Downloads FAKE examples from the VidProM dataset and extracts `.tar` files into `FAKE/`

**Things you may need to edit inside the script:**

* `MAX_WORKERS` (default may be too high for your network / OS)
* `TIMEOUT`

---

### Step B — Extract DINOv2 geometry features into an HDF5

```bash
python DATA/extract_training_features.py
```

This writes an HDF5 file:

* `path` (string)
* `label` (int; 1=real, 0=fake)
* `features` (float; shape `[N, 21]`)

**Things you may need to edit inside this script:**

* `batch_size`
* `device`

---

### Step C — Train the classifier

```bash
python train.py
```

* Loads all samples from the HDF5
* Balances classes by subsampling to equal priors
* Normalizes features (saves `mean.npy` and `std.npy`)
* Splits 50/50 train/test with stratification
* Trains a small MLP for a fixed number of epochs
* Picks an operating threshold `τ*` maximizing F1 on the training set
* Evaluates on test set; writes `test_predictions_all.csv`
* Saves model weights to `model.pt`

**Things you may need to edit inside `train.py`:**

* `device`
* DataLoader `batch_size`
* `num_workers`
* `epochs`, learning rate, hidden sizes

Outputs written in the working directory by default:

* `model.pt`
* `mean.npy`
* `std.npy`
* `best_tau.npy`
* `test_predictions_all.csv`

---

## Demo (Gradio)

Once you have `model.pt`, `mean.npy`, `std.npy`, and `best_tau.npy` in the repo root:

```bash
python demo.py
```

The demo supports:

* Uploading a video file, **or**
* Pasting a URL; it downloads the video via `yt-dlp` into a temp folder

---

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{
interno2025aigenerated,
title={{AI}-Generated Video Detection via Perceptual Straightening},
author={Christian Intern{\`o} and Robert Geirhos and Markus Olhofer and Sunny Liu and Barbara Hammer and David Klindt},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=LsmUgStXby}
}
```

## Acknowledgements
This research was partly funded by Honda Research Institute Europe and Cold Spring Harbor Laboratory. We would like to thank Eero Simoncelli for insightful discussions and feedback, as well as all our colleagues from Google DeepMind, the Machine Learning Group at Bielefeld University, Honda Research Institute for the insightful discussions and feedback.

All code in this repository was contributed by Sam Pagon (@sampagon).
