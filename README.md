# OccDet-Calib: Dangerous Zones in Occluded Object Detection

OccDet-Calib is a benchmark and analysis toolkit for studying **confidence calibration under structured partial visibility** in object detection.

Starting from lightly occluded, safety-relevant MS COCO instances, OccDet-Calib generates matched visibility-loss variants (overlap, distractor, truncation, and inpainting) and evaluates multiple detector families to analyze how **detection quality and calibration diverge under occlusion**.

> This repository accompanies the paper  
> **“OccDet-Calib: Dangerous Zones in Occluded Object Detection”**  
> (submitted to the *Journal of Visual Communication and Image Representation*).

---

## Key ideas

- **Structured occlusion benchmark** built from safety-relevant COCO classes (person, vehicle categories, stop sign).
- **Multiple variant families**:
  - Overlap (central occluders covering the target)
  - Distractor control (nearby clutter without covering the target)
  - Truncation control (border clipping instead of central overlap)
  - Inpainting subset (realism-oriented corruption)
- **Calibration-centric metrics**:
  - Detection quality via F1 at IoU 0.5
  - Detection Expected Calibration Error (DECE)
  - Bootstrap 95% confidence intervals for all metrics
- **Dangerous Zone (DZ)**:
  - Occlusion regime where detection remains usable but confidence has already become unsafe.
  - Defined via held-out calibration thresholds and evaluated on a separate split.
- **Mitigation baselines**:
  - Global temperature scaling (TS)
  - Simple occlusion-conditioned TS (OC-TS)
  - Lightweight detector-side visibility proxy (negative result)

---

## Core detectors

OccDet-Calib currently supports three detector families:

- **YOLOv8m** (one-stage, anchor-based CNN)
- **FCOS R50-FPN** (anchor-free one-stage CNN)
- **Deformable DETR R50** (transformer-based detector)

The code is organized so that additional detectors can be added via a simple wrapper interface.

---

## Repository structure

```text
OccDet-Calib/
├── src/                # Core benchmark and evaluation code
│   ├── data_gen/       # Occlusion variant generation (overlap, distractor, truncation, inpainting)
│   ├── detectors/      # Detector wrappers (YOLOv8m, FCOS, Deformable DETR)
│   ├── metrics/        # F1, DECE, bootstrap utilities
│   ├── dz/             # Dangerous Zone definition & held-out protocol
│   └── viz/            # Plotting / figure helpers
├── configs/            # Experiment configs (paths, thresholds, detector settings)
├── data/               # Local placeholders only (see "Data & assets" below)
├── outputs/            # Generated metrics, logs, and plots (git-ignored if large)
├── paper/              # Manuscript and figure assets (LaTeX, tikz, etc.)
└── README.md           # This file
```

---

## Getting started

### 1. Environment

We recommend Python 3.10+.

```bash
git clone https://github.com/<user>/occdet-calib.git
cd occdet-calib

# Example: create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The requirements file should include the main dependencies (PyTorch, torchvision, YOLOv8 / ultralytics, detectron2/mmdetection or the chosen FCOS/DETR framework, numpy, pandas, matplotlib, etc.).

### 2. Data & assets

OccDet-Calib builds on MS COCO 2017 validation images and annotations.

Because we cannot redistribute COCO directly, you must:

1. Download COCO 2017 val images and annotations from the official site.
2. Update `configs/data_paths.yaml` (or similar) to point to your local COCO paths.
3. Run the preprocessing script to create OccDet-Calib variants.

Example:

```bash
python src/data_gen/build_occdet_calib.py \
  --coco-root /path/to/coco2017 \
  --out-root data/occdet_calib
```

This will generate:

- Seed instances for safety-relevant classes.
- Overlap, distractor, truncation, and inpainting variants.
- Metadata with nominal and realized occlusion levels.

---

## Running the experiments

### 1. Detector sweeps and thresholds

We first run clean-data sweeps to select operating thresholds for each detector (e.g., confidence thresholds that maintain recall ≥ 0.60).

Example:

```bash
# YOLOv8m sweep on clean validation crops
python src/detectors/run_yolov8_sweep.py \
  --config configs/yolov8_clean.yaml
```

The resulting thresholds are referenced in the configs for occlusion experiments (e.g., YOLOv8m 0.40, FCOS 0.20, Deformable DETR 0.30).

### 2. Main overlap benchmark

```bash
python src/experiments/run_overlap_benchmark.py \
  --config configs/overlap_yolov8.yaml

python src/experiments/run_overlap_benchmark.py \
  --config configs/overlap_fcos.yaml

python src/experiments/run_overlap_benchmark.py \
  --config configs/overlap_detr.yaml
```

This will compute F1 and DECE across occlusion levels (0.0–0.8) and save metrics + bootstrap confidence intervals under `outputs/`.

### 3. Controls: distractor, truncation, inpainting

```bash
python src/experiments/run_controls.py \
  --config configs/controls_yolov8.yaml
```

Generates results for:

- Overlap vs distractor vs truncation.
- Overlap vs inpainting subset.

### 4. Dangerous Zone analysis

```bash
python src/dz/run_dz_analysis.py \
  --config configs/dz_heldout.yaml
```

This script:

- Splits the benchmark into calibration/evaluation halves by image ID.
- Derives per-detector DECE thresholds on the calibration split.
- Reports DZ flags on the evaluation split across occlusion levels and percentile choices.

### 5. Calibration baselines

```bash
python src/experiments/run_calibration_baselines.py \
  --config configs/calibration_baselines.yaml
```

Produces results for:

- Raw detector confidence.
- Global temperature scaling (TS).
- Simple occlusion-conditioned TS (OC-TS).

---

## Reproducing paper figures

Once metrics are computed, you can regenerate the main figures and tables:

```bash
python src/viz/plot_main_overlap.py       # F1 & DECE vs occlusion for all architectures
python src/viz/plot_dz_flags.py           # Dangerous Zone plots
python src/viz/plot_controls.py           # Overlap vs distractor vs truncation
python src/viz/plot_inpainting_subset.py  # Inpainting vs overlap
```

Outputs are written to `outputs/figures/` and correspond to the figures in the paper.

---

## Citation

If you find OccDet-Calib useful in your research, please cite:

```bibtex
@article{adari2025occdetcalib,
  title   = {OccDet-Calib: Dangerous Zones in Occluded Object Detection},
  author  = {Adari, Karthik and Uppala, Yojitha and Uppala, Sai Ram},
  journal = {Journal of Visual Communication and Image Representation},
  year    = {2026},
  note    = {submitted}
}
```

(You can update the bibliographic details once the paper is accepted and assigned volume/pages.)

---

## License

Specify the license clearly, e.g.:

- Code: MIT / Apache-2.0 (your choice).  
- Data: MS COCO is subject to its own license; OccDet-Calib variants depend on COCO and are therefore not redistributed here.

Example:

```markdown
Code is released under the MIT License.  
MS COCO data is not included in this repository. Please obtain COCO data from the official source and respect the original license terms.
```

---

## What to include / avoid mentioning

You **can mention**:

- The high-level ideas, benchmark design, and detector families.
- That this repo accompanies a paper (and that it is submitted).
- How to reconstruct the benchmark and run the experiments.
- Any public dependencies you use (YOLOv8, FCOS, Deformable DETR, COCO).

You should **avoid**:

- Copying the paper abstract verbatim (short paraphrase is fine).
- Including any confidential reviewer comments or rebuttal details.
- Making claims that go beyond what your experiments show.
