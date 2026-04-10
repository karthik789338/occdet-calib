# OccDet-Calib: Dangerous Zones in Occluded Object Detection

OccDet-Calib is a benchmark and analysis toolkit for studying **confidence calibration under structured partial visibility** in object detection.

It accompanies the paper:

> **OccDet-Calib: Dangerous Zones in Occluded Object Detection**  
> (submitted to the *Journal of Visual Communication and Image Representation*).

---

## Repository layout

```text
occdet-calib/
├── configs/        # YAML configs for data paths, detectors, and experiments
├── data/           # Local data folder (COCO + generated OccDet-Calib assets; not tracked)
├── environment/    # Environment files (requirements.txt, conda envs, etc.)
├── notebooks/      # Exploratory analysis and plotting notebooks
├── outputs/        # Logs, metrics, and figures produced by experiments (git-ignored if large)
├── paper/          # LaTeX source and figures for the manuscript
├── scripts/        # Convenience shell / batch scripts for running pipelines
├── src/            # Python source for the benchmark and experiments
│   ├── analysis/       # Scripted analyses and figure generation
│   ├── calibration/    # Calibration utilities (TS, OC-TS, visibility proxy, etc.)
│   ├── common/         # Shared helpers (I/O, argument parsing, logging)
│   ├── data/           # OccDet-Calib dataset construction & metadata
│   ├── detectors/      # Detector wrappers (YOLOv8m, FCOS R50-FPN, Deformable DETR R50)
│   ├── experiments/    # Entry points for main experiments (overlap, controls, DZ, baselines)
│   └── metrics/        # F1, DECE, bootstrap confidence intervals
├── yolov8m.pt      # YOLOv8m weights (not tracked in the public repo; see below)
└── README.md       # This file
```

---

## Setup

We recommend Python 3.10+.

```bash
git clone https://github.com/<user>/occdet-calib.git
cd occdet-calib

# Option A: use provided environment
# e.g., environment/conda-env.yml
conda env create -f environment/conda-env.yml
conda activate occdet-calib

# Option B: requirements.txt
pip install -r environment/requirements.txt
```

You will also need:

- MS COCO 2017 validation images and annotations.  
- Detector weights (e.g., YOLOv8m checkpoint). We do not commit large weight files; download them separately and update the corresponding paths in `configs/`.

---

## Building OccDet-Calib

1. Download COCO 2017 val from the official site.
2. Set your COCO paths in `configs/data_paths.yaml` (or similar).
3. Run the data generation script:

```bash
python -m src.data.build_occdet_calib \
  --config configs/build_occdet_calib.yaml
```

This creates the seed instances and the four variant families (overlap, distractor, truncation, inpainting) under `data/occdet_calib/`.

---

## Running experiments

### 1. Clean-data threshold sweeps

```bash
python -m src.experiments.run_clean_sweep \
  --config configs/clean/yolov8m.yaml

python -m src.experiments.run_clean_sweep \
  --config configs/clean/fcos_r50.yaml

python -m src.experiments.run_clean_sweep \
  --config configs/clean/deformable_detr_r50.yaml
```

These scripts log F1/recall curves and store chosen operating thresholds per detector.

### 2. Main overlap benchmark

```bash
python -m src.experiments.run_overlap \
  --config configs/overlap/yolov8m.yaml

python -m src.experiments.run_overlap \
  --config configs/overlap/fcos_r50.yaml

python -m src.experiments.run_overlap \
  --config configs/overlap/deformable_detr_r50.yaml
```

Metrics (F1, DECE, bootstrap intervals) are written into `outputs/metrics/`.

### 3. Controls & inpainting

```bash
python -m src.experiments.run_controls \
  --config configs/controls/yolov8m.yaml
```

Produces results for:

- Overlap vs distractor vs truncation.
- Overlap vs inpainting subset.

### 4. Dangerous Zone analysis

```bash
python -m src.experiments.run_dz \
  --config configs/dz/heldout.yaml
```

Implements the held-out calibration/evaluation split and reports DZ flags across occlusion levels and DECE percentile thresholds.

### 5. Calibration baselines

```bash
python -m src.experiments.run_calibration_baselines \
  --config configs/calibration/baselines.yaml
```

Evaluates:

- Raw detector confidence.
- Global temperature scaling (TS).
- Simple occlusion-conditioned TS (OC-TS).

---

## Reproducing figures

```bash
python -m src.analysis.plot_main_overlap
python -m src.analysis.plot_dz_flags
python -m src.analysis.plot_controls
python -m src.analysis.plot_inpainting
```

Figures are saved under `outputs/figures/` and correspond to the plots in the manuscript.

---

## Citation

If you use OccDet-Calib in your work, please cite:

```bibtex
@article{adari2025occdetcalib,
  title   = {OccDet-Calib: Dangerous Zones in Occluded Object Detection},
  author  = {Adari, Karthik and Uppala, Yojitha and Uppala, Sai Ram},
  journal = {Journal of Visual Communication and Image Representation},
  year    = {2026},
  note    = {submitted}
}
```

---

## License

Clarify code vs data:

- Code: MIT / Apache-2.0 (choose what you prefer).  
- Data: built on MS COCO; see the official COCO license and terms for usage.

---
