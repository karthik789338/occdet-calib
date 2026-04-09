#!/usr/bin/env bash
set -euo pipefail

cd ~/occdet-calib
source .venv/bin/activate

mkdir -p ~/occdet-calib/outputs/runs/occdet_v1_overlap_scale5000
mkdir -p ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000

python -m src.experiments.run_clean_baselines \
  --image_dir ~/occdet-calib/data/interim/occdet_v1_overlap_images_5000 \
  --output_csv ~/occdet-calib/outputs/runs/occdet_v1_overlap_scale5000/yolo_overlap_preds.csv \
  --model_config ~/occdet-calib/configs/models/yolo_v8m.yaml \
  --limit 100000

python -m src.experiments.filter_predictions_to_target_classes \
  --pred_path ~/occdet-calib/outputs/runs/occdet_v1_overlap_scale5000/yolo_overlap_preds.csv \
  --classes_yaml ~/occdet-calib/configs/classes.yaml \
  --output_path ~/occdet-calib/outputs/runs/occdet_v1_overlap_scale5000/yolo_overlap_preds_target_only.csv

python -m src.experiments.run_clean_eval_workflow \
  --pred_path ~/occdet-calib/outputs/runs/occdet_v1_overlap_scale5000/yolo_overlap_preds_target_only.csv \
  --gt_path ~/occdet-calib/data/processed/occdet_v1_scale5000/overlap_gt_eval.csv \
  --matched_out ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_matched.csv \
  --per_class_out ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_per_class.csv \
  --per_image_out ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_per_image.csv \
  --reliability_bins_out ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_reliability_bins.csv \
  --iou_threshold 0.5 \
  --classwise

python -m src.experiments.filter_matched_by_score \
  --matched_pred_path ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_matched.csv \
  --output_path ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_matched_thr040.csv \
  --score_threshold 0.4

python -m src.experiments.summarize_by_occlusion_level \
  --matched_pred_path ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_matched_thr040.csv \
  --gt_path ~/occdet-calib/data/processed/occdet_v1_scale5000/overlap_gt_eval.csv \
  --output_path ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_by_occlusion_thr040.csv

python -m src.experiments.summarize_dece_by_occlusion \
  --matched_pred_path ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_matched_thr040.csv \
  --gt_path ~/occdet-calib/data/processed/occdet_v1_scale5000/overlap_gt_eval.csv \
  --output_path ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_dece_by_occlusion_thr040.csv

python -m src.experiments.summarize_by_occluder_semantics \
  --matched_pred_path ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_matched_thr040.csv \
  --gt_path ~/occdet-calib/data/processed/occdet_v1_scale5000/overlap_gt_eval.csv \
  --output_path ~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_by_semantics_thr040.csv

echo
echo "=== DONE: scale5000 YOLO overlap pipeline ==="
echo "Key outputs:"
echo "~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_by_occlusion_thr040.csv"
echo "~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_dece_by_occlusion_thr040.csv"
echo "~/occdet-calib/outputs/evals/occdet_v1_overlap_scale5000/yolo_overlap_by_semantics_thr040.csv"
