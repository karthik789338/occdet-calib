#!/usr/bin/env bash
set -euo pipefail

ROOT=~/occdet-calib
REL=$ROOT/release_package

mkdir -p $REL/code
mkdir -p $REL/artifacts/{metadata,evals,analysis,phase1,phase2,phase4,paper_assets}
mkdir -p $REL/samples
mkdir -p $REL/docs

echo "Copying code/config/docs..."

cp -r $ROOT/src $REL/code/
cp -r $ROOT/configs $REL/code/
cp -r $ROOT/scripts $REL/code/
cp $ROOT/README.md $REL/code/ 2>/dev/null || true
cp $ROOT/.gitignore $REL/code/ 2>/dev/null || true

echo "Copying metadata..."
cp $ROOT/data/processed/occdet_v1_scale500/overlap_metadata.csv $REL/artifacts/metadata/ 2>/dev/null || true
cp $ROOT/data/processed/occdet_v1_scale500/overlap_gt_eval.csv $REL/artifacts/metadata/ 2>/dev/null || true
cp $ROOT/data/processed/occdet_v1_distractor_scale500/distractor_metadata.csv $REL/artifacts/metadata/ 2>/dev/null || true
cp $ROOT/data/processed/occdet_v1_distractor_scale500/distractor_gt_eval.csv $REL/artifacts/metadata/ 2>/dev/null || true
cp $ROOT/data/processed/occdet_v1_truncation_scale500/truncation_metadata.csv $REL/artifacts/metadata/ 2>/dev/null || true
cp $ROOT/data/processed/occdet_v1_truncation_scale500/truncation_gt_eval.csv $REL/artifacts/metadata/ 2>/dev/null || true
cp $ROOT/data/processed/occdet_v1_inpainting_scale500/inpainting_metadata.csv $REL/artifacts/metadata/ 2>/dev/null || true
cp $ROOT/data/processed/occdet_v1_inpainting_scale500/inpainting_gt_eval.csv $REL/artifacts/metadata/ 2>/dev/null || true
cp $ROOT/data/processed/bdd100k_natural/manifest.csv $REL/artifacts/metadata/ 2>/dev/null || true
cp $ROOT/data/processed/bdd100k_natural/gt_eval.csv $REL/artifacts/metadata/ 2>/dev/null || true

echo "Copying core analysis outputs..."
cp $ROOT/outputs/analysis/master_results_table.csv $REL/artifacts/analysis/ 2>/dev/null || true
cp $ROOT/outputs/analysis/master_dece_table.csv $REL/artifacts/analysis/ 2>/dev/null || true
cp $ROOT/outputs/analysis/dangerous_zone_summary.csv $REL/artifacts/analysis/ 2>/dev/null || true

echo "Copying phase outputs..."
cp -r $ROOT/outputs/phase1 $REL/artifacts/ 2>/dev/null || true
cp -r $ROOT/outputs/phase2 $REL/artifacts/ 2>/dev/null || true
cp -r $ROOT/outputs/phase4 $REL/artifacts/ 2>/dev/null || true

echo "Copying selected eval summaries..."
mkdir -p $REL/artifacts/evals
cp $ROOT/outputs/evals/occdet_v1_inpainting_scale500/yolo_overlap_vs_inpainting_perf.csv $REL/artifacts/evals/ 2>/dev/null || true
cp $ROOT/outputs/evals/occdet_v1_inpainting_scale500/yolo_overlap_vs_inpainting_dece.csv $REL/artifacts/evals/ 2>/dev/null || true
cp $ROOT/outputs/evals/bdd100k_natural/yolo_by_natural_group.csv $REL/artifacts/evals/ 2>/dev/null || true
cp $ROOT/outputs/evals/bdd100k_natural/fcos_by_natural_group.csv $REL/artifacts/evals/ 2>/dev/null || true
cp $ROOT/outputs/evals/bdd100k_natural/deformable_detr_by_natural_group.csv $REL/artifacts/evals/ 2>/dev/null || true

echo "Copying paper assets..."
cp -r $ROOT/paper_assets/* $REL/artifacts/paper_assets/ 2>/dev/null || true

echo "Copying sample images..."
cp $ROOT/paper_assets/montage/overlap_montage.png $REL/samples/ 2>/dev/null || true
cp $ROOT/paper_assets/montage/truncation_montage.png $REL/samples/ 2>/dev/null || true

# a few sample synthetic images if they exist
find $ROOT/data/interim/occdet_v1_overlap_images_500 -maxdepth 1 -type f | sort | head -n 5 | while read f; do cp "$f" $REL/samples/; done 2>/dev/null || true
find $ROOT/data/interim/occdet_v1_truncation_images_500 -maxdepth 1 -type f | sort | head -n 5 | while read f; do cp "$f" $REL/samples/; done 2>/dev/null || true
find $ROOT/data/interim/occdet_v1_distractor_images_500 -maxdepth 1 -type f | sort | head -n 5 | while read f; do cp "$f" $REL/samples/; done 2>/dev/null || true

echo "Writing release tree..."
find $REL | sort > $REL/docs/release_file_list.txt

echo "Creating tarball..."
tar -czf $ROOT/release_package.tar.gz -C $ROOT release_package

echo "Done."
echo "Release folder: $REL"
echo "Tarball: $ROOT/release_package.tar.gz"
