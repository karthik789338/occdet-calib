from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def fmt_range(series):
    uniq = sorted(series.unique())
    if len(uniq) == 1:
        return str(int(uniq[0])) if float(uniq[0]).is_integer() else str(uniq[0])
    return f"{int(min(uniq))}--{int(max(uniq))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master_results_csv', required=True)
    ap.add_argument('--inpainting_perf_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    args = ap.parse_args()

    master = pd.read_csv(args.master_results_csv)
    inpaint = pd.read_csv(args.inpainting_perf_csv)
    rows = []
    for exp, model in [('overlap_main','YOLOv8m'),('overlap_main','FCOS-R50'),('overlap_main','Deformable-DETR-R50'),('distractor_control','YOLOv8m'),('truncation_control','YOLOv8m')]:
        sub = master[(master['experiment']==exp) & (master['model']==model)]
        if len(sub)==0: continue
        rows.append({'variant':exp,'model':model,'gt_instances_per_level':fmt_range(sub['num_ground_truth']),'occlusion_levels':','.join(map(lambda x:f'{x:.1f}', sorted(sub['occlusion_level'].unique()))),'total_predictions':int(sub['num_predictions'].sum())})
    sub = inpaint[inpaint['condition']=='inpainting']
    rows.append({'variant':'inpainting_subset','model':'YOLOv8m','gt_instances_per_level':fmt_range(sub['num_ground_truth']),'occlusion_levels':','.join(map(lambda x:f'{x:.1f}', sorted(sub['occlusion_level'].unique()))),'total_predictions':int(sub['num_predictions'].sum())})
    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out)


if __name__ == '__main__':
    main()
