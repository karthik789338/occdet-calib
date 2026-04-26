from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compute_dece(conf, correct, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    out = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        out += (mask.sum() / total) * abs(conf[mask].mean() - correct[mask].mean())
    return float(out)


def bootstrap_one(df, n_boot=2000, seed=42, n_bins=10):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(df))
    f1s, deces = [], []
    score = df['score'].to_numpy(dtype=float)
    correct = df['correct'].to_numpy(dtype=float)
    tp_arr = (df['correct'].to_numpy(dtype=int) == 1).astype(int)
    fp_arr = (df['correct'].to_numpy(dtype=int) == 0).astype(int)
    gt_total = int(df['num_ground_truth'].iloc[0]) if 'num_ground_truth' in df.columns else None
    if gt_total is None:
        raise ValueError('Provide num_ground_truth column in matched rows or merge it before running bootstrap.')
    for _ in range(n_boot):
        s = rng.choice(idxs, size=len(df), replace=True)
        tp = tp_arr[s].sum()
        fp = fp_arr[s].sum()
        fn = max(0, gt_total - tp)
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        f1s.append(f1)
        deces.append(compute_dece(score[s], correct[s], n_bins=n_bins))
    return {
        'f1_lo': float(np.percentile(f1s, 2.5)),
        'f1_hi': float(np.percentile(f1s, 97.5)),
        'dece_lo': float(np.percentile(deces, 2.5)),
        'dece_hi': float(np.percentile(deces, 97.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matched_csv', required=True, help='Matched predictions with score, correct, occlusion_level, and num_ground_truth column.')
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.matched_csv)
    rows = []
    for occ, sub in df.groupby('occlusion_level'):
        stats = bootstrap_one(sub, n_boot=args.n_boot, seed=args.seed)
        rows.append({'occlusion_level': occ, **stats, 'n_predictions': len(sub)})
    out = pd.DataFrame(rows).sort_values('occlusion_level')
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out)


if __name__ == '__main__':
    main()
