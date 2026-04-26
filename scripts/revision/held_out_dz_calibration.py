from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compute_delta(df: pd.DataFrame, model: str, percentile: float) -> float:
    baseline = df[(df["model"] == model) & (df["occlusion_level"] == 0.0)]["dece"]
    if len(baseline) == 0:
        raise ValueError(f"No clean baseline rows for model={model}")
    return float(np.percentile(baseline, percentile))


def dangerous_zone_flag(row, delta, recall_floor, precision_floor):
    return (row["recall"] >= recall_floor) and (row["precision"] >= precision_floor) and (row["dece"] >= delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Per-image or per-seed DZ source rows with columns: split_id/model/occlusion_level/recall/precision/dece")
    ap.add_argument("--split_col", default="split_id")
    ap.add_argument("--calibration_value", default="calib")
    ap.add_argument("--evaluation_value", default="eval")
    ap.add_argument("--percentiles", nargs='+', type=float, default=[50,60,70,75,80,85,90])
    ap.add_argument("--recall_floor", type=float, default=0.60)
    ap.add_argument("--precision_floor", type=float, default=0.15)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_calib = df[df[args.split_col] == args.calibration_value].copy()
    df_eval = df[df[args.split_col] == args.evaluation_value].copy()

    sensitivity_rows = []
    eval_rows = []
    for pct in args.percentiles:
        delta_map = {m: compute_delta(df_calib, m, pct) for m in sorted(df["model"].unique())}
        tmp = df_eval.copy()
        tmp["delta_m"] = tmp["model"].map(delta_map)
        tmp["dangerous_zone"] = tmp.apply(lambda r: dangerous_zone_flag(r, r["delta_m"], args.recall_floor, args.precision_floor), axis=1)
        tmp["percentile"] = pct
        eval_rows.append(tmp)
        for model, delta in delta_map.items():
            dz_levels = tmp[(tmp["model"] == model) & (tmp["dangerous_zone"] == True)]["occlusion_level"].tolist()
            sensitivity_rows.append({"model": model, "percentile": pct, "delta_m": delta, "dz_levels": dz_levels})

    pd.concat(eval_rows, ignore_index=True).to_csv(outdir / 'heldout_dz_eval_rows.csv', index=False)
    pd.DataFrame(sensitivity_rows).to_csv(outdir / 'heldout_dz_sensitivity.csv', index=False)
    print('saved', outdir / 'heldout_dz_eval_rows.csv')
    print('saved', outdir / 'heldout_dz_sensitivity.csv')


if __name__ == "__main__":
    main()
