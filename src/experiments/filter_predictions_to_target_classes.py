from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_target_classes(classes_yaml_path: str) -> list[str]:
    with open(classes_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg.get("target_classes", None)
    if classes is None:
        raise KeyError("Expected key 'target_classes' in classes YAML.")

    return [str(x).strip().lower() for x in classes]


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)

    raise ValueError(f"Unsupported file type: {p.suffix}")


def save_table(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    elif p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {p.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter predictions to target classes.")
    parser.add_argument("--pred_path", type=str, required=True, help="Input predictions CSV/Parquet")
    parser.add_argument("--classes_yaml", type=str, required=True, help="Path to configs/classes.yaml")
    parser.add_argument("--output_path", type=str, required=True, help="Filtered predictions output")
    args = parser.parse_args()

    df = load_table(args.pred_path)
    if "class_name" not in df.columns:
        raise KeyError("Expected prediction table to contain 'class_name'.")

    target_classes = set(load_target_classes(args.classes_yaml))

    out = df.copy()
    out["class_name_norm"] = out["class_name"].astype(str).str.strip().str.lower()
    out = out[out["class_name_norm"].isin(target_classes)].copy()
    out = out.drop(columns=["class_name_norm"])

    save_table(out, args.output_path)

    print(f"Saved filtered predictions to: {args.output_path}")
    print(f"Rows written: {len(out)}")
    print(f"Classes kept: {sorted(out['class_name'].dropna().astype(str).unique().tolist()) if len(out) else []}")


if __name__ == "__main__":
    main()
