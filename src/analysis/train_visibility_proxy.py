from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    parser = argparse.ArgumentParser(description="Train lightweight visibility proxy model.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv).copy()

    features_num = ["score", "box_w", "box_h", "box_area", "aspect_ratio"]
    features_cat = ["model_name", "class_id", "class_name"]
    target = "visibility_bucket"

    X = df[features_num + features_cat]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", numeric_pipe, features_num),
        ("cat", categorical_pipe, features_cat),
    ])

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=200,
        random_state=args.random_seed,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\nConfusion matrix:\n")
        f.write(str(cm))

    print(report)
    print("Confusion matrix:")
    print(cm)
    print("Saved to:", out_dir / "classification_report.txt")


if __name__ == "__main__":
    main()
