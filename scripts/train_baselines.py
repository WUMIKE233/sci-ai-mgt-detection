"""Train lightweight MGT detection baselines on a normalized table.

Expected input columns:
  - text: input text
  - label: human/machine or 0/1
Optional analysis columns:
  - domain, generator, perturbation, or any group column supplied by --group-cols

This script is intentionally benchmark-agnostic. Dataset-specific converters
should write CSV/JSONL files with the expected columns, then call this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


POSITIVE_LABELS = {"1", "true", "machine", "mgt", "generated", "ai", "ai_generated", "machine_generated"}
NEGATIVE_LABELS = {"0", "false", "human", "human_written", "human-written", "real", "authentic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF baselines for MGT detection.")
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument("--train", type=Path, help="Training CSV/JSONL file.")
    parser.add_argument("--test", type=Path, help="Test CSV/JSONL file.")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--group-cols", default="domain,generator,perturbation")
    parser.add_argument("--models", default="logreg,linear_svm")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()

    if args.config:
        with args.config.open("r", encoding="utf-8") as f:
            config = json.load(f)
        for key, value in config.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, value)

    args.train = Path(args.train) if args.train else None
    args.test = Path(args.test) if args.test else None
    args.output_dir = Path(args.output_dir)
    if isinstance(args.group_cols, list):
        args.group_cols = [str(col).strip() for col in args.group_cols if str(col).strip()]
    else:
        args.group_cols = [col.strip() for col in str(args.group_cols).split(",") if col.strip()]
    if isinstance(args.models, list):
        args.models = [str(name).strip() for name in args.models if str(name).strip()]
    else:
        args.models = [name.strip() for name in str(args.models).split(",") if name.strip()]
    return args


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_labels(series: pd.Series) -> np.ndarray:
    labels = []
    unknown = set()
    for value in series:
        if pd.isna(value):
            unknown.add("<NA>")
            labels.append(None)
            continue
        if isinstance(value, (int, np.integer)) and value in {0, 1}:
            labels.append(int(value))
            continue
        if isinstance(value, (float, np.floating)) and value in {0.0, 1.0}:
            labels.append(int(value))
            continue
        token = str(value).strip().lower()
        if token in POSITIVE_LABELS:
            labels.append(1)
        elif token in NEGATIVE_LABELS:
            labels.append(0)
        else:
            unknown.add(str(value))
            labels.append(None)
    if unknown:
        raise ValueError(f"Unknown labels in input: {sorted(unknown)}")
    return np.asarray(labels, dtype=int)


def require_columns(df: pd.DataFrame, cols: Iterable[str], path_label: str) -> None:
    missing = [col for col in cols if col and col not in df.columns]
    if missing:
        raise ValueError(f"{path_label} missing required columns: {missing}")


def make_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(int(args.ngram_min), int(args.ngram_max)),
        max_features=int(args.max_features),
        min_df=1,
    )


def make_model(name: str, args: argparse.Namespace) -> Pipeline:
    vectorizer = make_vectorizer(args)
    if name == "logreg":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    elif name == "linear_svm":
        svm = LinearSVC(class_weight="balanced", random_state=42)
        try:
            estimator = CalibratedClassifierCV(estimator=svm, method="sigmoid", cv=3)
        except TypeError:
            estimator = CalibratedClassifierCV(base_estimator=svm, method="sigmoid", cv=3)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("tfidf", vectorizer), ("clf", estimator)])


def binary_ece(y_true: np.ndarray, p_pos: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (p_pos >= left) & (p_pos <= right)
        else:
            mask = (p_pos >= left) & (p_pos < right)
        if not np.any(mask):
            continue
        empirical = float(np.mean(y_true[mask]))
        confidence = float(np.mean(p_pos[mask]))
        ece += float(np.mean(mask)) * abs(empirical - confidence)
    return ece


def safe_metric(fn, *args):
    try:
        value = fn(*args)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    except Exception:
        return None


def compute_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> dict:
    y_pred = (p_pos >= 0.5).astype(int)
    has_both_classes = len(np.unique(y_true)) == 2
    precision, recall, f1_pos, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "macro_f1": safe_metric(lambda a, b: f1_score(a, b, average="macro", zero_division=0), y_true, y_pred),
        "positive_precision": float(precision),
        "positive_recall": float(recall),
        "positive_f1": float(f1_pos),
        "auroc": safe_metric(roc_auc_score, y_true, p_pos) if has_both_classes else None,
        "auprc": safe_metric(average_precision_score, y_true, p_pos) if has_both_classes else None,
        "brier": safe_metric(brier_score_loss, y_true, p_pos),
        "ece_10": binary_ece(y_true, p_pos, n_bins=10),
    }


def selective_curve(y_true: np.ndarray, p_pos: np.ndarray) -> list[dict]:
    y_pred = (p_pos >= 0.5).astype(int)
    confidence = np.maximum(p_pos, 1.0 - p_pos)
    order = np.argsort(-confidence)
    points = []
    for coverage in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        keep = max(1, int(math.ceil(len(order) * coverage)))
        idx = order[:keep]
        risk = float(np.mean(y_pred[idx] != y_true[idx]))
        threshold = float(confidence[idx[-1]])
        points.append({"coverage": coverage, "risk": risk, "confidence_threshold": threshold, "n_kept": int(keep)})
    return points


def evaluate_groups(df: pd.DataFrame, y_true: np.ndarray, p_pos: np.ndarray, group_cols: list[str]) -> list[dict]:
    rows = []
    eval_df = df.copy()
    eval_df["_y_true"] = y_true
    eval_df["_p_pos"] = p_pos
    for col in group_cols:
        if col not in eval_df.columns:
            continue
        for value, part in eval_df.groupby(col, dropna=False):
            metrics = compute_metrics(part["_y_true"].to_numpy(), part["_p_pos"].to_numpy())
            metrics.update({"group_col": col, "group_value": str(value)})
            rows.append(metrics)
    return rows


def main() -> None:
    args = parse_args()
    if not args.train or not args.test:
        raise SystemExit("--train and --test are required, either directly or through --config.")

    train_df = read_table(args.train)
    test_df = read_table(args.test)
    require_columns(train_df, [args.text_col, args.label_col], "train")
    require_columns(test_df, [args.text_col, args.label_col], "test")

    x_train = train_df[args.text_col].fillna("").astype(str).tolist()
    y_train = normalize_labels(train_df[args.label_col])
    x_test = test_df[args.text_col].fillna("").astype(str).tolist()
    y_test = normalize_labels(test_df[args.label_col])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    all_group_rows = []

    prediction_frame = test_df.copy()
    prediction_frame["_label_binary"] = y_test

    for model_name in args.models:
        model = make_model(model_name, args)
        model.fit(x_train, y_train)
        p_pos = model.predict_proba(x_test)[:, 1]
        all_metrics[model_name] = compute_metrics(y_test, p_pos)
        all_metrics[model_name]["selective_curve"] = selective_curve(y_test, p_pos)
        prediction_frame[f"{model_name}_p_machine"] = p_pos
        prediction_frame[f"{model_name}_prediction"] = (p_pos >= 0.5).astype(int)
        for row in evaluate_groups(test_df, y_test, p_pos, args.group_cols):
            row["model"] = model_name
            all_group_rows.append(row)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    prediction_frame.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(all_group_rows).to_csv(args.output_dir / "group_metrics.csv", index=False, encoding="utf-8")

    print(json.dumps({"output_dir": str(args.output_dir), "models": args.models}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
