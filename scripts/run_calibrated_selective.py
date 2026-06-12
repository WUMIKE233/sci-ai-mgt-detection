"""Run post-hoc calibration and selective prediction for MGT detection.

The script trains a TF-IDF logistic-regression detector on a fit split, learns
post-hoc calibration on a held-out calibration split, and evaluates calibrated
probabilities plus selective prediction on a separate test set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


POSITIVE_LABELS = {"1", "true", "machine", "mgt", "generated", "ai", "ai_generated", "machine_generated"}
NEGATIVE_LABELS = {"0", "false", "human", "human_written", "human-written", "real", "authentic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibration and selective prediction.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--group-cols", default="domain,generator,perturbation")
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--dataset-label", default="test subset")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/calibrated_selective"))
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


def make_model(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(int(args.ngram_min), int(args.ngram_max)),
                    max_features=int(args.max_features),
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=args.seed),
            ),
        ]
    )


def clip_probs(p_pos: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p_pos, dtype=float), 1e-6, 1.0 - 1e-6)


def logits_from_probs(p_pos: np.ndarray) -> np.ndarray:
    clipped = clip_probs(p_pos)
    return np.log(clipped / (1.0 - clipped))


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def fit_temperature(p_calib: np.ndarray, y_calib: np.ndarray) -> float:
    logits = logits_from_probs(p_calib)

    def nll(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        p = clip_probs(probs_from_logits(logits / temperature))
        return -float(np.mean(y_calib * np.log(p) + (1 - y_calib) * np.log(1 - p)))

    result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(p_pos: np.ndarray, temperature: float) -> np.ndarray:
    return clip_probs(probs_from_logits(logits_from_probs(p_pos) / temperature))


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
    p_pos = clip_probs(p_pos)
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
    p_pos = clip_probs(p_pos)
    y_pred = (p_pos >= 0.5).astype(int)
    confidence = np.maximum(p_pos, 1.0 - p_pos)
    order = np.argsort(-confidence)
    points = []
    for coverage in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        keep = max(1, int(math.ceil(len(order) * coverage)))
        idx = order[:keep]
        risk = float(np.mean(y_pred[idx] != y_true[idx]))
        points.append(
            {
                "coverage": coverage,
                "risk": risk,
                "confidence_threshold": float(confidence[idx[-1]]),
                "n_kept": int(keep),
            }
        )
    return points


def reliability_bins(y_true: np.ndarray, p_pos: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        if right == 1.0:
            mask = (p_pos >= left) & (p_pos <= right)
        else:
            mask = (p_pos >= left) & (p_pos < right)
        if not np.any(mask):
            rows.append({"bin": idx, "left": float(left), "right": float(right), "n": 0, "confidence": None, "accuracy": None})
            continue
        rows.append(
            {
                "bin": idx,
                "left": float(left),
                "right": float(right),
                "n": int(np.sum(mask)),
                "confidence": float(np.mean(p_pos[mask])),
                "accuracy": float(np.mean(y_true[mask])),
            }
        )
    return rows


def conformal_summary(y_calib: np.ndarray, p_calib: np.ndarray, y_test: np.ndarray, p_test: np.ndarray) -> list[dict]:
    p_calib = clip_probs(p_calib)
    p_test = clip_probs(p_test)
    calib_true_probs = np.where(y_calib == 1, p_calib, 1.0 - p_calib)
    scores = 1.0 - calib_true_probs
    rows = []
    for alpha in [0.05, 0.1, 0.2]:
        quantile = min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
        qhat = float(np.quantile(scores, quantile, method="higher"))
        sets = []
        for p in p_test:
            labels = []
            if 1.0 - (1.0 - p) <= qhat:
                labels.append(0)
            if 1.0 - p <= qhat:
                labels.append(1)
            sets.append(labels)
        contains_true = [int(y in labels) for y, labels in zip(y_test, sets)]
        singleton_mask = np.asarray([len(labels) == 1 for labels in sets], dtype=bool)
        singleton_preds = np.asarray([labels[0] if len(labels) == 1 else -1 for labels in sets], dtype=int)
        if np.any(singleton_mask):
            singleton_risk = float(np.mean(singleton_preds[singleton_mask] != y_test[singleton_mask]))
        else:
            singleton_risk = None
        rows.append(
            {
                "alpha": alpha,
                "qhat": qhat,
                "empirical_coverage": float(np.mean(contains_true)),
                "singleton_rate": float(np.mean(singleton_mask)),
                "abstain_or_ambiguous_rate": float(1.0 - np.mean(singleton_mask)),
                "risk_on_singletons": singleton_risk,
            }
        )
    return rows


def evaluate_groups(df: pd.DataFrame, y_true: np.ndarray, p_pos: np.ndarray, group_cols: list[str], method: str) -> list[dict]:
    rows = []
    eval_df = df.copy()
    eval_df["_y_true"] = y_true
    eval_df["_p_pos"] = p_pos
    for col in group_cols:
        if col not in eval_df.columns:
            continue
        for value, part in eval_df.groupby(col, dropna=False):
            metrics = compute_metrics(part["_y_true"].to_numpy(), part["_p_pos"].to_numpy())
            metrics.update({"method": method, "group_col": col, "group_value": str(value)})
            rows.append(metrics)
    return rows


def plot_reliability(bins_by_method: dict[str, list[dict]], output_path: Path, dataset_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1.0, label="Perfect calibration")
    for method, rows in bins_by_method.items():
        xs = [row["confidence"] for row in rows if row["confidence"] is not None]
        ys = [row["accuracy"] for row in rows if row["accuracy"] is not None]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=method)
    ax.set_xlabel("Mean predicted probability of machine text")
    ax.set_ylabel("Empirical machine-text frequency")
    ax.set_title(f"Reliability diagram on {dataset_label}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_coverage_risk(curves_by_method: dict[str, list[dict]], output_path: Path, dataset_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for method, rows in curves_by_method.items():
        xs = [row["coverage"] for row in rows]
        ys = [row["risk"] for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=method)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title(f"Coverage-risk curves on {dataset_label}")
    ax.set_xlim(0.48, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(metrics: dict, output_path: Path, config: dict) -> None:
    lines = [
        "# Calibrated Selective Prediction Smoke Report",
        "",
        f"This report is a real-data pipeline validation on {config['dataset_label']}. It is not a final manuscript result.",
        "",
        "## Configuration",
        "",
        f"- Train file: `{config['train']}`",
        f"- Test file: `{config['test']}`",
        f"- Calibration split size: `{config['calibration_size']}`",
        f"- Random seed: `{config['seed']}`",
        f"- Dataset label: `{config['dataset_label']}`",
        "",
        "## Test Metrics",
        "",
        "| Method | Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in metrics["methods"].items():
        lines.append(
            "| {method} | {accuracy:.4f} | {macro_f1:.4f} | {auroc:.4f} | {auprc:.4f} | {brier:.4f} | {ece:.4f} |".format(
                method=method,
                accuracy=values["accuracy"],
                macro_f1=values["macro_f1"],
                auroc=values["auroc"],
                auprc=values["auprc"],
                brier=values["brier"],
                ece=values["ece_10"],
            )
        )
    lines.extend(
        [
            "",
            "## Conformal Selective Summary",
            "",
            "| Method | Alpha | Empirical coverage | Singleton rate | Ambiguous/abstain rate | Risk on singletons |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, rows in metrics["conformal"].items():
        for row in rows:
            risk = "" if row["risk_on_singletons"] is None else f"{row['risk_on_singletons']:.4f}"
            lines.append(
                f"| {method} | {row['alpha']:.2f} | {row['empirical_coverage']:.4f} | {row['singleton_rate']:.4f} | {row['abstain_or_ambiguous_rate']:.4f} | {risk} |"
            )
    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            "- `reliability_diagram.png`",
            "- `coverage_risk.png`",
            "",
            "## Boundary",
            "",
            "This run uses a small sampled subset and a TF-IDF logistic-regression detector. Full-scale experiments and stronger baselines remain required before manuscript submission.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.train or not args.test:
        raise SystemExit("--train and --test are required, either directly or through --config.")

    train_df = read_table(args.train)
    test_df = read_table(args.test)
    require_columns(train_df, [args.text_col, args.label_col], "train")
    require_columns(test_df, [args.text_col, args.label_col], "test")

    y_all = normalize_labels(train_df[args.label_col])
    fit_df, calib_df, y_fit, y_calib = train_test_split(
        train_df,
        y_all,
        test_size=float(args.calibration_size),
        random_state=int(args.seed),
        stratify=y_all,
    )
    y_test = normalize_labels(test_df[args.label_col])

    model = make_model(args)
    model.fit(fit_df[args.text_col].fillna("").astype(str).tolist(), y_fit)

    p_calib_identity = clip_probs(model.predict_proba(calib_df[args.text_col].fillna("").astype(str).tolist())[:, 1])
    p_test_identity = clip_probs(model.predict_proba(test_df[args.text_col].fillna("").astype(str).tolist())[:, 1])

    temperature = fit_temperature(p_calib_identity, y_calib)
    p_calib_temperature = apply_temperature(p_calib_identity, temperature)
    p_test_temperature = apply_temperature(p_test_identity, temperature)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_calib_identity, y_calib)
    p_calib_isotonic = clip_probs(isotonic.predict(p_calib_identity))
    p_test_isotonic = clip_probs(isotonic.predict(p_test_identity))

    methods = {
        "identity": (p_calib_identity, p_test_identity),
        "temperature": (p_calib_temperature, p_test_temperature),
        "isotonic": (p_calib_isotonic, p_test_isotonic),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_df.copy()
    predictions["_label_binary"] = y_test
    metrics = {
        "config": {
            "train": str(args.train),
            "test": str(args.test),
            "calibration_size": float(args.calibration_size),
            "seed": int(args.seed),
            "fit_rows": int(len(fit_df)),
            "calibration_rows": int(len(calib_df)),
            "test_rows": int(len(test_df)),
            "dataset_label": str(args.dataset_label),
        },
        "temperature": temperature,
        "methods": {},
        "conformal": {},
    }
    reliability_by_method = {}
    curves_by_method = {}
    group_rows = []
    reliability_rows = []
    curve_rows = []

    for method, (p_calib, p_test) in methods.items():
        predictions[f"{method}_p_machine"] = p_test
        predictions[f"{method}_prediction"] = (p_test >= 0.5).astype(int)
        method_metrics = compute_metrics(y_test, p_test)
        method_curve = selective_curve(y_test, p_test)
        method_bins = reliability_bins(y_test, p_test, n_bins=10)
        method_metrics["selective_curve"] = method_curve
        metrics["methods"][method] = method_metrics
        metrics["conformal"][method] = conformal_summary(y_calib, p_calib, y_test, p_test)
        reliability_by_method[method] = method_bins
        curves_by_method[method] = method_curve

        for row in evaluate_groups(test_df, y_test, p_test, args.group_cols, method):
            group_rows.append(row)
        for row in method_bins:
            row["method"] = method
            reliability_rows.append(row)
        for row in method_curve:
            row["method"] = method
            curve_rows.append(row)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(group_rows).to_csv(args.output_dir / "group_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(reliability_rows).to_csv(args.output_dir / "reliability_bins.csv", index=False, encoding="utf-8")
    pd.DataFrame(curve_rows).to_csv(args.output_dir / "coverage_risk.csv", index=False, encoding="utf-8")

    plot_reliability(reliability_by_method, args.output_dir / "reliability_diagram.png", str(args.dataset_label))
    plot_coverage_risk(curves_by_method, args.output_dir / "coverage_risk.png", str(args.dataset_label))
    write_markdown_report(metrics, args.output_dir / "calibration_report.md", metrics["config"])

    print(json.dumps({"output_dir": str(args.output_dir), "temperature": temperature}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
