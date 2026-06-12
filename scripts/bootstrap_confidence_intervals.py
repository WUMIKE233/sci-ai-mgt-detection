"""Bootstrap confidence intervals for preliminary experiment predictions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


RUNS = [
    ("MAGE small", "TF-IDF baselines", "logreg", "outputs/mage_small_baseline/predictions.csv", "logreg_p_machine"),
    ("MAGE small", "TF-IDF baselines", "linear_svm", "outputs/mage_small_baseline/predictions.csv", "linear_svm_p_machine"),
    ("MAGE small", "Calibration/selective", "identity", "outputs/mage_small_calibrated_selective/predictions.csv", "identity_p_machine"),
    ("MAGE small", "Calibration/selective", "temperature", "outputs/mage_small_calibrated_selective/predictions.csv", "temperature_p_machine"),
    ("MAGE small", "Calibration/selective", "isotonic", "outputs/mage_small_calibrated_selective/predictions.csv", "isotonic_p_machine"),
    ("MAGE small", "Frozen Transformer probe", "distilbert_frozen_probe", "outputs/mage_small_transformer_probe/predictions.csv", "transformer_probe_p_machine"),
    ("MAGE small", "Fine-tuned Transformer", "distilbert_finetuned", "outputs/mage_small_transformer_finetune/predictions.csv", "transformer_finetune_p_machine"),
    ("RAID small", "TF-IDF baselines", "logreg", "outputs/raid_small_baseline/predictions.csv", "logreg_p_machine"),
    ("RAID small", "TF-IDF baselines", "linear_svm", "outputs/raid_small_baseline/predictions.csv", "linear_svm_p_machine"),
    ("RAID small", "Calibration/selective", "identity", "outputs/raid_small_calibrated_selective/predictions.csv", "identity_p_machine"),
    ("RAID small", "Calibration/selective", "temperature", "outputs/raid_small_calibrated_selective/predictions.csv", "temperature_p_machine"),
    ("RAID small", "Calibration/selective", "isotonic", "outputs/raid_small_calibrated_selective/predictions.csv", "isotonic_p_machine"),
    ("RAID small", "Frozen Transformer probe", "distilbert_frozen_probe", "outputs/raid_small_transformer_probe/predictions.csv", "transformer_probe_p_machine"),
    ("RAID small", "Fine-tuned Transformer", "distilbert_finetuned", "outputs/raid_small_transformer_finetune/predictions.csv", "transformer_finetune_p_machine"),
    ("SemEval Subtask A mono small", "TF-IDF baselines", "logreg", "outputs/semeval_subtaskA_mono_small_baseline/predictions.csv", "logreg_p_machine"),
    ("SemEval Subtask A mono small", "TF-IDF baselines", "linear_svm", "outputs/semeval_subtaskA_mono_small_baseline/predictions.csv", "linear_svm_p_machine"),
    ("SemEval Subtask A mono small", "Calibration/selective", "identity", "outputs/semeval_subtaskA_mono_small_calibrated_selective/predictions.csv", "identity_p_machine"),
    ("SemEval Subtask A mono small", "Calibration/selective", "temperature", "outputs/semeval_subtaskA_mono_small_calibrated_selective/predictions.csv", "temperature_p_machine"),
    ("SemEval Subtask A mono small", "Calibration/selective", "isotonic", "outputs/semeval_subtaskA_mono_small_calibrated_selective/predictions.csv", "isotonic_p_machine"),
    ("SemEval Subtask A mono small", "Frozen Transformer probe", "distilbert_frozen_probe", "outputs/semeval_subtaskA_mono_small_transformer_probe/predictions.csv", "transformer_probe_p_machine"),
    ("SemEval Subtask A mono small", "Fine-tuned Transformer", "distilbert_finetuned", "outputs/semeval_subtaskA_mono_small_transformer_finetune/predictions.csv", "transformer_finetune_p_machine"),
]


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
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def compute_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> dict[str, float]:
    y_pred = (p_pos >= 0.5).astype(int)
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "macro_f1": safe_metric(lambda a, b: f1_score(a, b, average="macro", zero_division=0), y_true, y_pred),
        "auroc": safe_metric(roc_auc_score, y_true, p_pos) if has_both_classes else np.nan,
        "auprc": safe_metric(average_precision_score, y_true, p_pos) if has_both_classes else np.nan,
        "brier": safe_metric(brier_score_loss, y_true, p_pos),
        "ece_10": binary_ece(y_true, p_pos, n_bins=10),
    }


def bootstrap_ci(y_true: np.ndarray, p_pos: np.ndarray, n_bootstrap: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    point = compute_metrics(y_true, p_pos)
    samples = {name: [] for name in point.keys()}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        metrics = compute_metrics(y_true[idx], p_pos[idx])
        for name, value in metrics.items():
            if not np.isnan(value):
                samples[name].append(value)
    rows = []
    for name, point_value in point.items():
        values = np.asarray(samples[name], dtype=float)
        if len(values) == 0 or np.isnan(point_value):
            rows.append({"metric": name, "point": point_value, "ci_low": np.nan, "ci_high": np.nan, "bootstrap_n": 0})
            continue
        low, high = np.percentile(values, [2.5, 97.5])
        rows.append({"metric": name, "point": float(point_value), "ci_low": float(low), "ci_high": float(high), "bootstrap_n": int(len(values))})
    return rows


def fmt_interval(point, low, high) -> str:
    if pd.isna(point):
        return ""
    if pd.isna(low) or pd.isna(high):
        return f"{point:.4f}"
    return f"{point:.4f} [{low:.4f}, {high:.4f}]"


def main() -> None:
    output_dir = Path("outputs/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_bootstrap = 1000
    seed = 2026
    for dataset, run, model, path, prob_col in RUNS:
        source = Path(path)
        if not source.exists():
            continue
        df = pd.read_csv(source)
        if "_label_binary" not in df.columns or prob_col not in df.columns:
            continue
        y_true = df["_label_binary"].to_numpy(dtype=int)
        p_pos = df[prob_col].to_numpy(dtype=float)
        for row in bootstrap_ci(y_true, p_pos, n_bootstrap=n_bootstrap, seed=seed):
            row.update({"dataset": dataset, "run": run, "model": model, "n": int(len(df)), "source": path, "probability_column": prob_col})
            rows.append(row)

    ci_df = pd.DataFrame(rows)
    csv_path = output_dir / "bootstrap_confidence_intervals.csv"
    ci_df.to_csv(csv_path, index=False, encoding="utf-8")

    md_path = Path("docs/bootstrap_confidence_intervals.md")
    metrics = ["accuracy", "macro_f1", "auroc", "brier", "ece_10"]
    lines = [
        "# Bootstrap Confidence Intervals",
        "",
        "This report gives 95% bootstrap confidence intervals for current preliminary predictions. These intervals quantify uncertainty within the sampled test sets; they do not fix the larger limitation that the datasets are still small sampled subsets.",
        "",
        f"Bootstrap resamples per run: `{n_bootstrap}`",
        f"CSV source: `{csv_path}`",
        "",
        "| Dataset | Run | Model | N | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (dataset, run, model, n), part in ci_df.groupby(["dataset", "run", "model", "n"], sort=False):
        lookup = {row["metric"]: row for _, row in part.iterrows()}
        cells = []
        for metric in metrics:
            row = lookup.get(metric)
            if row is None:
                cells.append("")
            else:
                cells.append(fmt_interval(row["point"], row["ci_low"], row["ci_high"]))
        lines.append(f"| {dataset} | {run} | {model} | {n} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- These intervals are conditional on the current sampled test sets.",
            "- They do not replace repeated seeds, larger stratified samples, or full benchmark processing.",
            "- Final manuscript tables should regenerate this file from locked predictions.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "rows": len(ci_df)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
