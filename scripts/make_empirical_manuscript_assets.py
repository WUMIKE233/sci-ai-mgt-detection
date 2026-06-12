"""Generate locked empirical manuscript tables, figures, and summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 2026

RUNS = [
    {
        "dataset": "SemEval-A monolingual full",
        "protocol": "official train-dev",
        "path": "outputs/locked/semeval_mono_full_baseline/predictions.csv",
        "prob_col": "logreg_p_machine",
        "method": "TF-IDF LR",
        "group_cols": ["domain", "generator", "original_source", "original_model"],
    },
    {
        "dataset": "SemEval-A multilingual full",
        "protocol": "official train-dev",
        "path": "outputs/locked/semeval_multi_full_baseline/predictions.csv",
        "prob_col": "logreg_p_machine",
        "method": "TF-IDF LR",
        "group_cols": ["domain", "generator", "original_source", "original_model"],
    },
    {
        "dataset": "MAGE in-distribution full",
        "protocol": "MAGE train-test",
        "path": "outputs/locked/mage_id_full_baseline/predictions.csv",
        "prob_col": "logreg_p_machine",
        "method": "TF-IDF LR",
        "group_cols": ["domain", "generator", "perturbation", "original_src"],
    },
    {
        "dataset": "MAGE GPT-4 OOD full",
        "protocol": "MAGE train to OOD GPT-4",
        "path": "outputs/locked/mage_ood_gpt_full_baseline/predictions.csv",
        "prob_col": "logreg_p_machine",
        "method": "TF-IDF LR",
        "group_cols": ["domain", "generator", "perturbation", "original_src"],
    },
    {
        "dataset": "MAGE paraphrase OOD full",
        "protocol": "MAGE train to paraphrase OOD",
        "path": "outputs/locked/mage_ood_para_full_baseline/predictions.csv",
        "prob_col": "logreg_p_machine",
        "method": "TF-IDF LR",
        "group_cols": ["domain", "generator", "perturbation", "original_src"],
    },
]

CALIBRATED_RUNS = [
    {
        "dataset": "SemEval-A monolingual full",
        "protocol": "fit-calibration-dev",
        "path": "outputs/locked/semeval_mono_full_calibrated/predictions.csv",
        "metrics": "outputs/locked/semeval_mono_full_calibrated/metrics.json",
        "group_cols": ["domain", "generator", "original_source", "original_model"],
    },
    {
        "dataset": "SemEval-A multilingual full",
        "protocol": "fit-calibration-dev",
        "path": "outputs/locked/semeval_multi_full_calibrated/predictions.csv",
        "metrics": "outputs/locked/semeval_multi_full_calibrated/metrics.json",
        "group_cols": ["domain", "generator", "original_source", "original_model"],
    },
    {
        "dataset": "MAGE in-distribution full",
        "protocol": "fit-calibration-test",
        "path": "outputs/locked/mage_id_full_calibrated/predictions.csv",
        "metrics": "outputs/locked/mage_id_full_calibrated/metrics.json",
        "group_cols": ["domain", "generator", "perturbation", "original_src"],
    },
    {
        "dataset": "MAGE GPT-4 OOD full",
        "protocol": "fit-calibration-OOD",
        "path": "outputs/locked/mage_ood_gpt_full_calibrated/predictions.csv",
        "metrics": "outputs/locked/mage_ood_gpt_full_calibrated/metrics.json",
        "group_cols": ["domain", "generator", "perturbation", "original_src"],
    },
    {
        "dataset": "MAGE paraphrase OOD full",
        "protocol": "fit-calibration-OOD",
        "path": "outputs/locked/mage_ood_para_full_calibrated/predictions.csv",
        "metrics": "outputs/locked/mage_ood_para_full_calibrated/metrics.json",
        "group_cols": ["domain", "generator", "perturbation", "original_src"],
    },
]

CALIBRATION_METHODS = [
    ("identity", "Uncalibrated LR"),
    ("temperature", "Temp-scaled LR"),
    ("isotonic", "Isotonic LR"),
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


def safe_metric(fn, *args) -> float:
    try:
        value = fn(*args)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> dict[str, float]:
    y_pred = (p_pos >= 0.5).astype(int)
    has_both = len(np.unique(y_true)) == 2
    return {
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "macro_f1": safe_metric(lambda a, b: f1_score(a, b, average="macro", zero_division=0), y_true, y_pred),
        "auroc": safe_metric(roc_auc_score, y_true, p_pos) if has_both else float("nan"),
        "auprc": safe_metric(average_precision_score, y_true, p_pos) if has_both else float("nan"),
        "brier": safe_metric(brier_score_loss, y_true, p_pos),
        "ece_10": binary_ece(y_true, p_pos, n_bins=10),
    }


def bootstrap_ci(y_true: np.ndarray, p_pos: np.ndarray) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    point = compute_metrics(y_true, p_pos)
    samples = {name: [] for name in point}
    n = len(y_true)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        metrics = compute_metrics(y_true[idx], p_pos[idx])
        for name, value in metrics.items():
            if not np.isnan(value):
                samples[name].append(value)
    result = {}
    for name, point_value in point.items():
        values = np.asarray(samples[name], dtype=float)
        if len(values) == 0 or np.isnan(point_value):
            result[name] = (point_value, float("nan"), float("nan"))
            continue
        low, high = np.percentile(values, [2.5, 97.5])
        result[name] = (float(point_value), float(low), float(high))
    return result


def fmt_ci(point: float, low: float, high: float) -> str:
    if pd.isna(point):
        return ""
    if pd.isna(low) or pd.isna(high):
        return f"{point:.4f}"
    return f"{point:.4f} [{low:.4f}, {high:.4f}]"


def write_markdown_table(df: pd.DataFrame, path: Path, columns: list[str], headers: list[str]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_metric_rows() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        path = Path(run["path"])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        y_true = df["_label_binary"].to_numpy(dtype=int)
        p_pos = df[run["prob_col"]].to_numpy(dtype=float)
        ci = bootstrap_ci(y_true, p_pos)
        row = {
            "dataset": run["dataset"],
            "protocol": run["protocol"],
            "method": run["method"],
            "n": int(len(df)),
            "source": run["path"],
        }
        for metric, (point, low, high) in ci.items():
            row[f"{metric}_point"] = point
            row[f"{metric}_low"] = low
            row[f"{metric}_high"] = high
            row[f"{metric}_ci"] = fmt_ci(point, low, high)
        rows.append(row)

    for run in CALIBRATED_RUNS:
        path = Path(run["path"])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        y_true = df["_label_binary"].to_numpy(dtype=int)
        for method, label in CALIBRATION_METHODS:
            prob_col = f"{method}_p_machine"
            if prob_col not in df.columns:
                continue
            p_pos = df[prob_col].to_numpy(dtype=float)
            ci = bootstrap_ci(y_true, p_pos)
            row = {
                "dataset": run["dataset"],
                "protocol": run["protocol"],
                "method": label,
                "n": int(len(df)),
                "source": run["path"],
            }
            for metric, (point, low, high) in ci.items():
                row[f"{metric}_point"] = point
                row[f"{metric}_low"] = low
                row[f"{metric}_high"] = high
                row[f"{metric}_ci"] = fmt_ci(point, low, high)
            rows.append(row)
    return pd.DataFrame(rows)


def collect_worst_groups() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        path = Path(run["path"])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        y_true = df["_label_binary"].to_numpy(dtype=int)
        p_pos = df[run["prob_col"]].to_numpy(dtype=float)
        y_pred = (p_pos >= 0.5).astype(int)
        confidence = np.maximum(p_pos, 1.0 - p_pos)
        scored = df.copy()
        scored["_is_error"] = y_true != y_pred
        scored["_confidence"] = confidence
        scored["_error_type"] = np.where(
            scored["_is_error"],
            np.where(y_pred == 1, "false_positive_machine", "false_negative_machine"),
            "correct",
        )
        for col in run["group_cols"]:
            if col not in scored.columns:
                continue
            for value, part in scored.groupby(col, dropna=False):
                if len(part) < 20:
                    continue
                rows.append(
                    {
                        "dataset": run["dataset"],
                        "method": run["method"],
                        "group_col": col,
                        "group_value": str(value),
                        "n": int(len(part)),
                        "accuracy": float(1.0 - part["_is_error"].mean()),
                        "error_rate": float(part["_is_error"].mean()),
                        "mean_confidence": float(part["_confidence"].mean()),
                        "high_conf_error_rate": float(((part["_is_error"]) & (part["_confidence"] >= 0.9)).mean()),
                        "false_positive_machine": int((part["_error_type"] == "false_positive_machine").sum()),
                        "false_negative_machine": int((part["_error_type"] == "false_negative_machine").sum()),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["error_rate", "n"], ascending=[False, False]).reset_index(drop=True)


def collect_selective_rows() -> pd.DataFrame:
    rows = []
    for run in CALIBRATED_RUNS:
        metrics_path = Path(run["metrics"])
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for method, label in CALIBRATION_METHODS:
            method_metrics = metrics["methods"].get(method, {})
            curve = method_metrics.get("selective_curve", [])
            row80 = min(curve, key=lambda r: abs(float(r["coverage"]) - 0.8)) if curve else {}
            conformal = metrics.get("conformal", {}).get(method, [])
            conf10 = next((r for r in conformal if abs(float(r["alpha"]) - 0.1) < 1e-9), {})
            rows.append(
                {
                    "dataset": run["dataset"],
                    "method": label,
                    "accuracy": method_metrics.get("accuracy"),
                    "brier": method_metrics.get("brier"),
                    "ece_10": method_metrics.get("ece_10"),
                    "risk_at_80_coverage": row80.get("risk"),
                    "threshold_at_80_coverage": row80.get("confidence_threshold"),
                    "conformal_coverage_alpha_0_10": conf10.get("empirical_coverage"),
                    "singleton_rate_alpha_0_10": conf10.get("singleton_rate"),
                    "singleton_risk_alpha_0_10": conf10.get("risk_on_singletons"),
                }
            )
    return pd.DataFrame(rows)


def plot_accuracy(metrics: pd.DataFrame, output_path: Path) -> None:
    part = metrics[metrics["method"] == "TF-IDF LR"].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    labels = ["\n".join(wrap(x, width=18, break_long_words=False)) for x in part["dataset"]]
    x = np.arange(len(part))
    y = part["accuracy_point"].to_numpy(dtype=float)
    low = part["accuracy_low"].to_numpy(dtype=float)
    high = part["accuracy_high"].to_numpy(dtype=float)
    ax.bar(x, y, color="#4C78A8", alpha=0.9)
    ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Locked empirical TF-IDF Logistic Regression accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ece(selective: pd.DataFrame, output_path: Path) -> None:
    if selective.empty:
        return
    pivot = selective.pivot(index="dataset", columns="method", values="ece_10")
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylabel("ECE-10")
    ax.set_xlabel("")
    ax.set_title("Calibration error under the locked protocol")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    labels = ["\n".join(wrap(str(x), width=18, break_long_words=False)) for x in pivot.index]
    ax.set_xticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selective_risk(selective: pd.DataFrame, output_path: Path) -> None:
    if selective.empty:
        return
    pivot = selective.pivot(index="dataset", columns="method", values="risk_at_80_coverage")
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylabel("Risk at 80% coverage")
    ax.set_xlabel("")
    ax.set_title("Selective prediction risk at fixed coverage")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    labels = ["\n".join(wrap(str(x), width=18, break_long_words=False)) for x in pivot.index]
    ax.set_xticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(metrics: pd.DataFrame, selective: pd.DataFrame, worst: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Locked Empirical Experiment Summary",
        "",
        f"Bootstrap resamples per metric interval: `{N_BOOTSTRAP}`.",
        "",
        "## Detector Performance",
        "",
        "| Dataset | Protocol | Method | N | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['protocol']} | {row['method']} | {int(row['n'])} | "
            f"{row['accuracy_ci']} | {row['macro_f1_ci']} | {row['auroc_ci']} | {row['brier_ci']} | {row['ece_10_ci']} |"
        )
    if not selective.empty:
        lines.extend(
            [
                "",
                "## Calibration and Selective Prediction",
                "",
                "| Dataset | Method | ECE-10 | Brier | Risk at 80% coverage | Conformal coverage at alpha=0.10 | Singleton rate |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in selective.iterrows():
            lines.append(
                "| {dataset} | {method} | {ece:.4f} | {brier:.4f} | {risk:.4f} | {coverage:.4f} | {singleton:.4f} |".format(
                    dataset=row["dataset"],
                    method=row["method"],
                    ece=row["ece_10"],
                    brier=row["brier"],
                    risk=row["risk_at_80_coverage"],
                    coverage=row["conformal_coverage_alpha_0_10"],
                    singleton=row["singleton_rate_alpha_0_10"],
                )
            )
    if not worst.empty:
        lines.extend(
            [
                "",
                "## Highest Error Subgroups",
                "",
                "| Dataset | Method | Group | Value | N | Accuracy | Error rate | High-conf error rate |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in worst.head(15).iterrows():
            lines.append(
                "| {dataset} | {method} | {group_col} | {group_value} | {n} | {accuracy:.4f} | {error_rate:.4f} | {high_conf_error_rate:.4f} |".format(
                    **row.to_dict()
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    table_dir = Path("manuscript/tables")
    figure_dir = Path("manuscript/figures")
    docs_dir = Path("docs")
    output_dir = Path("outputs/locked/statistical_analysis")
    for directory in [table_dir, figure_dir, docs_dir, output_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    metrics = collect_metric_rows()
    if metrics.empty:
        raise SystemExit("No locked prediction files found. Run locked experiments first.")
    metrics.to_csv(output_dir / "empirical_metrics_with_ci.csv", index=False, encoding="utf-8")

    metric_cols = ["dataset", "protocol", "method", "n", "accuracy_ci", "macro_f1_ci", "auroc_ci", "brier_ci", "ece_10_ci"]
    write_markdown_table(
        metrics[metric_cols],
        table_dir / "table1_empirical_metrics_with_ci.md",
        metric_cols,
        ["Dataset", "Protocol", "Method", "N", "Accuracy (95% CI)", "Macro-F1 (95% CI)", "AUROC (95% CI)", "Brier (95% CI)", "ECE-10 (95% CI)"],
    )
    metrics[metric_cols].to_csv(table_dir / "table1_empirical_metrics_with_ci.csv", index=False, encoding="utf-8")

    selective = collect_selective_rows()
    selective.to_csv(output_dir / "empirical_selective_metrics.csv", index=False, encoding="utf-8")
    if not selective.empty:
        selective_cols = [
            "dataset",
            "method",
            "ece_10",
            "brier",
            "risk_at_80_coverage",
            "threshold_at_80_coverage",
            "conformal_coverage_alpha_0_10",
            "singleton_rate_alpha_0_10",
        ]
        write_markdown_table(
            selective[selective_cols],
            table_dir / "table2_empirical_calibration_selective.md",
            selective_cols,
            ["Dataset", "Method", "ECE-10", "Brier", "Risk@80%Cov", "Threshold@80%Cov", "Conformal Cov alpha=0.10", "Singleton Rate"],
        )
        selective[selective_cols].to_csv(table_dir / "table2_empirical_calibration_selective.csv", index=False, encoding="utf-8")

    worst = collect_worst_groups()
    worst.to_csv(output_dir / "empirical_subgroup_errors.csv", index=False, encoding="utf-8")
    if not worst.empty:
        worst_cols = ["dataset", "method", "group_col", "group_value", "n", "accuracy", "error_rate", "high_conf_error_rate", "false_positive_machine", "false_negative_machine"]
        write_markdown_table(
            worst.head(15)[worst_cols],
            table_dir / "table3_empirical_worst_subgroups.md",
            worst_cols,
            ["Dataset", "Method", "Group", "Value", "N", "Accuracy", "Error Rate", "High-Conf Error Rate", "FP Machine", "FN Machine"],
        )
        worst.head(15)[worst_cols].to_csv(table_dir / "table3_empirical_worst_subgroups.csv", index=False, encoding="utf-8")

    plot_accuracy(metrics, figure_dir / "figure1_empirical_accuracy_with_ci.png")
    plot_ece(selective, figure_dir / "figure2_empirical_ece.png")
    plot_selective_risk(selective, figure_dir / "figure3_empirical_selective_risk.png")

    captions = [
        "# Figure Captions",
        "",
        "## Figure 1. Locked empirical TF-IDF Logistic Regression accuracy",
        "",
        "Accuracy for the all-training-row TF-IDF Logistic Regression detector on SemEval-2024 Subtask A full train-dev splits and MAGE full in-distribution/OOD evaluation files. Error bars show 95% nonparametric bootstrap intervals.",
        "",
        "## Figure 2. Calibration error under the locked protocol",
        "",
        "Expected Calibration Error with 10 bins for uncalibrated, temperature-scaled, and isotonic Logistic Regression probabilities. Calibration is fitted on held-out training data and evaluated on the locked test or dev split.",
        "",
        "## Figure 3. Selective prediction risk at fixed coverage",
        "",
        "Selective risk when retaining the 80% most confident predictions. Lower values indicate fewer errors among non-abstained examples, but must be interpreted with the fixed coverage constraint.",
    ]
    (figure_dir / "empirical_figure_captions.md").write_text("\n".join(captions) + "\n", encoding="utf-8")
    write_summary(metrics, selective, worst, docs_dir / "empirical_experiment_summary.md")
    print(
        json.dumps(
            {
                "metrics": str(output_dir / "empirical_metrics_with_ci.csv"),
                "selective": str(output_dir / "empirical_selective_metrics.csv"),
                "subgroups": str(output_dir / "empirical_subgroup_errors.csv"),
                "summary": str(docs_dir / "empirical_experiment_summary.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
