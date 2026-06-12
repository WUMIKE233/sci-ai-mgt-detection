"""Generate manuscript-ready preliminary tables and figures.

The assets generated here are intentionally labeled preliminary. They help turn
the current evidence into reusable manuscript components while preserving the
boundary that final submission requires locked full-scale experiments.
"""

from __future__ import annotations

import json
from textwrap import wrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "logreg": "TF-IDF LR",
    "linear_svm": "TF-IDF SVM",
    "identity": "Uncalibrated LR",
    "temperature": "Temp-scaled LR",
    "isotonic": "Isotonic LR",
    "distilbert_frozen_probe": "Frozen DistilBERT",
    "distilbert_finetuned": "Fine-tuned DistilBERT",
}

MODEL_ORDER = [
    "TF-IDF LR",
    "TF-IDF SVM",
    "Uncalibrated LR",
    "Temp-scaled LR",
    "Isotonic LR",
    "Frozen DistilBERT",
    "Fine-tuned DistilBERT",
]

GROUP_MODEL_LABELS = {
    "fine_tuned_distilbert": "Fine-tuned DistilBERT",
    "frozen_distilbert_probe": "Frozen DistilBERT",
}

GROUP_COLUMN_LABELS = {
    "original_src": "source",
    "generator": "generator",
    "domain": "domain",
    "perturbation": "perturbation",
}


def fmt_ci(point, low, high) -> str:
    if pd.isna(point):
        return ""
    if pd.isna(low) or pd.isna(high):
        return f"{point:.4f}"
    return f"{point:.4f} [{low:.4f}, {high:.4f}]"


def load_ci_wide() -> pd.DataFrame:
    ci = pd.read_csv("outputs/statistical_analysis/bootstrap_confidence_intervals.csv")
    rows = []
    for (dataset, run, model, n), part in ci.groupby(["dataset", "run", "model", "n"], sort=False):
        row = {"dataset": dataset, "run": run, "model": model, "model_label": MODEL_LABELS.get(model, model), "n": int(n)}
        for _, metric_row in part.iterrows():
            metric = metric_row["metric"]
            row[f"{metric}_point"] = metric_row["point"]
            row[f"{metric}_low"] = metric_row["ci_low"]
            row[f"{metric}_high"] = metric_row["ci_high"]
            row[f"{metric}_ci"] = fmt_ci(metric_row["point"], metric_row["ci_low"], metric_row["ci_high"])
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide["model_order"] = wide["model_label"].map({name: idx for idx, name in enumerate(MODEL_ORDER)}).fillna(99)
    return wide.sort_values(["dataset", "model_order", "run"]).reset_index(drop=True)


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


def plot_metric_bars(df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    datasets = list(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(13, 5.2 * len(datasets)), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        part = df[df["dataset"] == dataset].sort_values("model_order")
        x = np.arange(len(part))
        y = part[f"{metric}_point"].to_numpy(dtype=float)
        low = part[f"{metric}_low"].to_numpy(dtype=float)
        high = part[f"{metric}_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - low, high - y])
        ax.bar(x, y, color="#4C78A8", alpha=0.88)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222222", elinewidth=1.1, capsize=3)
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(part["model_label"], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1.02 if metric != "ece_10" else max(0.35, float(np.nanmax(high)) + 0.05))
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"Preliminary {ylabel} with 95% bootstrap intervals", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_worst_subgroups(output_path: Path) -> pd.DataFrame:
    groups = pd.read_csv("outputs/error_analysis/subgroup_error_summary.csv")
    worst = groups[groups["n"] >= 10].sort_values(["error_rate", "n"], ascending=[False, False]).head(12).copy()
    worst["model_label"] = worst["model"].map(GROUP_MODEL_LABELS).fillna(worst["model"])
    worst["group_label"] = worst["group_col"].map(GROUP_COLUMN_LABELS).fillna(worst["group_col"])
    worst["label"] = [
        "\n".join(
            wrap(
                f"{row.dataset.replace(' small', '')} | {row.model_label} | {row.group_label}={row.group_value}",
                width=58,
                break_long_words=False,
            )
        )
        for row in worst.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(9, 6.6))
    y = np.arange(len(worst))
    ax.barh(y, worst["error_rate"], color="#F58518", alpha=0.9)
    for idx, value in enumerate(worst["error_rate"]):
        ax.text(min(float(value) + 0.015, 0.97), idx, f"{float(value):.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(worst["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Error rate")
    ax.set_title("Worst preliminary subgroups by error rate")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return worst


def write_captions(path: Path) -> None:
    lines = [
        "# Figure Captions",
        "",
        "## Figure 1. Preliminary model comparison by accuracy",
        "",
        "Accuracy on the current MAGE, RAID, and SemEval small sampled test sets. Error bars show 95% bootstrap confidence intervals over test examples. These intervals quantify uncertainty within the sampled subsets and do not replace full benchmark evaluation.",
        "",
        "## Figure 2. Preliminary calibration error comparison",
        "",
        "Expected Calibration Error (ECE-10) on the current MAGE, RAID, and SemEval small sampled test sets. Lower values indicate better probability calibration. Results remain preliminary because calibration and evaluation were performed on small sampled subsets.",
        "",
        "## Figure 3. Worst preliminary subgroups",
        "",
        "Highest-error subgroups among the current Transformer baselines. The strongest observed failure mode is paraphrased human-source biomedical text in MAGE, where machine-paraphrased examples are often classified as human-written. This supports treating detector output as a risk signal rather than an authorship verdict.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_current_index(path: Path) -> None:
    lines = [
        "# Current Project Status Index",
        "",
        "This file is the preferred entry point for the current state of the SCI manuscript project. Older reports remain useful as stage logs, but some older limitation statements have been superseded by later analysis.",
        "",
        "## Latest Evidence Files",
        "",
        "- `docs/preliminary_experiment_summary.md`: compact table of all current preliminary model metrics.",
        "- `docs/bootstrap_confidence_intervals.md`: 95% bootstrap confidence intervals from current prediction files.",
        "- `docs/error_analysis.md`: current subgroup and high-confidence error analysis.",
        "- `docs/submission_readiness_report.md`: conservative submission readiness audit and blocker list.",
        "- `docs/transformer_finetune_preliminary_report.md`: small-sample fine-tuned DistilBERT baseline.",
        "- `docs/raid_preliminary_report.md`: RAID labeled-subset data boundary and preliminary robustness observations.",
        "- `docs/semeval_preliminary_report.md`: official SemEval Subtask A monolingual small-sample baseline and calibration results.",
        "- `docs/semeval_seed_sweep_report.md`: repeated-seed stability report for low-cost SemEval TF-IDF and calibration runs.",
        "- `docs/neural_seed_sweep_report.md`: repeated-seed stability report for SemEval frozen and fine-tuned DistilBERT neural baselines.",
        "- `docs/semeval_subtaskA_data_note.md`: SemEval-2024 Task 8 Subtask A schema, download, and conversion notes.",
        "- `docs/environment_snapshot.md`: local Python/CUDA/package reproducibility snapshot.",
        "- `docs/submission_package_checklist.md`: manuscript, data, reproducibility, and author-metadata submission checklist.",
        "- `docs/submission_docx_qa.md`: structural QA record for the generated DOCX submission package.",
        "- `https://github.com/WUMIKE233/sci-ai-mgt-detection`: public code repository excluding raw data, processed benchmark extracts, generated outputs, and local submission-private materials.",
        "- `outputs/reproducibility/sci_ai_mgt_detection_code_package.zip`: local code archive excluding raw datasets.",
        "",
        "## Manuscript Assets",
        "",
        "- `manuscript/tables/table1_preliminary_metrics_with_ci.md`",
        "- `manuscript/tables/table2_worst_subgroups.md`",
        "- `manuscript/figures/figure1_accuracy_with_ci.png`",
        "- `manuscript/figures/figure2_ece_with_ci.png`",
        "- `manuscript/figures/figure3_worst_subgroups.png`",
        "- `manuscript/figures/figure_captions.md`",
        "- `submission_package/manuscript_draft_eswa.docx`",
        "- `submission_package/cover_letter_draft.docx`",
        "- `submission_package/declarations_draft.docx`",
        "- `submission_package/author_metadata_template.docx`",
        "",
        "## Current Completion Boundary",
        "",
        "The project now has real-data preliminary experiments, Transformer baselines, bootstrap intervals, error analysis, official SemEval Subtask A monolingual small-sample TF-IDF and DistilBERT runs, a three-seed SemEval stability sweep for low-cost models, a three-seed SemEval neural baseline sweep, generated DOCX submission drafts, a public GitHub code repository, a local reproducibility code archive, and a repeatable submission-readiness audit. The latest audit status is still BLOCKED because final Web of Science JCR/CAS target-journal verification remains required.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    table_dir = Path("manuscript/tables")
    figure_dir = Path("manuscript/figures")
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    ci = load_ci_wide()
    table_cols = ["dataset", "model_label", "n", "accuracy_ci", "macro_f1_ci", "auroc_ci", "brier_ci", "ece_10_ci"]
    table_headers = ["Dataset", "Model", "N", "Accuracy (95% CI)", "Macro-F1 (95% CI)", "AUROC (95% CI)", "Brier (95% CI)", "ECE-10 (95% CI)"]
    table1 = ci[table_cols].copy()
    table1.to_csv(table_dir / "table1_preliminary_metrics_with_ci.csv", index=False, encoding="utf-8")
    write_markdown_table(table1, table_dir / "table1_preliminary_metrics_with_ci.md", table_cols, table_headers)

    plot_df = ci[ci["model_label"].isin(["TF-IDF LR", "TF-IDF SVM", "Frozen DistilBERT", "Fine-tuned DistilBERT"])].copy()
    plot_metric_bars(plot_df, "accuracy", "Accuracy", figure_dir / "figure1_accuracy_with_ci.png")
    plot_metric_bars(plot_df, "ece_10", "ECE-10", figure_dir / "figure2_ece_with_ci.png")

    worst = plot_worst_subgroups(figure_dir / "figure3_worst_subgroups.png")
    worst_cols = ["dataset", "model", "group_col", "group_value", "n", "accuracy", "error_rate", "high_conf_error_rate", "false_positive_machine", "false_negative_machine"]
    worst[worst_cols].to_csv(table_dir / "table2_worst_subgroups.csv", index=False, encoding="utf-8")
    write_markdown_table(
        worst[worst_cols],
        table_dir / "table2_worst_subgroups.md",
        worst_cols,
        ["Dataset", "Model", "Group", "Value", "N", "Accuracy", "Error Rate", "High-Conf Error Rate", "FP Machine", "FN Machine"],
    )

    write_captions(figure_dir / "figure_captions.md")
    write_current_index(Path("docs/current_status_index.md"))
    print(json.dumps({"tables": str(table_dir), "figures": str(figure_dir), "status": "docs/current_status_index.md"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
