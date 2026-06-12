"""Summarize preliminary experiment metrics into CSV and Markdown tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUNS = [
    ("MAGE small", "TF-IDF baselines", "outputs/mage_small_baseline/metrics.json", "baseline"),
    ("MAGE small", "Calibration/selective", "outputs/mage_small_calibrated_selective/metrics.json", "calibration"),
    ("MAGE small", "Frozen Transformer probe", "outputs/mage_small_transformer_probe/metrics.json", "single_metrics"),
    ("MAGE small", "Fine-tuned Transformer", "outputs/mage_small_transformer_finetune/metrics.json", "finetune"),
    ("RAID small", "TF-IDF baselines", "outputs/raid_small_baseline/metrics.json", "baseline"),
    ("RAID small", "Calibration/selective", "outputs/raid_small_calibrated_selective/metrics.json", "calibration"),
    ("RAID small", "Frozen Transformer probe", "outputs/raid_small_transformer_probe/metrics.json", "single_metrics"),
    ("RAID small", "Fine-tuned Transformer", "outputs/raid_small_transformer_finetune/metrics.json", "finetune"),
    ("SemEval Subtask A mono small", "TF-IDF baselines", "outputs/semeval_subtaskA_mono_small_baseline/metrics.json", "baseline"),
    ("SemEval Subtask A mono small", "Calibration/selective", "outputs/semeval_subtaskA_mono_small_calibrated_selective/metrics.json", "calibration"),
    ("SemEval Subtask A mono small", "Frozen Transformer probe", "outputs/semeval_subtaskA_mono_small_transformer_probe/metrics.json", "single_metrics"),
    ("SemEval Subtask A mono small", "Fine-tuned Transformer", "outputs/semeval_subtaskA_mono_small_transformer_finetune/metrics.json", "finetune"),
]


def metric_row(dataset: str, run: str, model: str, metrics: dict, path: str) -> dict:
    return {
        "dataset": dataset,
        "run": run,
        "model": model,
        "n": metrics.get("n"),
        "positive_rate": metrics.get("positive_rate"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "brier": metrics.get("brier"),
        "ece_10": metrics.get("ece_10"),
        "source": path,
    }


def rows_from_file(dataset: str, run: str, path: str, kind: str) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    if kind == "baseline":
        for model, metrics in data.items():
            rows.append(metric_row(dataset, run, model, metrics, path))
    elif kind == "calibration":
        for model, metrics in data["methods"].items():
            rows.append(metric_row(dataset, run, model, metrics, path))
    elif kind == "single_metrics":
        rows.append(metric_row(dataset, run, "distilbert_frozen_probe", data["metrics"], path))
    elif kind == "finetune":
        rows.append(metric_row(dataset, run, "distilbert_finetuned", data["test_metrics"], path))
    else:
        raise ValueError(f"Unknown run kind: {kind}")
    return rows


def fmt(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    rows = []
    for dataset, run, path, kind in RUNS:
        rows.extend(rows_from_file(dataset, run, path, kind))
    if not rows:
        raise SystemExit("No metrics found to summarize.")

    output_dir = Path("outputs/experiment_summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "preliminary_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    md_path = Path("docs/preliminary_experiment_summary.md")
    display_cols = ["dataset", "run", "model", "accuracy", "macro_f1", "auroc", "brier", "ece_10"]
    lines = [
        "# Preliminary Experiment Summary",
        "",
        "This table summarizes real-data preliminary experiments. These results are not yet submission-grade because they use small samples, single seeds, and incomplete benchmark coverage.",
        "",
        f"CSV source: `{csv_path}`",
        "",
        "| Dataset | Run | Model | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in df[display_cols].iterrows():
        lines.append(
            "| {dataset} | {run} | {model} | {accuracy} | {macro_f1} | {auroc} | {brier} | {ece} |".format(
                dataset=row["dataset"],
                run=row["run"],
                model=row["model"],
                accuracy=fmt(row["accuracy"]),
                macro_f1=fmt(row["macro_f1"]),
                auroc=fmt(row["auroc"]),
                brier=fmt(row["brier"]),
                ece=fmt(row["ece_10"]),
            )
        )
    lines.extend(
        [
            "",
            "## Current Strongest Preliminary Rows",
            "",
            "- MAGE small: fine-tuned DistilBERT has the highest accuracy among current runs, but paraphrase failures remain.",
            "- RAID small: frozen and fine-tuned DistilBERT are both strong; calibrated Linear SVM remains a competitive lightweight baseline.",
            "- SemEval Subtask A monolingual dev: current TF-IDF and DistilBERT baselines show clear cross-generator difficulty because the dev generator is unseen in the sampled train set.",
            "",
            "## Required Before Submission",
            "",
            "- Larger stratified samples or full benchmark processing.",
            "- Repeated random seeds and confidence intervals.",
            "- Full SemEval/M4 processing, including official test/gold handling where permitted.",
            "- Final table generation from locked scripts only.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "rows": len(df)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
