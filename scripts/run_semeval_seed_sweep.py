"""Run a repeated-seed SemEval Subtask A small-sample sweep.

This script repeats the low-cost SemEval pipeline across multiple sampling
seeds. It targets stability evidence for the preliminary manuscript package:
prepare sampled CSVs, run TF-IDF baselines, run calibration/selective
prediction, and aggregate seed-level metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [13, 42, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SemEval Subtask A repeated-seed small-sample experiments.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/semeval2024_task8/subtaskA/SubtaskA"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/semeval2024_task8_seed_sweep"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/semeval_seed_sweep"))
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--max-train-per-group", type=int, default=500)
    parser.add_argument("--max-dev-per-group", type=int, default=500)
    parser.add_argument("--track", choices=["monolingual", "multilingual"], default="monolingual")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    return seeds or DEFAULT_SEEDS


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def metric_row(seed: int, dataset_split: str, run: str, model: str, metrics: dict, source: str) -> dict:
    return {
        "seed": seed,
        "dataset_split": dataset_split,
        "run": run,
        "model": model,
        "n": metrics.get("n"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "brier": metrics.get("brier"),
        "ece_10": metrics.get("ece_10"),
        "source": source,
    }


def load_seed_metrics(seed: int, output_root: Path) -> list[dict]:
    rows = []
    seed_dir = output_root / f"seed_{seed}"
    baseline_path = seed_dir / "baseline" / "metrics.json"
    if baseline_path.exists():
        with baseline_path.open("r", encoding="utf-8") as f:
            baseline = json.load(f)
        for model, metrics in baseline.items():
            rows.append(metric_row(seed, "SemEval Subtask A monolingual dev", "TF-IDF baselines", model, metrics, str(baseline_path)))

    calibration_path = seed_dir / "calibrated_selective" / "metrics.json"
    if calibration_path.exists():
        with calibration_path.open("r", encoding="utf-8") as f:
            calibration = json.load(f)
        for model, metrics in calibration.get("methods", {}).items():
            rows.append(
                metric_row(
                    seed,
                    "SemEval Subtask A monolingual dev",
                    "Calibration/selective",
                    model,
                    metrics,
                    str(calibration_path),
                )
            )
    return rows


def write_markdown(metrics: pd.DataFrame, summary: pd.DataFrame, path: Path, seeds: list[int]) -> None:
    lines = [
        "# SemEval Seed Sweep Report",
        "",
        "This report repeats the low-cost SemEval-2024 Task 8 Subtask A monolingual small-sample pipeline across sampling seeds. It is preliminary stability evidence, not final benchmark evidence.",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        "",
        "## Per-Seed Metrics",
        "",
        "| Seed | Run | Model | N | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.sort_values(["run", "model", "seed"]).iterrows():
        lines.append(
            f"| {int(row['seed'])} | {row['run']} | {row['model']} | {int(row['n'])} | "
            f"{fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | {fmt(row['auroc'])} | {fmt(row['brier'])} | {fmt(row['ece_10'])} |"
        )

    lines.extend(
        [
            "",
            "## Across-Seed Summary",
            "",
            "| Run | Model | Accuracy Mean | Accuracy SD | Macro-F1 Mean | Macro-F1 SD | AUROC Mean | AUROC SD | ECE-10 Mean | ECE-10 SD |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.sort_values(["run", "model"]).iterrows():
        lines.append(
            f"| {row['run']} | {row['model']} | {fmt(row['accuracy_mean'])} | {fmt(row['accuracy_std'])} | "
            f"{fmt(row['macro_f1_mean'])} | {fmt(row['macro_f1_std'])} | {fmt(row['auroc_mean'])} | {fmt(row['auroc_std'])} | "
            f"{fmt(row['ece_10_mean'])} | {fmt(row['ece_10_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- The seed changes the sampled training subset and the calibration split.",
            "- The dev set remains the same because the current `max-dev-per-group=500` covers the sampled monolingual dev groups.",
            "- Final submission still requires locked full processing, more seeds for neural baselines, and journal-ready statistical analysis.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.processed_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        processed_dir = args.processed_root / f"seed_{seed}"
        seed_dir = args.output_root / f"seed_{seed}"
        train_path = processed_dir / f"semeval_subtaskA_{args.track}_train.csv"
        dev_path = processed_dir / f"semeval_subtaskA_{args.track}_dev.csv"

        run_command(
            [
                args.python,
                "scripts/prepare_semeval_subtaskA.py",
                "--raw-dir",
                str(args.raw_dir),
                "--output-dir",
                str(processed_dir),
                "--track",
                args.track,
                "--max-train-per-group",
                str(args.max_train_per_group),
                "--max-dev-per-group",
                str(args.max_dev_per_group),
                "--seed",
                str(seed),
            ]
        )
        run_command(
            [
                args.python,
                "scripts/train_baselines.py",
                "--train",
                str(train_path),
                "--test",
                str(dev_path),
                "--models",
                "logreg,linear_svm",
                "--group-cols",
                "domain,generator,track,original_source",
                "--max-features",
                "50000",
                "--ngram-min",
                "1",
                "--ngram-max",
                "2",
                "--output-dir",
                str(seed_dir / "baseline"),
            ]
        )
        run_command(
            [
                args.python,
                "scripts/run_calibrated_selective.py",
                "--train",
                str(train_path),
                "--test",
                str(dev_path),
                "--group-cols",
                "domain,generator,track,original_source",
                "--calibration-size",
                "0.25",
                "--seed",
                str(seed),
                "--max-features",
                "50000",
                "--ngram-min",
                "1",
                "--ngram-max",
                "2",
                "--dataset-label",
                f"SemEval-2024 Task 8 Subtask A monolingual dev seed {seed}",
                "--output-dir",
                str(seed_dir / "calibrated_selective"),
            ]
        )

    rows = []
    for seed in seeds:
        rows.extend(load_seed_metrics(seed, args.output_root))
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise SystemExit("No seed-sweep metrics were collected.")

    metrics_path = args.output_root / "seed_sweep_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")

    summary = (
        metrics.groupby(["run", "model"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            ece_10_mean=("ece_10", "mean"),
            ece_10_std=("ece_10", "std"),
        )
        .reset_index(drop=True)
    )
    summary_path = args.output_root / "seed_sweep_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    report_path = Path("docs/semeval_seed_sweep_report.md")
    write_markdown(metrics, summary, report_path, seeds)

    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "summary": str(summary_path),
                "report": str(report_path),
                "seeds": seeds,
                "rows": int(len(metrics)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
