"""Run repeated-seed neural SemEval Subtask A small-sample experiments.

The sweep repeats the sampled SemEval monolingual pipeline and then runs two
DistilBERT neural baselines: a frozen embedding probe and a one-epoch
fine-tuned classifier. It is designed as preliminary stability evidence for
the manuscript package.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [13, 42, 2026]
GROUP_COLS = "domain,generator,track,original_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed neural SemEval experiments.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/semeval2024_task8/subtaskA/SubtaskA"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/semeval2024_task8_neural_seed_sweep"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/neural_seed_sweep"))
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--max-train-per-group", type=int, default=500)
    parser.add_argument("--max-dev-per-group", type=int, default=500)
    parser.add_argument("--track", choices=["monolingual", "multilingual"], default="monolingual")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--finetune-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    return seeds or DEFAULT_SEEDS


def run_command(command: list[str]) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def metric_row(seed: int, run: str, model: str, metrics: dict, source: Path) -> dict:
    return {
        "seed": seed,
        "dataset_split": "SemEval-2024 Task 8 Subtask A monolingual dev",
        "run": run,
        "model": model,
        "n": metrics.get("n"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "brier": metrics.get("brier"),
        "ece_10": metrics.get("ece_10"),
        "source": str(source),
    }


def load_seed_metrics(seed: int, output_root: Path) -> list[dict]:
    rows = []
    seed_dir = output_root / f"seed_{seed}"

    probe_path = seed_dir / "transformer_probe" / "metrics.json"
    if probe_path.exists():
        with probe_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(metric_row(seed, "Frozen Transformer probe", "distilbert_frozen_probe", data["metrics"], probe_path))

    finetune_path = seed_dir / "transformer_finetune" / "metrics.json"
    if finetune_path.exists():
        with finetune_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(metric_row(seed, "Fine-tuned Transformer", "distilbert_finetuned", data["test_metrics"], finetune_path))

    return rows


def write_markdown(metrics: pd.DataFrame, summary: pd.DataFrame, path: Path, args: argparse.Namespace, seeds: list[int]) -> None:
    lines = [
        "# Neural Seed Sweep Report",
        "",
        "This report repeats DistilBERT neural baselines on the SemEval-2024 Task 8 Subtask A monolingual small-sample setup. It provides preliminary repeated-seed stability evidence and is not a final full-benchmark result.",
        "",
        "## Configuration",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Model: `{args.model_name}`",
        f"- Max length: `{args.max_length}`",
        f"- Fine-tuning epochs: `{args.epochs}`",
        f"- Max train rows per label/generator group: `{args.max_train_per_group}`",
        f"- Max dev rows per label/generator group: `{args.max_dev_per_group}`",
        f"- Device setting: `{args.device}`",
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
            "| Run | Model | Seeds | Accuracy Mean | Accuracy SD | Macro-F1 Mean | Macro-F1 SD | AUROC Mean | AUROC SD | ECE-10 Mean | ECE-10 SD |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.sort_values(["run", "model"]).iterrows():
        lines.append(
            f"| {row['run']} | {row['model']} | {int(row['seed_count'])} | "
            f"{fmt(row['accuracy_mean'])} | {fmt(row['accuracy_std'])} | "
            f"{fmt(row['macro_f1_mean'])} | {fmt(row['macro_f1_std'])} | "
            f"{fmt(row['auroc_mean'])} | {fmt(row['auroc_std'])} | "
            f"{fmt(row['ece_10_mean'])} | {fmt(row['ece_10_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- The seed changes the sampled training subset, validation split, and model initialization.",
            "- The current dev sample is shared across seeds when the configured cap covers the available monolingual dev groups.",
            "- These results support pipeline stability checks only. Submission-grade claims still require locked full processing, journal-specific reporting, and final confidence intervals.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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

        if not args.skip_probe:
            run_command(
                [
                    args.python,
                    "scripts/train_transformer_probe.py",
                    "--train",
                    str(train_path),
                    "--test",
                    str(dev_path),
                    "--text-col",
                    "text",
                    "--label-col",
                    "label",
                    "--group-cols",
                    GROUP_COLS,
                    "--model-name",
                    args.model_name,
                    "--pooling",
                    "mean",
                    "--max-length",
                    str(args.max_length),
                    "--batch-size",
                    str(args.probe_batch_size),
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--cache-dir",
                    str(args.output_root / "transformer_cache" / f"seed_{seed}"),
                    "--output-dir",
                    str(seed_dir / "transformer_probe"),
                ]
            )

        if not args.skip_finetune:
            run_command(
                [
                    args.python,
                    "scripts/train_transformer_finetune.py",
                    "--train",
                    str(train_path),
                    "--test",
                    str(dev_path),
                    "--text-col",
                    "text",
                    "--label-col",
                    "label",
                    "--group-cols",
                    GROUP_COLS,
                    "--model-name",
                    args.model_name,
                    "--max-length",
                    str(args.max_length),
                    "--batch-size",
                    str(args.finetune_batch_size),
                    "--epochs",
                    str(args.epochs),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--weight-decay",
                    str(args.weight_decay),
                    "--validation-size",
                    str(args.validation_size),
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(seed_dir / "transformer_finetune"),
                ]
            )

    rows = []
    for seed in seeds:
        rows.extend(load_seed_metrics(seed, args.output_root))
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise SystemExit("No neural seed-sweep metrics were collected.")

    metrics_path = args.output_root / "neural_seed_sweep_metrics.csv"
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
    summary_path = args.output_root / "neural_seed_sweep_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    report_path = Path("docs/neural_seed_sweep_report.md")
    write_markdown(metrics, summary, report_path, args, seeds)

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
