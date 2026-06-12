"""Generate high-confidence error and subgroup failure reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RUNS = [
    ("MAGE small", "fine_tuned_distilbert", "outputs/mage_small_transformer_finetune/predictions.csv", "transformer_finetune_p_machine", ["domain", "generator", "perturbation", "original_src"]),
    ("MAGE small", "frozen_distilbert_probe", "outputs/mage_small_transformer_probe/predictions.csv", "transformer_probe_p_machine", ["domain", "generator", "perturbation", "original_src"]),
    ("RAID small", "fine_tuned_distilbert", "outputs/raid_small_transformer_finetune/predictions.csv", "transformer_finetune_p_machine", ["domain", "generator", "perturbation", "source_split"]),
    ("RAID small", "frozen_distilbert_probe", "outputs/raid_small_transformer_probe/predictions.csv", "transformer_probe_p_machine", ["domain", "generator", "perturbation", "source_split"]),
    ("SemEval Subtask A mono small", "fine_tuned_distilbert", "outputs/semeval_subtaskA_mono_small_transformer_finetune/predictions.csv", "transformer_finetune_p_machine", ["domain", "generator", "track", "original_source"]),
    ("SemEval Subtask A mono small", "frozen_distilbert_probe", "outputs/semeval_subtaskA_mono_small_transformer_probe/predictions.csv", "transformer_probe_p_machine", ["domain", "generator", "track", "original_source"]),
]


def add_prediction_columns(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    result = df.copy()
    result["_p_machine"] = result[prob_col].astype(float)
    result["_prediction"] = (result["_p_machine"] >= 0.5).astype(int)
    result["_confidence"] = np.maximum(result["_p_machine"], 1.0 - result["_p_machine"])
    result["_is_error"] = result["_prediction"] != result["_label_binary"].astype(int)
    result["_error_type"] = np.where(
        result["_is_error"],
        np.where(result["_prediction"] == 1, "false_positive_machine", "false_negative_machine"),
        "correct",
    )
    return result


def summarize_groups(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in group_cols:
        if col not in df.columns:
            continue
        for value, part in df.groupby(col, dropna=False):
            rows.append(
                {
                    "group_col": col,
                    "group_value": str(value),
                    "n": int(len(part)),
                    "error_rate": float(part["_is_error"].mean()),
                    "accuracy": float(1.0 - part["_is_error"].mean()),
                    "mean_confidence": float(part["_confidence"].mean()),
                    "high_conf_error_rate": float(((part["_is_error"]) & (part["_confidence"] >= 0.9)).mean()),
                    "false_positive_machine": int((part["_error_type"] == "false_positive_machine").sum()),
                    "false_negative_machine": int((part["_error_type"] == "false_negative_machine").sum()),
                }
            )
    return pd.DataFrame(rows)


def truncate_text(text: str, max_chars: int = 360) -> str:
    text = str(text).replace("\r", " ").replace("\n", " ")
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def main() -> None:
    output_dir = Path("outputs/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_group_rows = []
    high_conf_rows = []

    for dataset, model, path, prob_col, group_cols in RUNS:
        source = Path(path)
        if not source.exists():
            continue
        df = pd.read_csv(source)
        if "_label_binary" not in df.columns or prob_col not in df.columns:
            continue
        scored = add_prediction_columns(df, prob_col)
        group_df = summarize_groups(scored, group_cols)
        group_df.insert(0, "model", model)
        group_df.insert(0, "dataset", dataset)
        all_group_rows.append(group_df)

        errors = scored[scored["_is_error"]].sort_values("_confidence", ascending=False).head(30).copy()
        keep_cols = [col for col in ["text", "label", "domain", "generator", "perturbation", "original_src", "source_split", "source_id", "_label_binary", "_prediction", "_p_machine", "_confidence", "_error_type"] if col in errors.columns]
        errors = errors[keep_cols]
        errors.insert(0, "model", model)
        errors.insert(0, "dataset", dataset)
        if "text" in errors.columns:
            errors["text"] = errors["text"].map(truncate_text)
        high_conf_rows.append(errors)

    groups = pd.concat(all_group_rows, ignore_index=True) if all_group_rows else pd.DataFrame()
    high_conf = pd.concat(high_conf_rows, ignore_index=True) if high_conf_rows else pd.DataFrame()
    group_path = output_dir / "subgroup_error_summary.csv"
    error_path = output_dir / "high_confidence_errors.csv"
    groups.to_csv(group_path, index=False, encoding="utf-8")
    high_conf.to_csv(error_path, index=False, encoding="utf-8")

    worst = groups[groups["n"] >= 10].sort_values(["error_rate", "n"], ascending=[False, False]).head(20)
    md_path = Path("docs/error_analysis.md")
    lines = [
        "# Error Analysis",
        "",
        "This report summarizes subgroup failures and high-confidence errors for the current Transformer baselines. It is preliminary because the data are small sampled subsets.",
        "",
        f"Subgroup CSV: `{group_path}`",
        f"High-confidence error CSV: `{error_path}`",
        "",
        "## Worst Subgroups",
        "",
        "| Dataset | Model | Group | Value | N | Accuracy | Error Rate | High-Conf Error Rate | FP machine | FN machine |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in worst.iterrows():
        lines.append(
            "| {dataset} | {model} | {group_col} | {group_value} | {n} | {accuracy:.4f} | {error_rate:.4f} | {high_conf_error_rate:.4f} | {fp} | {fn} |".format(
                dataset=row["dataset"],
                model=row["model"],
                group_col=row["group_col"],
                group_value=row["group_value"],
                n=int(row["n"]),
                accuracy=row["accuracy"],
                error_rate=row["error_rate"],
                high_conf_error_rate=row["high_conf_error_rate"],
                fp=int(row["false_positive_machine"]),
                fn=int(row["false_negative_machine"]),
            )
        )
    lines.extend(
        [
            "",
            "## Main Observations",
            "",
            "- MAGE paraphrased human-source examples remain the most severe observed failure mode.",
            "- RAID synonym and some model-specific groups remain challenging despite strong aggregate metrics.",
            "- SemEval Subtask A monolingual dev shows persistent unseen-generator difficulty for both frozen and fine-tuned DistilBERT baselines.",
            "- High-confidence errors should be reviewed before writing final claims about reliability.",
            "",
            "## Required Before Submission",
            "",
            "- Repeat error analysis on larger samples and final model predictions.",
            "- Include representative examples only when dataset license and journal policy permit text excerpts.",
            "- Avoid treating detector outputs as proof of authorship.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"groups": str(group_path), "errors": str(error_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
