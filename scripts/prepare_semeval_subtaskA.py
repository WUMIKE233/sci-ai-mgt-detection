"""Normalize SemEval-2024 Task 8 Subtask A JSONL files.

The official SemEval-2024 Task 8 repository documents Subtask A as binary
human-written versus machine-generated text classification. Labels are
0=human and 1=machine. This converter maps the official JSONL format into the
project's normalized CSV schema where `machine` is the positive class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OFFICIAL_REPO = "https://github.com/mbzuai-nlp/SemEval2024-task8"
SUBTASK_A_DRIVE_ID = "1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized SemEval-2024 Task 8 Subtask A CSV files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/semeval2024_task8/subtaskA/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/semeval2024_task8"))
    parser.add_argument("--track", choices=["monolingual", "multilingual"], default="monolingual")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--dev-file", type=Path)
    parser.add_argument("--max-train-per-group", type=int, default=500)
    parser.add_argument("--max-dev-per-group", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_file(raw_dir: Path, split: str, track: str) -> Path:
    return raw_dir / f"subtaskA_{split}_{track}.jsonl"


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download SemEval-2024 Task 8 Subtask A from {OFFICIAL_REPO} "
            f"or with: gdown --folder https://drive.google.com/drive/folders/{SUBTASK_A_DRIVE_ID}"
        )
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"{path} contains no JSONL records")
    return pd.DataFrame(records)


def normalize_label(value) -> str:
    if pd.isna(value):
        raise ValueError("Missing label in SemEval Subtask A row")
    label = int(value)
    if label == 0:
        return "human"
    if label == 1:
        return "machine"
    raise ValueError(f"Unexpected SemEval Subtask A label: {value!r}")


def sample_by_group(df: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    if max_per_group <= 0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    group_cols = ["label", "original_source", "original_model"]
    parts = []
    for _, part in df.groupby(group_cols, dropna=False, sort=True):
        parts.append(part.sample(n=min(len(part), max_per_group), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def normalize(path: Path, split: str, track: str, max_per_group: int, seed: int) -> pd.DataFrame:
    df = read_jsonl(path)
    required = {"id", "text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    original_source = df["source"].fillna("unknown").astype(str) if "source" in df else pd.Series("unknown", index=df.index)
    original_model = df["model"].fillna("unknown").astype(str) if "model" in df else pd.Series("unknown", index=df.index)
    normalized = pd.DataFrame(
        {
            "text": df["text"].fillna("").astype(str),
            "label": df["label"].map(normalize_label),
            "domain": original_source,
            "generator": original_model,
            "perturbation": "none",
            "track": track,
            "source_dataset": "SemEval-2024 Task 8 Subtask A",
            "source_split": split,
            "source_file": path.name,
            "source_id": df["id"].astype(str),
            "original_source": original_source,
            "original_model": original_model,
            "original_label": df["label"].astype(int),
        }
    )
    return sample_by_group(normalized, max_per_group, seed)


def summarize(df: pd.DataFrame) -> dict:
    fields = ["label", "domain", "generator", "track", "source_split"]
    result = {"rows": int(len(df))}
    for field in fields:
        result[field + "s"] = df[field].value_counts(dropna=False).head(50).to_dict()
    return result


def main() -> None:
    args = parse_args()
    train_file = args.train_file or default_file(args.raw_dir, "train", args.track)
    dev_file = args.dev_file or default_file(args.raw_dir, "dev", args.track)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = normalize(train_file, "train", args.track, args.max_train_per_group, args.seed)
    dev_df = normalize(dev_file, "dev", args.track, args.max_dev_per_group, args.seed)

    prefix = f"semeval_subtaskA_{args.track}"
    train_path = args.output_dir / f"{prefix}_train.csv"
    dev_path = args.output_dir / f"{prefix}_dev.csv"
    meta_path = args.output_dir / f"{prefix}_metadata.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    dev_df.to_csv(dev_path, index=False, encoding="utf-8")

    metadata = {
        "dataset": "SemEval-2024 Task 8",
        "subtask": "A",
        "track": args.track,
        "official_repo": OFFICIAL_REPO,
        "license_note": "Official GitHub repository is Apache-2.0; verify downloaded data terms before submission.",
        "download_hint": f"gdown --folder https://drive.google.com/drive/folders/{SUBTASK_A_DRIVE_ID}",
        "label_mapping": {"0": "human", "1": "machine"},
        "sampling": {
            "max_train_per_group": int(args.max_train_per_group),
            "max_dev_per_group": int(args.max_dev_per_group),
            "seed": int(args.seed),
            "group_columns": ["label", "source", "model"],
            "note": "Use 0 for max_*_per_group to process all rows for final experiments.",
        },
        "train": {"path": str(train_path), **summarize(train_df)},
        "dev": {"path": str(dev_path), **summarize(dev_df)},
        "limitations": [
            "This converter covers Subtask A full-text binary classification only.",
            "Final submission claims should use the locked full dataset or a predeclared stratified protocol.",
            "Official test labels and any redistribution restrictions must be verified before final reporting.",
        ],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps({"train": str(train_path), "dev": str(dev_path), "metadata": str(meta_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
