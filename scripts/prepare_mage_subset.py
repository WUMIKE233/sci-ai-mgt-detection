"""Normalize small MAGE CSV files into the project table schema.

MAGE uses label 1 for human text and label 0 for machine-generated or
machine-paraphrased text. The project baseline expects `human` / `machine`
labels and treats `machine` as the positive class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized MAGE subset CSV files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mage"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mage"))
    parser.add_argument("--train-file", default="test_ood_set_gpt.csv")
    parser.add_argument("--test-file", default="test_ood_set_gpt_para.csv")
    parser.add_argument("--max-train-per-source", type=int, default=120)
    parser.add_argument("--max-test-per-source", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_domain(src: str) -> str:
    return str(src).split("_")[0]


def infer_generator(src: str, label: int) -> str:
    src_text = str(src).lower()
    if int(label) == 1:
        return "human"
    if "gpt4" in src_text:
        return "gpt4"
    if "gpt" in src_text:
        return "gpt"
    if "para" in src_text:
        return "machine_paraphrase"
    return "machine_unknown"


def infer_perturbation(src: str) -> str:
    return "paraphrase" if "para" in str(src).lower() else "none"


def normalize(path: Path, max_per_source: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "label", "src"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    sampled_parts = []
    for _, part in df.groupby("src", sort=True):
        sampled_parts.append(part.sample(n=min(len(part), max_per_source), random_state=seed))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    normalized = pd.DataFrame()
    normalized["text"] = sampled["text"].fillna("").astype(str)
    normalized["label"] = sampled["label"].map(lambda value: "human" if int(value) == 1 else "machine")
    normalized["domain"] = sampled["src"].map(infer_domain)
    normalized["generator"] = [infer_generator(src, label) for src, label in zip(sampled["src"], sampled["label"])]
    normalized["perturbation"] = sampled["src"].map(infer_perturbation)
    normalized["source_dataset"] = "MAGE"
    normalized["source_file"] = path.name
    normalized["source_id"] = sampled.index.astype(str)
    normalized["original_src"] = sampled["src"].astype(str)
    normalized["original_label"] = sampled["label"].astype(int)
    return normalized


def summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "labels": df["label"].value_counts().to_dict(),
        "domains": df["domain"].value_counts().to_dict(),
        "generators": df["generator"].value_counts().to_dict(),
        "perturbations": df["perturbation"].value_counts().to_dict(),
        "source_files": df["source_file"].value_counts().to_dict(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = normalize(args.raw_dir / args.train_file, args.max_train_per_source, args.seed)
    test_df = normalize(args.raw_dir / args.test_file, args.max_test_per_source, args.seed)

    train_path = args.output_dir / "mage_train_small.csv"
    test_path = args.output_dir / "mage_test_ood_para_small.csv"
    meta_path = args.output_dir / "metadata.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    metadata = {
        "dataset": "MAGE",
        "raw_dir": str(args.raw_dir),
        "label_mapping": {"1": "human", "0": "machine"},
        "train": {"path": str(train_path), **summary(train_df)},
        "test": {"path": str(test_path), **summary(test_df)},
        "notes": [
            "This is a small reproducibility smoke subset, not a final manuscript experiment.",
            "The test file includes paraphrased examples and should be treated as an OOD/paraphrase stress test.",
        ],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps({"train": str(train_path), "test": str(test_path), "metadata": str(meta_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
