"""Prepare locked MAGE CSV files for manuscript experiments.

MAGE uses label 1 for human text and label 0 for machine-generated or
machine-paraphrased text. This converter preserves full input files by default
and writes the normalized project schema expected by the training scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare locked normalized MAGE CSV files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mage"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/locked/mage"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-per-src",
        type=int,
        default=0,
        help="0 keeps all training rows; positive values apply deterministic per-src sampling.",
    )
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


def normalize(path: Path, split: str, seed: int, max_per_src: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "label", "src"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if max_per_src > 0:
        parts = []
        for _, part in df.groupby("src", sort=True):
            parts.append(part.sample(n=min(len(part), max_per_src), random_state=seed))
        df = pd.concat(parts, ignore_index=True)
    else:
        df = df.reset_index(drop=True)

    normalized = pd.DataFrame()
    normalized["text"] = df["text"].fillna("").astype(str)
    normalized["label"] = df["label"].map(lambda value: "human" if int(value) == 1 else "machine")
    normalized["domain"] = df["src"].map(infer_domain)
    normalized["generator"] = [infer_generator(src, label) for src, label in zip(df["src"], df["label"])]
    normalized["perturbation"] = df["src"].map(infer_perturbation)
    normalized["source_dataset"] = "MAGE"
    normalized["source_split"] = split
    normalized["source_file"] = path.name
    normalized["source_id"] = df.index.astype(str)
    normalized["original_src"] = df["src"].astype(str)
    normalized["original_label"] = df["label"].astype(int)
    return normalized


def summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "labels": df["label"].value_counts().to_dict(),
        "domains": df["domain"].value_counts().head(20).to_dict(),
        "generators": df["generator"].value_counts().to_dict(),
        "perturbations": df["perturbation"].value_counts().to_dict(),
        "source_files": df["source_file"].value_counts().to_dict(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "train": ("train.csv", "mage_train_full.csv", int(args.max_train_per_src)),
        "validation": ("valid.csv", "mage_validation_full.csv", 0),
        "test_id": ("test.csv", "mage_test_in_distribution_full.csv", 0),
        "test_ood_gpt": ("test_ood_set_gpt.csv", "mage_test_ood_gpt_full.csv", 0),
        "test_ood_paraphrase": ("test_ood_set_gpt_para.csv", "mage_test_ood_paraphrase_full.csv", 0),
    }

    metadata = {
        "dataset": "MAGE",
        "raw_dir": str(args.raw_dir),
        "seed": int(args.seed),
        "max_train_per_src": int(args.max_train_per_src),
        "label_mapping": {"1": "human", "0": "machine"},
        "license_note": "The MAGE dataset card records Apache-2.0; verify redistribution terms before submission.",
        "files": {},
    }

    for split, (raw_name, out_name, max_per_src) in inputs.items():
        raw_path = args.raw_dir / raw_name
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        df = normalize(raw_path, split, int(args.seed), max_per_src=max_per_src)
        out_path = args.output_dir / out_name
        df.to_csv(out_path, index=False, encoding="utf-8")
        metadata["files"][split] = {"path": str(out_path), **summary(df)}

    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "metadata": str(meta_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
