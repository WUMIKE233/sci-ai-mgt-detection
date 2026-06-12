"""Create a small labeled RAID subset without downloading full CSV files.

RAID's public test split is unlabeled. This script samples labeled rows from
the `raid/train` and optionally `raid/extra` splits through the Hugging Face
dataset viewer rows API. It is meant for pipeline validation and early failure
analysis, not final benchmark reporting.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


API_URL = "https://datasets-server.huggingface.co/rows"


def parse_offsets(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a normalized labeled RAID subset.")
    parser.add_argument("--dataset", default="liamdugan/raid")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/raid"))
    parser.add_argument("--train-offsets", default="0,1000,100000,1000000,2000000,3000000")
    parser.add_argument("--extra-offsets", default="0,50000,250000")
    parser.add_argument("--block-length", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int, timeout: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    url = f"{API_URL}?{params}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for item in payload.get("rows", []):
        row = item.get("row", {})
        row["_row_idx"] = item.get("row_idx")
        row["_source_config"] = config
        row["_source_split"] = split
        row["_source_offset"] = offset
        rows.append(row)
    return rows


def normalize_rows(rows: Iterable[dict]) -> pd.DataFrame:
    normalized = []
    for row in rows:
        model = str(row.get("model") or "").strip()
        if not model:
            continue
        text = str(row.get("generation") or "").strip()
        if not text:
            continue
        label = "human" if model.lower() == "human" else "machine"
        attack = str(row.get("attack") or "none").strip() or "none"
        normalized.append(
            {
                "text": text,
                "label": label,
                "domain": str(row.get("domain") or "unknown"),
                "generator": model,
                "perturbation": attack,
                "decoding": "" if row.get("decoding") is None else str(row.get("decoding")),
                "repetition_penalty": "" if row.get("repetition_penalty") is None else str(row.get("repetition_penalty")),
                "title": "" if row.get("title") is None else str(row.get("title")),
                "source_dataset": "RAID",
                "source_split": str(row.get("_source_split")),
                "source_config": str(row.get("_source_config")),
                "source_offset": int(row.get("_source_offset")),
                "source_row_idx": int(row.get("_row_idx")),
                "source_id": str(row.get("id") or ""),
                "adv_source_id": "" if row.get("adv_source_id") is None else str(row.get("adv_source_id")),
                "raid_source_id": "" if row.get("source_id") is None else str(row.get("source_id")),
            }
        )
    return pd.DataFrame(normalized)


def balanced_cap(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = random.Random(seed)
    labels = sorted(df["label"].unique())
    per_label = max(1, max_rows // max(1, len(labels)))
    parts = []
    for label in labels:
        part = df[df["label"] == label]
        n = min(len(part), per_label)
        parts.append(part.sample(n=n, random_state=rng.randint(0, 1_000_000)))
    capped = pd.concat(parts, ignore_index=True)
    remaining = max_rows - len(capped)
    if remaining > 0:
        rest = df.drop(capped.index, errors="ignore")
        if len(rest):
            capped = pd.concat(
                [capped, rest.sample(n=min(remaining, len(rest)), random_state=rng.randint(0, 1_000_000))],
                ignore_index=True,
            )
    return capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def summary(df: pd.DataFrame) -> dict:
    fields = ["label", "domain", "generator", "perturbation", "source_split"]
    result = {"rows": int(len(df))}
    for field in fields:
        result[field + "s"] = df[field].value_counts(dropna=False).head(30).to_dict()
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    failures = []
    for offset in parse_offsets(args.train_offsets):
        try:
            collected.extend(fetch_rows(args.dataset, "raid", "train", offset, args.block_length, args.timeout))
        except Exception as exc:
            failures.append({"config": "raid", "split": "train", "offset": offset, "error": repr(exc)})
    for offset in parse_offsets(args.extra_offsets):
        try:
            collected.extend(fetch_rows(args.dataset, "raid", "extra", offset, args.block_length, args.timeout))
        except Exception as exc:
            failures.append({"config": "raid", "split": "extra", "offset": offset, "error": repr(exc)})

    df = normalize_rows(collected)
    if df.empty:
        raise SystemExit(f"No labeled RAID rows could be collected. Failures: {failures}")

    df = df.drop_duplicates(subset=["source_id", "text"]).reset_index(drop=True)
    df = balanced_cap(df, args.max_rows, args.seed)

    stratify_cols = df["label"] + "|" + df["domain"].astype(str)
    counts = stratify_cols.value_counts()
    stratify = stratify_cols if counts.min() >= 2 and len(counts) < len(df) else df["label"]
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    train_path = args.output_dir / "raid_train_small.csv"
    test_path = args.output_dir / "raid_test_small.csv"
    meta_path = args.output_dir / "metadata.json"
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    metadata = {
        "dataset": args.dataset,
        "method": "Hugging Face dataset viewer rows API",
        "source_note": "RAID public test split is unlabeled; this subset samples labeled train/extra rows.",
        "label_mapping": {"model == human": "human", "model != human": "machine"},
        "train_offsets": parse_offsets(args.train_offsets),
        "extra_offsets": parse_offsets(args.extra_offsets),
        "block_length": int(args.block_length),
        "max_rows": int(args.max_rows),
        "seed": int(args.seed),
        "failures": failures,
        "combined": summary(df),
        "train": {"path": str(train_path), **summary(train_df)},
        "test": {"path": str(test_path), **summary(test_df)},
        "limitations": [
            "Offset sampling is a lightweight pipeline validation strategy, not a statistically complete RAID benchmark.",
            "Final experiments should use a larger stratified sample or official full split processing.",
            "The unlabeled RAID test split cannot be used for supervised metrics without labels.",
        ],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps({"train": str(train_path), "test": str(test_path), "metadata": str(meta_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
