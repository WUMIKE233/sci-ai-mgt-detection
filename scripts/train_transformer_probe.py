"""Train a frozen-transformer embedding probe for MGT detection.

This is a lightweight Transformer baseline: a pretrained encoder is frozen,
texts are embedded once, and Logistic Regression is trained on the resulting
vectors. It is designed for reproducible baseline evidence without the cost of
full fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoModel, AutoTokenizer


POSITIVE_LABELS = {"1", "true", "machine", "mgt", "generated", "ai", "ai_generated", "machine_generated"}
NEGATIVE_LABELS = {"0", "false", "human", "human_written", "human-written", "real", "authentic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen Transformer embedding probe.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--group-cols", default="domain,generator,perturbation")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/transformer_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transformer_probe"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.config:
        with args.config.open("r", encoding="utf-8") as f:
            config = json.load(f)
        for key, value in config.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, value)

    args.train = Path(args.train) if args.train else None
    args.test = Path(args.test) if args.test else None
    args.cache_dir = Path(args.cache_dir)
    args.output_dir = Path(args.output_dir)
    if isinstance(args.group_cols, list):
        args.group_cols = [str(col).strip() for col in args.group_cols if str(col).strip()]
    else:
        args.group_cols = [col.strip() for col in str(args.group_cols).split(",") if col.strip()]
    return args


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_labels(series: pd.Series) -> np.ndarray:
    labels = []
    unknown = set()
    for value in series:
        if pd.isna(value):
            unknown.add("<NA>")
            labels.append(None)
            continue
        if isinstance(value, (int, np.integer)) and value in {0, 1}:
            labels.append(int(value))
            continue
        if isinstance(value, (float, np.floating)) and value in {0.0, 1.0}:
            labels.append(int(value))
            continue
        token = str(value).strip().lower()
        if token in POSITIVE_LABELS:
            labels.append(1)
        elif token in NEGATIVE_LABELS:
            labels.append(0)
        else:
            unknown.add(str(value))
            labels.append(None)
    if unknown:
        raise ValueError(f"Unknown labels in input: {sorted(unknown)}")
    return np.asarray(labels, dtype=int)


def require_columns(df: pd.DataFrame, cols: Iterable[str], path_label: str) -> None:
    missing = [col for col in cols if col and col not in df.columns]
    if missing:
        raise ValueError(f"{path_label} missing required columns: {missing}")


def hash_texts(texts: list[str], model_name: str, pooling: str, max_length: int) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(pooling.encode("utf-8"))
    digest.update(str(max_length).encode("utf-8"))
    for text in texts:
        digest.update(b"\0")
        digest.update(text.encode("utf-8", errors="replace"))
    return digest.hexdigest()[:16]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            hidden = output.last_hidden_state
            if pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            embeddings.append(pooled.detach().cpu().numpy())
    return np.vstack(embeddings).astype(np.float32)


def load_or_embed(texts: list[str], split_name: str, args: argparse.Namespace, tokenizer, model, device: torch.device) -> np.ndarray:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hash_texts(texts, args.model_name, args.pooling, int(args.max_length))
    cache_path = args.cache_dir / f"{split_name}_{cache_key}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    embeddings = embed_texts(texts, tokenizer, model, device, int(args.batch_size), int(args.max_length), args.pooling)
    np.save(cache_path, embeddings)
    return embeddings


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


def safe_metric(fn, *args):
    try:
        value = fn(*args)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    except Exception:
        return None


def selective_curve(y_true: np.ndarray, p_pos: np.ndarray) -> list[dict]:
    y_pred = (p_pos >= 0.5).astype(int)
    confidence = np.maximum(p_pos, 1.0 - p_pos)
    order = np.argsort(-confidence)
    points = []
    for coverage in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        keep = max(1, int(math.ceil(len(order) * coverage)))
        idx = order[:keep]
        points.append(
            {
                "coverage": coverage,
                "risk": float(np.mean(y_pred[idx] != y_true[idx])),
                "confidence_threshold": float(confidence[idx[-1]]),
                "n_kept": int(keep),
            }
        )
    return points


def compute_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> dict:
    y_pred = (p_pos >= 0.5).astype(int)
    has_both_classes = len(np.unique(y_true)) == 2
    precision, recall, f1_pos, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "macro_f1": safe_metric(lambda a, b: f1_score(a, b, average="macro", zero_division=0), y_true, y_pred),
        "positive_precision": float(precision),
        "positive_recall": float(recall),
        "positive_f1": float(f1_pos),
        "auroc": safe_metric(roc_auc_score, y_true, p_pos) if has_both_classes else None,
        "auprc": safe_metric(average_precision_score, y_true, p_pos) if has_both_classes else None,
        "brier": safe_metric(brier_score_loss, y_true, p_pos),
        "ece_10": binary_ece(y_true, p_pos, n_bins=10),
        "selective_curve": selective_curve(y_true, p_pos),
    }


def evaluate_groups(df: pd.DataFrame, y_true: np.ndarray, p_pos: np.ndarray, group_cols: list[str]) -> list[dict]:
    rows = []
    eval_df = df.copy()
    eval_df["_y_true"] = y_true
    eval_df["_p_pos"] = p_pos
    for col in group_cols:
        if col not in eval_df.columns:
            continue
        for value, part in eval_df.groupby(col, dropna=False):
            metrics = compute_metrics(part["_y_true"].to_numpy(), part["_p_pos"].to_numpy())
            metrics.pop("selective_curve", None)
            metrics.update({"model": "transformer_probe", "group_col": col, "group_value": str(value)})
            rows.append(metrics)
    return rows


def write_report(metrics: dict, output_path: Path) -> None:
    lines = [
        "# Frozen Transformer Probe Report",
        "",
        "This is a lightweight Transformer baseline using frozen encoder embeddings plus Logistic Regression.",
        "",
        "## Configuration",
        "",
    ]
    for key in ["model_name", "pooling", "max_length", "device", "train", "test"]:
        lines.append(f"- {key}: `{metrics['config'][key]}`")
    values = metrics["metrics"]
    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            "| Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |",
            "|---:|---:|---:|---:|---:|---:|",
            "| {accuracy:.4f} | {macro_f1:.4f} | {auroc:.4f} | {auprc:.4f} | {brier:.4f} | {ece:.4f} |".format(
                accuracy=values["accuracy"],
                macro_f1=values["macro_f1"],
                auroc=values["auroc"],
                auprc=values["auprc"],
                brier=values["brier"],
                ece=values["ece_10"],
            ),
            "",
            "## Boundary",
            "",
            "This is not full fine-tuning. Submission-grade experiments should include a fine-tuned encoder or justify the frozen-probe baseline as a reproducible lightweight comparator.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.train or not args.test:
        raise SystemExit("--train and --test are required, either directly or through --config.")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    train_df = read_table(args.train)
    test_df = read_table(args.test)
    require_columns(train_df, [args.text_col, args.label_col], "train")
    require_columns(test_df, [args.text_col, args.label_col], "test")

    x_train = train_df[args.text_col].fillna("").astype(str).tolist()
    y_train = normalize_labels(train_df[args.label_col])
    x_test = test_df[args.text_col].fillna("").astype(str).tolist()
    y_test = normalize_labels(test_df[args.label_col])

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, use_safetensors=True)
    model.to(device)

    x_train_emb = load_or_embed(x_train, "train", args, tokenizer, model, device)
    x_test_emb = load_or_embed(x_test, "test", args, tokenizer, model, device)

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=int(args.seed))
    classifier.fit(x_train_emb, y_train)
    p_pos = classifier.predict_proba(x_test_emb)[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_df.copy()
    predictions["_label_binary"] = y_test
    predictions["transformer_probe_p_machine"] = p_pos
    predictions["transformer_probe_prediction"] = (p_pos >= 0.5).astype(int)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8")

    metrics = {
        "config": {
            "train": str(args.train),
            "test": str(args.test),
            "model_name": str(args.model_name),
            "pooling": str(args.pooling),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "metrics": compute_metrics(y_test, p_pos),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(evaluate_groups(test_df, y_test, p_pos, args.group_cols)).to_csv(
        args.output_dir / "group_metrics.csv", index=False, encoding="utf-8"
    )
    write_report(metrics, args.output_dir / "transformer_probe_report.md")

    print(json.dumps({"output_dir": str(args.output_dir), "device": str(device)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
