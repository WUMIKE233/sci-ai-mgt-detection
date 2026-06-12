"""Fine-tune a Transformer classifier for MGT detection.

This script provides a submission-oriented baseline stronger than TF-IDF and
frozen embeddings. It uses a small custom PyTorch loop so the outputs stay
transparent and reproducible.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


POSITIVE_LABELS = {"1", "true", "machine", "mgt", "generated", "ai", "ai_generated", "machine_generated"}
NEGATIVE_LABELS = {"0", "false", "human", "human_written", "human-written", "real", "authentic"}


class EncodedTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer classifier for MGT detection.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--group-cols", default="domain,generator,perturbation")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transformer_finetune"))
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
    args.output_dir = Path(args.output_dir)
    if isinstance(args.group_cols, list):
        args.group_cols = [str(col).strip() for col in args.group_cols if str(col).strip()]
    else:
        args.group_cols = [col.strip() for col in str(args.group_cols).split(",") if col.strip()]
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


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


def class_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_model(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs = []
    labels = []
    losses = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
            p = torch.softmax(output.logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.append(p)
            labels.append(batch["labels"].detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs), float(np.mean(losses)) if losses else 0.0


def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device, weights: torch.Tensor) -> float:
    model.train()
    losses = []
    for batch in loader:
        labels = batch["labels"].to(device)
        model_inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        optimizer.zero_grad(set_to_none=True)
        output = model(**model_inputs)
        loss = torch.nn.functional.cross_entropy(output.logits, labels, weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


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
            metrics.update({"model": "transformer_finetune", "group_col": col, "group_value": str(value)})
            rows.append(metrics)
    return rows


def save_state_dict_cpu(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def write_report(metrics: dict, output_path: Path) -> None:
    config = metrics["config"]
    values = metrics["test_metrics"]
    lines = [
        "# Fine-Tuned Transformer Report",
        "",
        "This report records a small-sample fine-tuned Transformer classifier baseline.",
        "",
        "## Configuration",
        "",
    ]
    for key in ["model_name", "max_length", "batch_size", "epochs", "learning_rate", "device", "train", "test"]:
        lines.append(f"- {key}: `{config[key]}`")
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
            "## Training History",
            "",
            "| Epoch | Train loss | Validation loss | Validation macro-F1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["history"]:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.4f} | {row['validation_loss']:.4f} | {row['validation_macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a small-sample fine-tuning run. Submission-grade claims require larger stratified samples, confidence intervals, and repeated seeds.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.train or not args.test:
        raise SystemExit("--train and --test are required, either directly or through --config.")

    set_seed(int(args.seed))
    device = resolve_device(args.device)

    train_df = read_table(args.train)
    test_df = read_table(args.test)
    require_columns(train_df, [args.text_col, args.label_col], "train")
    require_columns(test_df, [args.text_col, args.label_col], "test")

    y_all = normalize_labels(train_df[args.label_col])
    train_part, valid_part, y_train, y_valid = train_test_split(
        train_df,
        y_all,
        test_size=float(args.validation_size),
        random_state=int(args.seed),
        stratify=y_all,
    )
    y_test = normalize_labels(test_df[args.label_col])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        use_safetensors=True,
    )
    model.to(device)

    train_ds = EncodedTextDataset(train_part[args.text_col].fillna("").astype(str).tolist(), y_train, tokenizer, int(args.max_length))
    valid_ds = EncodedTextDataset(valid_part[args.text_col].fillna("").astype(str).tolist(), y_valid, tokenizer, int(args.max_length))
    test_ds = EncodedTextDataset(test_df[args.text_col].fillna("").astype(str).tolist(), y_test, tokenizer, int(args.max_length))

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_ds, batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    weights = class_weights(y_train, device)
    history = []
    best_state = None
    best_f1 = -1.0

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, weights)
        y_valid_eval, p_valid, valid_loss = evaluate_model(model, valid_loader, device)
        valid_metrics = compute_metrics(y_valid_eval, p_valid)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": valid_loss,
                "validation_macro_f1": valid_metrics["macro_f1"],
                "validation_accuracy": valid_metrics["accuracy"],
                "validation_auroc": valid_metrics["auroc"],
            }
        )
        if valid_metrics["macro_f1"] is not None and valid_metrics["macro_f1"] > best_f1:
            best_f1 = valid_metrics["macro_f1"]
            best_state = save_state_dict_cpu(model)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    y_test_eval, p_test, test_loss = evaluate_model(model, test_loader, device)
    test_metrics = compute_metrics(y_test_eval, p_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_df.copy()
    predictions["_label_binary"] = y_test_eval
    predictions["transformer_finetune_p_machine"] = p_test
    predictions["transformer_finetune_prediction"] = (p_test >= 0.5).astype(int)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(evaluate_groups(test_df, y_test_eval, p_test, args.group_cols)).to_csv(
        args.output_dir / "group_metrics.csv", index=False, encoding="utf-8"
    )

    metrics = {
        "config": {
            "train": str(args.train),
            "test": str(args.test),
            "model_name": str(args.model_name),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "validation_size": float(args.validation_size),
            "device": str(device),
            "fit_rows": int(len(train_part)),
            "validation_rows": int(len(valid_part)),
            "test_rows": int(len(test_df)),
            "seed": int(args.seed),
        },
        "history": history,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_report(metrics, args.output_dir / "transformer_finetune_report.md")

    print(json.dumps({"output_dir": str(args.output_dir), "device": str(device), "best_validation_macro_f1": best_f1}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
