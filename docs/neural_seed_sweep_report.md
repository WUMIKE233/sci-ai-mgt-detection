# Neural Seed Sweep Report

Status: historical development log. The current submission-scope manuscript is bounded to TF-IDF Logistic Regression reliability analysis unless stronger full-scope neural baselines are added later.

This report repeats DistilBERT neural baselines on the SemEval-2024 Task 8 Subtask A monolingual small-sample setup. It is a historical repeated-seed stability check and is not used as current submission-scope evidence.

## Configuration

- Seeds: `13, 42, 2026`
- Model: `distilbert-base-uncased`
- Max length: `256`
- Fine-tuning epochs: `1`
- Max train rows per label/generator group: `500`
- Max dev rows per label/generator group: `500`
- Device setting: `auto`

## Per-Seed Metrics

| Seed | Run | Model | N | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 13 | Fine-tuned Transformer | distilbert_finetuned | 5000 | 0.6914 | 0.6892 | 0.7722 | 0.2546 | 0.2227 |
| 42 | Fine-tuned Transformer | distilbert_finetuned | 5000 | 0.6374 | 0.6182 | 0.7009 | 0.2872 | 0.2283 |
| 2026 | Fine-tuned Transformer | distilbert_finetuned | 5000 | 0.6556 | 0.6552 | 0.7122 | 0.2740 | 0.2252 |
| 13 | Frozen Transformer probe | distilbert_frozen_probe | 5000 | 0.5776 | 0.5382 | 0.6628 | 0.3162 | 0.2737 |
| 42 | Frozen Transformer probe | distilbert_frozen_probe | 5000 | 0.5796 | 0.5383 | 0.6416 | 0.3238 | 0.2759 |
| 2026 | Frozen Transformer probe | distilbert_frozen_probe | 5000 | 0.5676 | 0.5227 | 0.6480 | 0.3233 | 0.2757 |

## Across-Seed Summary

| Run | Model | Seeds | Accuracy Mean | Accuracy SD | Macro-F1 Mean | Macro-F1 SD | AUROC Mean | AUROC SD | ECE-10 Mean | ECE-10 SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fine-tuned Transformer | distilbert_finetuned | 3 | 0.6615 | 0.0275 | 0.6542 | 0.0355 | 0.7284 | 0.0383 | 0.2254 | 0.0028 |
| Frozen Transformer probe | distilbert_frozen_probe | 3 | 0.5749 | 0.0064 | 0.5331 | 0.0090 | 0.6508 | 0.0109 | 0.2751 | 0.0012 |

## Historical Interpretation Boundary

- The seed changes the sampled training subset, validation split, and model initialization.
- The current dev sample is shared across seeds when the configured cap covers the available monolingual dev groups.
- These results support pipeline stability checks only. The submitted manuscript instead relies on locked full SemEval-A and MAGE processing, journal-specific reporting, and bootstrap confidence intervals.
