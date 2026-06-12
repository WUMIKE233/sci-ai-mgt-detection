# Fine-Tuned Transformer Preliminary Report

Date: 2026-06-12

## Purpose

This report records the first small-sample fine-tuned Transformer baseline for the SCI manuscript project. It upgrades the model evidence beyond TF-IDF and frozen embeddings.

The current run fine-tunes `distilbert-base-uncased` for binary human-vs-machine text classification.

## Environment

Successful runs used:

```powershell
W:\PY311\python.exe
```

Verified environment from prior checks:

- Python 3.11.7
- torch 2.5.1+cu118
- CUDA available: True
- transformers 4.56.2
- sklearn 1.7.2

## Commands

```powershell
W:\PY311\python.exe scripts\train_transformer_finetune.py --config configs\transformer_finetune.raid_small.json
W:\PY311\python.exe scripts\train_transformer_finetune.py --config configs\transformer_finetune.mage_small.json
```

## RAID Labeled Offset Sample

Output directory:

- `outputs/raid_small_transformer_finetune/`

Configuration:

- Fit rows: 448
- Validation rows: 112
- Test rows: 240
- Epochs: 2
- Batch size: 8
- Learning rate: 2e-5

| Model | Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned DistilBERT | 0.9375 | 0.9197 | 0.9809 | 0.9938 | 0.0487 | 0.0432 |

Training history:

| Epoch | Train loss | Validation loss | Validation macro-F1 |
|---:|---:|---:|---:|
| 1 | 0.4749 | 0.2110 | 0.9025 |
| 2 | 0.2318 | 0.0867 | 0.9535 |

Preliminary RAID notes:

- Fine-tuned DistilBERT reaches similar accuracy to the frozen probe but improves AUROC and ECE in this sampled split.
- The `code` domain remains the lowest-performing domain.
- The `synonym` perturbation remains the hardest perturbation in this sample.
- The `mistral-chat` group remains a weak point.

## MAGE OOD/Paraphrase Subset

Output directory:

- `outputs/mage_small_transformer_finetune/`

Configuration:

- Fit rows: 768
- Validation rows: 192
- Test rows: 960
- Epochs: 2
- Batch size: 8
- Learning rate: 2e-5

| Model | Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned DistilBERT | 0.8271 | 0.8209 | 0.9303 | 0.9693 | 0.1508 | 0.1500 |

Training history:

| Epoch | Train loss | Validation loss | Validation macro-F1 |
|---:|---:|---:|---:|
| 1 | 0.4872 | 0.2351 | 0.9108 |
| 2 | 0.1224 | 0.2110 | 0.9427 |

Preliminary MAGE notes:

- Fine-tuning improves MAGE small-subset accuracy over the frozen DistilBERT probe.
- `machine_paraphrase` remains a major failure mode, with group accuracy 0.5656.
- `pubmed_human_para` is the most severe observed subgroup failure, with group accuracy 0.1250.
- The gap between validation performance and OOD/paraphrase test performance supports the paper's distribution-shift framing.

## Interpretation Boundary

These results strengthen the manuscript because a real fine-tuned encoder baseline is now available. They still remain preliminary because:

- The datasets are small sampled subsets.
- Only one random seed has been run.
- No confidence intervals have been computed.
- Full MAGE/RAID processing and SemEval/M4 conversion remain required.
- The DistilBERT baseline is adequate for a first baseline, but a stronger encoder such as RoBERTa/DeBERTa may be needed for final SCI submission depending on compute budget.
