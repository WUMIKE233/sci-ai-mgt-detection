# Historical Frozen Transformer Probe Report

Status: historical development log. This report records early small-sample neural experiments; the current submitted claims are bounded to the locked TF-IDF Logistic Regression reliability study.

Date: 2026-06-12

## Purpose

This report records the first Transformer-based baseline for the SCI manuscript project. The goal is to move beyond TF-IDF baselines while keeping the experiment lightweight and reproducible.

The current method is a **frozen encoder probe**, not full fine-tuning:

1. Load `distilbert-base-uncased` with safetensors.
2. Embed each text using mean pooling over the last hidden state.
3. Train Logistic Regression on the frozen embeddings.
4. Evaluate the same metrics used by the TF-IDF baselines.

## Environment

The default Python 3.14 environment does not currently have `torch`. The successful Transformer runs used:

```powershell
W:\PY311\python.exe
```

Verified runtime:

- Python 3.11.7
- torch 2.5.1+cu118
- CUDA available: True
- transformers 4.56.2
- sklearn 1.7.2

`prajjwal1/bert-tiny` was not used because the current `transformers` version refuses to load PyTorch `.bin` weights under torch 2.5.1 after the `torch.load` security restriction. `distilbert-base-uncased` loads through safetensors and avoids that issue.

## Commands

```powershell
W:\PY311\python.exe scripts\train_transformer_probe.py --config configs\transformer_probe.raid_small.json
W:\PY311\python.exe scripts\train_transformer_probe.py --config configs\transformer_probe.mage_small.json
```

## RAID Labeled Offset Sample

Output directory:

- `outputs/raid_small_transformer_probe/`

| Model | Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| DistilBERT frozen mean-pooling probe | 0.9417 | 0.9269 | 0.9774 | 0.9934 | 0.0519 | 0.0622 |

Historical notes:

- The frozen Transformer probe outperformed the TF-IDF Logistic Regression and TF-IDF calibrated Linear SVM baselines on this RAID sample.
- The `code` domain remained harder than books/news/Czech in the sampled split.
- `synonym` and `perplexity_misspelling` attacks remained harder than `insert_paragraphs`.

## MAGE OOD/Paraphrase Subset

Output directory:

- `outputs/mage_small_transformer_probe/`

| Model | Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| DistilBERT frozen mean-pooling probe | 0.7990 | 0.7941 | 0.9317 | 0.9681 | 0.1518 | 0.1909 |

Historical notes:

- The frozen Transformer probe improved over the TF-IDF baselines on the MAGE stress subset.
- `machine_paraphrase` remained the main failure mode, with group accuracy 0.4781.
- This supports the manuscript's planned focus on paraphrase-like distribution shift and selective prediction.

## Interpretation Boundary

These results are bounded to early development checks because:

- The encoder is frozen and not fine-tuned.
- The data are small sampled subsets.
- No bootstrap confidence intervals have been computed.
- They are not used as current submission-scope evidence.

The results are nevertheless useful because they confirm that Transformer-based baselines can run on this machine and that the same evaluation protocol can be applied to MAGE and RAID in future work.
