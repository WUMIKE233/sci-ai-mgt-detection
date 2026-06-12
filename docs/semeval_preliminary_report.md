# SemEval-2024 Subtask A Monolingual Preliminary Report

Status: historical development log. The current manuscript uses full locked SemEval-A monolingual and multilingual train-dev experiments.

## Purpose

This report records the first official SemEval-2024 Task 8 Subtask A monolingual run in this workspace. It closes the previous gap where SemEval/M4 existed only as a plan.

## Data

- Official source: `https://github.com/mbzuai-nlp/SemEval2024-task8`
- Download method used: `python -m gdown --folder https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc`
- Completed local files:
  - `data/raw/semeval2024_task8/subtaskA/SubtaskA/subtaskA_train_monolingual.jsonl`
  - `data/raw/semeval2024_task8/subtaskA/SubtaskA/subtaskA_dev_monolingual.jsonl`
- Incomplete local file after timeout:
  - `data/raw/semeval2024_task8/subtaskA/SubtaskA/subtaskA_train_multilingual.jsonl*.part`
- Converted subset:
  - Train: `data/processed/semeval2024_task8/semeval_subtaskA_monolingual_train.csv`
  - Dev: `data/processed/semeval2024_task8/semeval_subtaskA_monolingual_dev.csv`
  - Metadata: `data/processed/semeval2024_task8/semeval_subtaskA_monolingual_metadata.json`

## Sampling and Split Boundary

The converter used `--max-train-per-group 500 --max-dev-per-group 500`, grouped by label, source, and model. This produced:

- Train: 12,500 rows
- Dev: 5,000 rows
- Train generators: human, davinci, chatGPT, dolly, cohere
- Dev generators: human, bloomz

This split is useful for cross-generator stress testing because `bloomz` appears in dev but not in the sampled training set.

## Preliminary Metrics

| Run | Model | N | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |
|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF baselines | logreg | 5000 | 0.6138 | 0.5991 | 0.6865 | 0.2369 | 0.1122 |
| TF-IDF baselines | linear_svm | 5000 | 0.6422 | 0.6400 | 0.7050 | 0.2516 | 0.1648 |
| Calibration/selective | identity | 5000 | 0.6112 | 0.5992 | 0.6814 | 0.2343 | 0.0963 |
| Calibration/selective | temperature | 5000 | 0.6112 | 0.5992 | 0.6814 | 0.2796 | 0.2070 |
| Calibration/selective | isotonic | 5000 | 0.6204 | 0.6166 | 0.6805 | 0.2546 | 0.1623 |
| Frozen Transformer probe | DistilBERT mean-pool probe | 5000 | 0.5796 | 0.5383 | 0.6416 | 0.3238 | 0.2759 |
| Fine-tuned Transformer | DistilBERT sequence classifier | 5000 | 0.6380 | 0.6184 | 0.7022 | 0.2870 | 0.2280 |

## Preliminary Interpretation

The low-to-mid 0.6 accuracy range is consistent with a difficult cross-generator setting. The result supports the paper's motivation that MGT detectors should be evaluated under unseen-generator distribution shift, not only on aggregate in-domain accuracy.

Temperature scaling did not improve calibration in this run. The uncalibrated logistic model had lower ECE than the temperature-scaled variant, which is a useful negative result for the calibration section.

Frozen DistilBERT embeddings underperformed the TF-IDF SVM in this setting, and a one-epoch fine-tuned DistilBERT sequence classifier reached similar but not clearly superior aggregate performance. This is useful negative evidence: stronger neural architectures do not automatically solve generator transfer when the evaluation generator differs from the sampled training generators.

## Repeated-Seed Stability

The low-cost SemEval TF-IDF and calibration pipeline was repeated with seeds 13, 42, and 2026. The seed changes the sampled training subset and calibration split while the sampled dev set remains fixed. The TF-IDF Linear SVM accuracy mean was 0.6437 with standard deviation 0.0022, and the logistic-regression baseline accuracy mean was 0.6126 with standard deviation 0.0013. This suggests the low-cost SemEval signal is stable across these sampled training seeds, although it does not replace repeated-seed Transformer experiments.

Full details are in `docs/semeval_seed_sweep_report.md`.

## Current Boundary

This is still not a final SemEval/M4 result. Final submission needs a locked full-processing protocol, repeated seeds, stronger encoder decisions, official test/gold handling where permitted, and regeneration of all tables and figures from locked prediction files.
