# RAID Labeled Subset Preliminary Report

Status: historical development log. RAID is cited only for related-work and data-source context in the current submission-scope manuscript, not for main empirical claims.

Date: 2026-06-12

## Purpose

This report records the first RAID labeled-subset pipeline validation. It adds an adversarial/robustness-oriented dataset to the project without downloading RAID's multi-GB CSV files.

This is a real-data pre-experiment, not a final submission-grade result.

## Dataset Boundary

RAID provides three relevant splits on Hugging Face:

- `raid/train`: labeled, includes metadata.
- `raid/extra`: labeled, includes metadata for extra domains such as code, Czech, and German.
- `raid_test/test`: unlabeled blind-test style split with only `id` and `generation`.

Therefore, supervised metrics in this project must come from `train` and `extra` unless official labels are obtained for the public test split.

## Sampling Method

Command:

```powershell
python scripts\prepare_raid_subset.py
```

The script uses the Hugging Face dataset viewer rows API with multiple offsets rather than downloading full files. It maps:

- `model == human` to `label=human`
- `model != human` to `label=machine`

Generated files:

- `data/processed/raid/raid_train_small.csv`
- `data/processed/raid/raid_test_small.csv`
- `data/processed/raid/metadata.json`

## Subset Composition

Combined subset:

- Rows: 800
- Labels: 600 machine, 200 human
- Domains: abstracts, code, Czech, books, news
- Perturbations: none, insert_paragraphs, synonym, perplexity_misspelling
- Source splits: 500 from `train`, 300 from `extra`

Train/test split:

- Train: 560 rows
- Test: 240 rows

## Baseline Results

Command:

```powershell
python scripts\train_baselines.py --config configs\baseline.raid_small.json
```

| Model | Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8750 | 0.8461 | 0.9584 | 0.9864 | 0.1151 | 0.2032 |
| TF-IDF + Calibrated Linear SVM | 0.9250 | 0.9011 | 0.9745 | 0.9918 | 0.0564 | 0.0374 |

## Calibration and Selective Prediction Results

Command:

```powershell
python scripts\run_calibrated_selective.py --config configs\calibrated_selective.raid_small.json
```

| Method | Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| Identity / uncalibrated probability | 0.8792 | 0.8506 | 0.9456 | 0.9823 | 0.1252 | 0.1905 |
| Temperature scaling | 0.8792 | 0.8506 | 0.9456 | 0.9823 | 0.0888 | 0.0669 |
| Isotonic calibration | 0.8750 | 0.8184 | 0.9352 | 0.9704 | 0.0862 | 0.0408 |

Generated outputs:

- `outputs/raid_small_baseline/metrics.json`
- `outputs/raid_small_baseline/group_metrics.csv`
- `outputs/raid_small_calibrated_selective/metrics.json`
- `outputs/raid_small_calibrated_selective/reliability_diagram.png`
- `outputs/raid_small_calibrated_selective/coverage_risk.png`

## Preliminary Failure Patterns

- The small sample is easier overall than the MAGE paraphrase stress subset.
- `synonym` and `perplexity_misspelling` perturbations are harder than `insert_paragraphs` in this sample.
- The `code` domain shows lower baseline performance than books/news/Czech in this sampled split.
- Temperature scaling substantially reduces ECE in this small sample without changing the decision threshold outputs.

## Interpretation Boundary

These observations support the manuscript's planned focus on robustness, attack-specific analysis, and calibration. They must remain preliminary because:

- The subset is offset-sampled rather than statistically complete.
- The public RAID test split is unlabeled.
- Full benchmark results require larger stratified processing over RAID train/extra or official evaluation labels.
- Only TF-IDF baselines were run; transformer baselines remain required.
