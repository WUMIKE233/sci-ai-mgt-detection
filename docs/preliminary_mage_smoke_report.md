# Historical MAGE Pipeline Smoke Report

Status: historical development log. This file records an early smoke test and is not the evidence base for the current submission-scope manuscript.

Date: 2026-06-12

## Purpose

This report records a small real-data pipeline validation run. It is a historical smoke test and must not be copied into the Results section as evidence of the submitted manuscript.

## Data

Source dataset: MAGE (`yaful/MAGE`) on Hugging Face.

Verified metadata:

- License: Apache-2.0 according to the Hugging Face dataset card.
- Repository SHA: `342663f0a2b775455c023f5d36a1341ff0ec5402`
- Downloaded local files:
  - `data/raw/mage/test_ood_set_gpt.csv`
  - `data/raw/mage/test_ood_set_gpt_para.csv`

Important label mapping:

- MAGE `label=1` means human text.
- MAGE `label=0` means machine-generated or machine-paraphrased text.
- The project-normalized label maps `label=0` to `machine` so that `machine` remains the positive class in metrics.

Normalized files:

- `data/processed/mage/mage_train_small.csv`
- `data/processed/mage/mage_test_ood_para_small.csv`
- `data/processed/mage/metadata.json`

The small training subset contains 960 rows, balanced between 480 human and 480 machine examples. The small test subset contains 960 rows: 320 human, 320 GPT-4 paraphrase/generated examples, and 320 machine-paraphrased human examples.

## Command

```powershell
python scripts\prepare_mage_subset.py
python scripts\train_baselines.py --config configs\baseline.mage_small.json
```

## Output

- `outputs/mage_small_baseline/metrics.json`
- `outputs/mage_small_baseline/predictions.csv`
- `outputs/mage_small_baseline/group_metrics.csv`

## Smoke-Test Metrics

| Model | Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.7094 | 0.7087 | 0.9210 | 0.9627 | 0.1989 | 0.2508 |
| TF-IDF + Calibrated Linear SVM | 0.7188 | 0.7180 | 0.9163 | 0.9628 | 0.2031 | 0.2693 |

## Observed Failure Pattern

The small test set suggests a useful stress-test pattern: ordinary GPT-4 paraphrased/generated examples were often detected as machine, while machine-paraphrased human-source examples were frequently misclassified. In this small sample, the `machine_paraphrase` group had much lower accuracy than the `gpt4` group for both baselines.

This pattern supported the paper's motivation that MGT detection should evaluate robustness and uncertainty under paraphrase-like distribution shift. It is now superseded by the locked full SemEval-A and MAGE evidence package.

## Limitations

- The training data is sampled from an OOD file, not the official full training split.
- The test distribution is intentionally stress-heavy and class-imbalanced.
- Only TF-IDF baselines were evaluated.
- No confidence intervals were computed.
- No calibration method beyond the SVM probability wrapper was evaluated.
- No full RAID or SemEval/M4 experiments were run.

## Historical Follow-up Plan

1. Full MAGE train/valid/test files were later downloaded and used for the locked submission-scope evidence.
2. SemEval-A conversion was added and used for the locked submission-scope evidence.
3. RAID was scoped to related-work and repository context rather than main empirical evidence.
4. Transformer baseline runs were retained as historical development checks.
5. Post-hoc calibration and selective prediction were added as explicit methods.
6. Reliability diagrams and coverage-risk plots were generated for the current empirical assets.

## Follow-up Completed

The first post-hoc calibration and selective prediction pipeline was implemented after this smoke test. Current submission evidence is summarized in `docs/empirical_experiment_summary.md`.
