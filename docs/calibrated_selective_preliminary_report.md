# Calibrated Selective Prediction Preliminary Report

Status: historical development log. This report records early pipeline validation and is not the evidence base for the current submission-scope manuscript.

Date: 2026-06-12

## Purpose

This report records the first end-to-end calibration and selective prediction run on a small MAGE subset. It is a real-data pipeline validation, but it is not yet a submission-grade result.

## Command

```powershell
python scripts\run_calibrated_selective.py --config configs\calibrated_selective.mage_small.json
```

## Data Split

- Input training subset: `data/processed/mage/mage_train_small.csv`
- Input test subset: `data/processed/mage/mage_test_ood_para_small.csv`
- Fit split: 720 rows
- Calibration split: 240 rows
- Test split: 960 rows
- Calibration split size: 0.25
- Random seed: 42

The test split is an OOD/paraphrase stress subset, not a representative final evaluation set.

## Outputs

- `outputs/mage_small_calibrated_selective/metrics.json`
- `outputs/mage_small_calibrated_selective/predictions.csv`
- `outputs/mage_small_calibrated_selective/group_metrics.csv`
- `outputs/mage_small_calibrated_selective/reliability_bins.csv`
- `outputs/mage_small_calibrated_selective/coverage_risk.csv`
- `outputs/mage_small_calibrated_selective/reliability_diagram.png`
- `outputs/mage_small_calibrated_selective/coverage_risk.png`
- `outputs/mage_small_calibrated_selective/calibration_report.md`

## Test Metrics

| Method | Accuracy | Macro-F1 | AUROC | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| Identity / uncalibrated probability | 0.6979 | 0.6967 | 0.9097 | 0.9526 | 0.2042 | 0.2610 |
| Temperature scaling | 0.6979 | 0.6967 | 0.9097 | 0.9526 | 0.2114 | 0.2605 |
| Isotonic calibration | 0.7990 | 0.7922 | 0.8910 | 0.9302 | 0.2065 | 0.2512 |

## Interpretation Boundary

The small run suggests that isotonic calibration can change the operating point and reduce selective risk in this sampled OOD/paraphrase setting, but it does not prove a general performance gain. AUROC decreased for isotonic calibration, and ECE changed only modestly. The paper should therefore frame calibration as a reliability and decision-policy component rather than a guaranteed accuracy booster.

## Conformal/Selective Findings

At alpha 0.10, isotonic conformal prediction produced:

- Empirical coverage: 0.8198
- Singleton rate: 0.7688
- Ambiguous/abstain rate: 0.2312
- Risk on singletons: 0.2344

This supports the planned selective-detection discussion: abstention can expose uncertainty, but coverage and risk must be reported together.

## Required Upgrade Before Submission

1. Use full MAGE train/validation/test or a justified stratified sample.
2. Add SemEval/M4 and RAID conversions.
3. Run transformer baselines, not only TF-IDF logistic regression.
4. Bootstrap confidence intervals for each metric.
5. Generate journal-ready figures with final captions.
6. Use held-out calibration splits that are independent of final test sets.
