# SemEval Seed Sweep Report

Status: historical development log. The current manuscript uses locked full SemEval-A and MAGE empirical outputs.

This report repeats the low-cost SemEval-2024 Task 8 Subtask A monolingual small-sample pipeline across sampling seeds. It is preliminary stability evidence, not final benchmark evidence.

Seeds: `13, 42, 2026`

## Per-Seed Metrics

| Seed | Run | Model | N | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 13 | Calibration/selective | identity | 5000 | 0.6112 | 0.5983 | 0.6814 | 0.2345 | 0.0986 |
| 42 | Calibration/selective | identity | 5000 | 0.6112 | 0.5992 | 0.6814 | 0.2343 | 0.0963 |
| 2026 | Calibration/selective | identity | 5000 | 0.6134 | 0.5985 | 0.6810 | 0.2356 | 0.1020 |
| 13 | Calibration/selective | isotonic | 5000 | 0.6192 | 0.6190 | 0.6801 | 0.2631 | 0.1800 |
| 42 | Calibration/selective | isotonic | 5000 | 0.6204 | 0.6166 | 0.6805 | 0.2546 | 0.1623 |
| 2026 | Calibration/selective | isotonic | 5000 | 0.6262 | 0.6262 | 0.6804 | 0.2610 | 0.1773 |
| 13 | Calibration/selective | temperature | 5000 | 0.6112 | 0.5983 | 0.6814 | 0.2846 | 0.2175 |
| 42 | Calibration/selective | temperature | 5000 | 0.6112 | 0.5992 | 0.6814 | 0.2796 | 0.2070 |
| 2026 | Calibration/selective | temperature | 5000 | 0.6134 | 0.5985 | 0.6810 | 0.2844 | 0.2190 |
| 13 | TF-IDF baselines | linear_svm | 5000 | 0.6428 | 0.6409 | 0.7056 | 0.2512 | 0.1641 |
| 42 | TF-IDF baselines | linear_svm | 5000 | 0.6422 | 0.6400 | 0.7050 | 0.2516 | 0.1648 |
| 2026 | TF-IDF baselines | linear_svm | 5000 | 0.6462 | 0.6438 | 0.7053 | 0.2500 | 0.1590 |
| 13 | TF-IDF baselines | logreg | 5000 | 0.6128 | 0.5979 | 0.6845 | 0.2373 | 0.1110 |
| 42 | TF-IDF baselines | logreg | 5000 | 0.6138 | 0.5991 | 0.6865 | 0.2369 | 0.1122 |
| 2026 | TF-IDF baselines | logreg | 5000 | 0.6112 | 0.5940 | 0.6827 | 0.2392 | 0.1161 |

## Across-Seed Summary

| Run | Model | Accuracy Mean | Accuracy SD | Macro-F1 Mean | Macro-F1 SD | AUROC Mean | AUROC SD | ECE-10 Mean | ECE-10 SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibration/selective | identity | 0.6119 | 0.0013 | 0.5987 | 0.0005 | 0.6813 | 0.0002 | 0.0990 | 0.0029 |
| Calibration/selective | isotonic | 0.6219 | 0.0037 | 0.6206 | 0.0050 | 0.6803 | 0.0002 | 0.1732 | 0.0096 |
| Calibration/selective | temperature | 0.6119 | 0.0013 | 0.5987 | 0.0005 | 0.6813 | 0.0002 | 0.2145 | 0.0065 |
| TF-IDF baselines | linear_svm | 0.6437 | 0.0022 | 0.6416 | 0.0020 | 0.7053 | 0.0003 | 0.1626 | 0.0032 |
| TF-IDF baselines | logreg | 0.6126 | 0.0013 | 0.5970 | 0.0027 | 0.6846 | 0.0019 | 0.1131 | 0.0027 |

## Interpretation Boundary

- The seed changes the sampled training subset and the calibration split.
- The dev set remains the same because the current `max-dev-per-group=500` covers the sampled monolingual dev groups.
- Final submission still requires locked full processing, more seeds for neural baselines, and journal-ready statistical analysis.
