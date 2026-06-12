# Historical Small-Sample Experiment Summary

Status: historical development log. The current manuscript uses locked SemEval-A and MAGE outputs summarized in `docs/empirical_experiment_summary.md`.

This table summarizes early real-data development experiments. These results are retained as pipeline history only; the submitted manuscript uses the locked full SemEval-A and MAGE evidence package summarized in `docs/empirical_experiment_summary.md`.

CSV source: `outputs\experiment_summary\preliminary_metrics.csv`

| Dataset | Run | Model | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |
|---|---|---|---:|---:|---:|---:|---:|
| MAGE small | TF-IDF baselines | logreg | 0.7094 | 0.7087 | 0.9210 | 0.1989 | 0.2508 |
| MAGE small | TF-IDF baselines | linear_svm | 0.7188 | 0.7180 | 0.9163 | 0.2031 | 0.2693 |
| MAGE small | Calibration/selective | identity | 0.6979 | 0.6967 | 0.9097 | 0.2042 | 0.2610 |
| MAGE small | Calibration/selective | temperature | 0.6979 | 0.6967 | 0.9097 | 0.2114 | 0.2605 |
| MAGE small | Calibration/selective | isotonic | 0.7990 | 0.7922 | 0.8910 | 0.2065 | 0.2512 |
| MAGE small | Frozen Transformer probe | distilbert_frozen_probe | 0.7990 | 0.7941 | 0.9317 | 0.1518 | 0.1909 |
| MAGE small | Fine-tuned Transformer | distilbert_finetuned | 0.8271 | 0.8209 | 0.9303 | 0.1508 | 0.1500 |
| RAID small | TF-IDF baselines | logreg | 0.8750 | 0.8461 | 0.9584 | 0.1151 | 0.2032 |
| RAID small | TF-IDF baselines | linear_svm | 0.9250 | 0.9011 | 0.9745 | 0.0564 | 0.0374 |
| RAID small | Calibration/selective | identity | 0.8792 | 0.8506 | 0.9456 | 0.1252 | 0.1905 |
| RAID small | Calibration/selective | temperature | 0.8792 | 0.8506 | 0.9456 | 0.0888 | 0.0669 |
| RAID small | Calibration/selective | isotonic | 0.8750 | 0.8184 | 0.9352 | 0.0862 | 0.0408 |
| RAID small | Frozen Transformer probe | distilbert_frozen_probe | 0.9417 | 0.9269 | 0.9774 | 0.0519 | 0.0622 |
| RAID small | Fine-tuned Transformer | distilbert_finetuned | 0.9375 | 0.9197 | 0.9809 | 0.0487 | 0.0432 |
| SemEval Subtask A mono small | TF-IDF baselines | logreg | 0.6138 | 0.5991 | 0.6865 | 0.2369 | 0.1122 |
| SemEval Subtask A mono small | TF-IDF baselines | linear_svm | 0.6422 | 0.6400 | 0.7050 | 0.2516 | 0.1648 |
| SemEval Subtask A mono small | Calibration/selective | identity | 0.6112 | 0.5992 | 0.6814 | 0.2343 | 0.0963 |
| SemEval Subtask A mono small | Calibration/selective | temperature | 0.6112 | 0.5992 | 0.6814 | 0.2796 | 0.2070 |
| SemEval Subtask A mono small | Calibration/selective | isotonic | 0.6204 | 0.6166 | 0.6805 | 0.2546 | 0.1623 |
| SemEval Subtask A mono small | Frozen Transformer probe | distilbert_frozen_probe | 0.5796 | 0.5383 | 0.6416 | 0.3238 | 0.2759 |
| SemEval Subtask A mono small | Fine-tuned Transformer | distilbert_finetuned | 0.6380 | 0.6184 | 0.7022 | 0.2870 | 0.2280 |

## Historical Strongest Rows

- MAGE small: fine-tuned DistilBERT had the highest accuracy among these early runs, but paraphrase failures remained visible.
- RAID small: frozen and fine-tuned DistilBERT were both strong; calibrated Linear SVM remained a competitive lightweight baseline.
- SemEval Subtask A monolingual dev: TF-IDF and DistilBERT baselines showed clear cross-generator difficulty because the dev generator was unseen in the sampled train set.

## Historical Limitations

- These early rows used small samples and did not define the current submitted claims.
- The current manuscript supersedes this file with full SemEval-A and MAGE processing, bootstrap intervals, subgroup analysis, and locked manuscript assets.
