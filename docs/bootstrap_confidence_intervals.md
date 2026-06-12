# Bootstrap Confidence Intervals

Status: historical development log. The current manuscript uses the locked empirical statistics listed in `docs/empirical_experiment_summary.md` and `outputs/locked/statistical_analysis/`.

This report gives 95% bootstrap confidence intervals for early sampled prediction files. It is retained as pipeline history only; the submitted manuscript uses the locked statistics in `outputs/locked/statistical_analysis/`.

Bootstrap resamples per run: `1000`
CSV source: `outputs\statistical_analysis\bootstrap_confidence_intervals.csv`

| Dataset | Run | Model | N | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MAGE small | TF-IDF baselines | logreg | 960 | 0.7094 [0.6792, 0.7375] | 0.7087 [0.6789, 0.7371] | 0.9210 [0.9038, 0.9365] | 0.1989 [0.1922, 0.2054] | 0.2508 [0.2346, 0.2759] |
| MAGE small | TF-IDF baselines | linear_svm | 960 | 0.7188 [0.6896, 0.7469] | 0.7180 [0.6891, 0.7463] | 0.9163 [0.8999, 0.9327] | 0.2031 [0.1853, 0.2218] | 0.2693 [0.2468, 0.2927] |
| MAGE small | Calibration/selective | identity | 960 | 0.6979 [0.6677, 0.7271] | 0.6967 [0.6659, 0.7259] | 0.9097 [0.8904, 0.9285] | 0.2042 [0.1984, 0.2100] | 0.2610 [0.2365, 0.2825] |
| MAGE small | Calibration/selective | temperature | 960 | 0.6979 [0.6677, 0.7271] | 0.6967 [0.6659, 0.7259] | 0.9097 [0.8904, 0.9285] | 0.2114 [0.1927, 0.2310] | 0.2605 [0.2365, 0.2850] |
| MAGE small | Calibration/selective | isotonic | 960 | 0.7990 [0.7739, 0.8240] | 0.7922 [0.7648, 0.8185] | 0.8910 [0.8694, 0.9098] | 0.2065 [0.1880, 0.2263] | 0.2512 [0.2286, 0.2756] |
| MAGE small | Frozen Transformer probe | distilbert_frozen_probe | 960 | 0.7990 [0.7729, 0.8260] | 0.7941 [0.7674, 0.8221] | 0.9317 [0.9155, 0.9460] | 0.1518 [0.1341, 0.1695] | 0.1909 [0.1693, 0.2118] |
| MAGE small | Fine-tuned Transformer | distilbert_finetuned | 960 | 0.8271 [0.8010, 0.8531] | 0.8209 [0.7949, 0.8471] | 0.9303 [0.9116, 0.9452] | 0.1508 [0.1277, 0.1738] | 0.1500 [0.1280, 0.1750] |
| RAID small | TF-IDF baselines | logreg | 240 | 0.8750 [0.8333, 0.9125] | 0.8461 [0.7922, 0.8926] | 0.9584 [0.9330, 0.9792] | 0.1151 [0.1014, 0.1280] | 0.2032 [0.1727, 0.2372] |
| RAID small | TF-IDF baselines | linear_svm | 240 | 0.9250 [0.8917, 0.9583] | 0.9011 [0.8544, 0.9405] | 0.9745 [0.9555, 0.9885] | 0.0564 [0.0392, 0.0752] | 0.0374 [0.0342, 0.0765] |
| RAID small | Calibration/selective | identity | 240 | 0.8792 [0.8375, 0.9167] | 0.8506 [0.7990, 0.8962] | 0.9456 [0.9175, 0.9699] | 0.1252 [0.1116, 0.1380] | 0.1905 [0.1694, 0.2285] |
| RAID small | Calibration/selective | temperature | 240 | 0.8792 [0.8375, 0.9167] | 0.8506 [0.7990, 0.8962] | 0.9456 [0.9175, 0.9699] | 0.0888 [0.0675, 0.1106] | 0.0669 [0.0561, 0.1145] |
| RAID small | Calibration/selective | isotonic | 240 | 0.8750 [0.8333, 0.9167] | 0.8184 [0.7545, 0.8733] | 0.9352 [0.9040, 0.9618] | 0.0862 [0.0603, 0.1133] | 0.0408 [0.0237, 0.0821] |
| RAID small | Frozen Transformer probe | distilbert_frozen_probe | 240 | 0.9417 [0.9083, 0.9667] | 0.9269 [0.8896, 0.9600] | 0.9774 [0.9593, 0.9913] | 0.0519 [0.0355, 0.0712] | 0.0622 [0.0466, 0.0926] |
| RAID small | Fine-tuned Transformer | distilbert_finetuned | 240 | 0.9375 [0.9042, 0.9667] | 0.9197 [0.8778, 0.9548] | 0.9809 [0.9660, 0.9931] | 0.0487 [0.0287, 0.0714] | 0.0432 [0.0272, 0.0741] |
| SemEval Subtask A mono small | TF-IDF baselines | logreg | 5000 | 0.6138 [0.5998, 0.6266] | 0.5991 [0.5859, 0.6120] | 0.6865 [0.6720, 0.7003] | 0.2369 [0.2315, 0.2425] | 0.1122 [0.0993, 0.1271] |
| SemEval Subtask A mono small | TF-IDF baselines | linear_svm | 5000 | 0.6422 [0.6286, 0.6548] | 0.6400 [0.6262, 0.6520] | 0.7050 [0.6910, 0.7184] | 0.2516 [0.2441, 0.2598] | 0.1648 [0.1536, 0.1786] |
| SemEval Subtask A mono small | Calibration/selective | identity | 5000 | 0.6112 [0.5970, 0.6248] | 0.5992 [0.5856, 0.6129] | 0.6814 [0.6671, 0.6958] | 0.2343 [0.2295, 0.2395] | 0.0963 [0.0837, 0.1103] |
| SemEval Subtask A mono small | Calibration/selective | temperature | 5000 | 0.6112 [0.5970, 0.6248] | 0.5992 [0.5856, 0.6129] | 0.6814 [0.6671, 0.6958] | 0.2796 [0.2708, 0.2890] | 0.2070 [0.1955, 0.2222] |
| SemEval Subtask A mono small | Calibration/selective | isotonic | 5000 | 0.6204 [0.6074, 0.6332] | 0.6166 [0.6038, 0.6297] | 0.6805 [0.6661, 0.6947] | 0.2546 [0.2470, 0.2627] | 0.1623 [0.1497, 0.1751] |
| SemEval Subtask A mono small | Frozen Transformer probe | distilbert_frozen_probe | 5000 | 0.5796 [0.5660, 0.5926] | 0.5383 [0.5250, 0.5517] | 0.6416 [0.6254, 0.6560] | 0.3238 [0.3142, 0.3345] | 0.2759 [0.2632, 0.2898] |
| SemEval Subtask A mono small | Fine-tuned Transformer | distilbert_finetuned | 5000 | 0.6380 [0.6246, 0.6514] | 0.6184 [0.6040, 0.6327] | 0.7022 [0.6881, 0.7160] | 0.2870 [0.2767, 0.2973] | 0.2280 [0.2163, 0.2429] |

## Historical Interpretation Boundary

- These intervals are conditional on early sampled test sets.
- They do not define the current submission-scope claims.
- Current manuscript tables are generated from locked prediction files and the empirical assets listed in `docs/empirical_experiment_summary.md`.
