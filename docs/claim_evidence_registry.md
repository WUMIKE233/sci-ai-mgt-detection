# Claim-Evidence Registry

本文件用于防止把尚未完成的实验写成论文结论。每条主张必须绑定证据来源。

| Claim ID | Manuscript claim | Required evidence | Status | Notes |
|---|---|---|---|---|
| C1 | MGT detector reliability degrades under cross-language, cross-generator, cross-domain, and paraphrase-style shifts. | Locked SemEval-A full train-dev results, MAGE in-distribution/OOD results, subgroup analysis, and benchmark literature. | `supported for submitted scope` | Supported for TF-IDF Logistic Regression on SemEval-A and MAGE. Full RAID and commercial-detector claims are outside scope. |
| C2 | Calibration is not uniformly beneficial under distribution shift. | ECE, Brier score, accuracy, and selective-risk comparisons for uncalibrated, temperature-scaled, and isotonic LR. | `supported` | The final wording avoids claiming universal improvement. |
| C3 | Selective prediction is useful as a diagnostic risk-management layer. | Coverage-risk curves, risk at 80% coverage, conformal coverage, and singleton rates. | `supported with boundary` | Selective prediction does not solve all conditions; SemEval multilingual remains high-risk. |
| C4 | Paraphrase-style and OOD generator settings expose failure modes hidden by aggregate metrics. | MAGE GPT-4 OOD, MAGE paraphrase OOD, and subgroup error analysis. | `supported for paraphrase/OOD scope` | The submitted manuscript does not claim full RAID robustness. |
| C5 | The framework is useful for practical content-risk workflows. | Quantitative reliability metrics plus discussion framing detectors as calibrated risk signals. | `supported cautiously` | The manuscript avoids legal/authorship-proof claims. |
| C6 | The method is reproducible and reusable. | Public code repository, dataset-access instructions, environment snapshot, reproducibility archive, and generated tables/figures. | `supported with redistribution limits` | Raw third-party datasets, processed extracts, saved predictions, checkpoints, and large outputs are not redistributed. |

## Evidence Notes

- `2026-06-12`: The submitted manuscript uses locked empirical outputs in `outputs/locked/` and the summary in `docs/empirical_experiment_summary.md`.
- `2026-06-12`: The locked SemEval-A monolingual condition uses 119,757 training rows and 5,000 dev rows.
- `2026-06-12`: The locked SemEval-A multilingual condition uses 172,417 training rows and 4,000 dev rows.
- `2026-06-12`: The locked MAGE conditions use 319,071 training rows, 56,819 in-distribution test rows, 1,562 GPT-4 OOD rows, and 2,362 paraphrase OOD rows.
- `2026-06-12`: The main manuscript tables and figures are generated from `manuscript/tables/table1_empirical_metrics_with_ci.md`, `manuscript/tables/table2_empirical_calibration_selective.md`, `manuscript/tables/table3_empirical_worst_subgroups.md`, and the empirical figures under `manuscript/figures/`.
- `2026-06-13`: RAID remains a background source and development-check dataset only. It is not used for the submitted manuscript's main empirical claims.

## Current permissible wording

Use:

- "This study evaluates..."
- "The proposed protocol is designed to test..."
- "The submitted manuscript reports..."
- "The current submitted scope is..."

Do not use for the submitted manuscript unless new evidence is added:

- "Our method outperforms..."
- "The proposed method significantly improves..."
- "We demonstrate..."
- "The detector is robust..."
- "The first..."
