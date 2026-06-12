# Current Project Status Index

This file is the preferred entry point for the current state of the SCI manuscript project. Older development-stage reports remain useful as stage logs, but the manuscript now uses the locked empirical experiment outputs listed below.

## Latest Evidence Files

- `docs/empirical_experiment_summary.md`: locked empirical metrics, calibration/selective prediction results, and subgroup error summary.
- `outputs/locked/statistical_analysis/empirical_metrics_with_ci.csv`: 500-resample bootstrap confidence intervals from locked prediction files.
- `outputs/locked/statistical_analysis/empirical_selective_metrics.csv`: calibration, selective-risk, and conformal summary metrics.
- `outputs/locked/statistical_analysis/empirical_subgroup_errors.csv`: source/generator/domain subgroup error analysis.
- `docs/submission_readiness_report.md`: conservative submission readiness audit and warning list.
- `docs/semeval_subtaskA_data_note.md`: SemEval-2024 Task 8 Subtask A schema, download, and conversion notes.
- `docs/environment_snapshot.md`: local Python/CUDA/package reproducibility snapshot.
- `docs/submission_package_checklist.md`: manuscript, data, reproducibility, and author-metadata submission checklist.
- `docs/submission_docx_qa.md`: structural QA record for the generated DOCX submission package.
- `https://github.com/WUMIKE233/sci-ai-mgt-detection`: public code repository excluding raw data, processed benchmark extracts, generated outputs, and local submission-private materials.
- `outputs/reproducibility/sci_ai_mgt_detection_code_package.zip`: local code archive excluding raw datasets.

## Manuscript Assets

- `manuscript/tables/table1_empirical_metrics_with_ci.md`
- `manuscript/tables/table2_empirical_calibration_selective.md`
- `manuscript/tables/table3_empirical_worst_subgroups.md`
- `manuscript/figures/figure1_empirical_accuracy_with_ci.png`
- `manuscript/figures/figure2_empirical_ece.png`
- `manuscript/figures/figure3_empirical_selective_risk.png`
- `manuscript/figures/empirical_figure_captions.md`
- `submission_package/manuscript_draft_eswa.docx`
- `submission_package/cover_letter_draft.docx`
- `submission_package/declarations_draft.docx`
- `submission_package/author_metadata_template.docx`

## Current Completion Boundary

The project now has locked empirical experiments on the full SemEval-2024 Task 8 Subtask A monolingual and multilingual train-dev files, plus full MAGE train/test/OOD files available locally. It also has calibration/selective prediction results, bootstrap intervals, subgroup error analysis, generated DOCX submission drafts, a public GitHub code repository, a local reproducibility code archive, captured third-party journal partition evidence, and a repeatable submission-readiness audit. The latest audit status is WARN because no blockers remain, but an official Web of Science JCR/CAS export is still recommended if the institution requires formal proof.
