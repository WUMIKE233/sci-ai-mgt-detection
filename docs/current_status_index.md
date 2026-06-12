# Current Project Status Index

This file is the preferred entry point for the current state of the SCI manuscript project. Older reports remain useful as stage logs, but some older limitation statements have been superseded by later analysis.

## Latest Evidence Files

- `docs/preliminary_experiment_summary.md`: compact table of all current preliminary model metrics.
- `docs/bootstrap_confidence_intervals.md`: 95% bootstrap confidence intervals from current prediction files.
- `docs/error_analysis.md`: current subgroup and high-confidence error analysis.
- `docs/submission_readiness_report.md`: conservative submission readiness audit and blocker list.
- `docs/transformer_finetune_preliminary_report.md`: small-sample fine-tuned DistilBERT baseline.
- `docs/raid_preliminary_report.md`: RAID labeled-subset data boundary and preliminary robustness observations.
- `docs/semeval_preliminary_report.md`: official SemEval Subtask A monolingual small-sample baseline and calibration results.
- `docs/semeval_seed_sweep_report.md`: repeated-seed stability report for low-cost SemEval TF-IDF and calibration runs.
- `docs/neural_seed_sweep_report.md`: repeated-seed stability report for SemEval frozen and fine-tuned DistilBERT neural baselines.
- `docs/semeval_subtaskA_data_note.md`: SemEval-2024 Task 8 Subtask A schema, download, and conversion notes.
- `docs/environment_snapshot.md`: local Python/CUDA/package reproducibility snapshot.
- `docs/submission_package_checklist.md`: manuscript, data, reproducibility, and author-metadata submission checklist.
- `docs/submission_docx_qa.md`: structural QA record for the generated DOCX submission package.
- `https://github.com/WUMIKE233/sci-ai-mgt-detection`: public code repository excluding raw data, processed benchmark extracts, generated outputs, and local submission-private materials.
- `outputs/reproducibility/sci_ai_mgt_detection_code_package.zip`: local code archive excluding raw datasets.

## Manuscript Assets

- `manuscript/tables/table1_preliminary_metrics_with_ci.md`
- `manuscript/tables/table2_worst_subgroups.md`
- `manuscript/figures/figure1_accuracy_with_ci.png`
- `manuscript/figures/figure2_ece_with_ci.png`
- `manuscript/figures/figure3_worst_subgroups.png`
- `manuscript/figures/figure_captions.md`
- `submission_package/manuscript_draft_eswa.docx`
- `submission_package/cover_letter_draft.docx`
- `submission_package/declarations_draft.docx`
- `submission_package/author_metadata_template.docx`

## Current Completion Boundary

The project now has real-data preliminary experiments, Transformer baselines, bootstrap intervals, error analysis, official SemEval Subtask A monolingual small-sample TF-IDF and DistilBERT runs, a three-seed SemEval stability sweep for low-cost models, a three-seed SemEval neural baseline sweep, generated DOCX submission drafts, a public GitHub code repository, a local reproducibility code archive, and a repeatable submission-readiness audit. The latest audit status is still BLOCKED because final Web of Science JCR/CAS target-journal verification remains required.
