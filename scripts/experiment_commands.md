# Experiment Commands

These commands are placeholders for the next implementation stage. They must be updated after dataset locations and local Python environment are confirmed.

## 1. Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Download datasets

```powershell
python scripts\probe_hf_dataset.py yaful/MAGE liamdugan/raid --output data\dataset_probe.json
W:\PY311\python.exe scripts\download_mage_full.py --output-dir data\raw\mage

# SemEval-2024 Task 8 Subtask A official download, after installing gdown:
# gdown --folder https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc

# Small MAGE OOD files already used for pipeline validation:
# data\raw\mage\test_ood_set_gpt.csv
# data\raw\mage\test_ood_set_gpt_para.csv
# Full MAGE raw files now downloaded:
# data\raw\mage\train.csv
# data\raw\mage\valid.csv
# data\raw\mage\test.csv
```

## 3. Prepare a small real MAGE subset

```powershell
python scripts\prepare_mage_subset.py
```

Expected local outputs:

- `data/processed/mage/mage_train_small.csv`
- `data/processed/mage/mage_test_ood_para_small.csv`
- `data/processed/mage/metadata.json`

This is a real-data pipeline validation subset, not the final manuscript experiment.

## 4. Smoke-test the baseline script

```powershell
python scripts\train_baselines.py --config configs\baseline.example.json
```

Expected local outputs:

- `outputs/baseline_smoke/metrics.json`
- `outputs/baseline_smoke/predictions.csv`
- `outputs/baseline_smoke/group_metrics.csv`

These smoke-test outputs are only for validating the script. They must not be used as manuscript evidence.

## 5. Run the small MAGE baseline

```powershell
python scripts\train_baselines.py --config configs\baseline.mage_small.json
```

Expected local outputs:

- `outputs/mage_small_baseline/metrics.json`
- `outputs/mage_small_baseline/predictions.csv`
- `outputs/mage_small_baseline/group_metrics.csv`

These results are useful for debugging the pipeline and shaping the paper, but final submission still requires full-scale experiments and confidence intervals.

## 6. Run final real baselines

```powershell
# FINAL_CONFIG_NEEDED: create configs\baseline.real.json after full dataset conversion.
# python scripts\train_baselines.py --config configs\baseline.real.json
```

## 7. Prepare and run a small labeled RAID subset

```powershell
python scripts\prepare_raid_subset.py
python scripts\train_baselines.py --config configs\baseline.raid_small.json
python scripts\run_calibrated_selective.py --config configs\calibrated_selective.raid_small.json
```

Expected local outputs:

- `data/processed/raid/raid_train_small.csv`
- `data/processed/raid/raid_test_small.csv`
- `data/processed/raid/metadata.json`
- `outputs/raid_small_baseline/metrics.json`
- `outputs/raid_small_calibrated_selective/metrics.json`

RAID's public `raid_test/test.csv` is unlabeled. The small supervised subset samples labeled rows from `raid/train` and `raid/extra`.

## 8. Run calibration

```powershell
python scripts\run_calibrated_selective.py --config configs\calibrated_selective.mage_small.json
```

Expected local outputs:

- `outputs/mage_small_calibrated_selective/metrics.json`
- `outputs/mage_small_calibrated_selective/predictions.csv`
- `outputs/mage_small_calibrated_selective/reliability_diagram.png`
- `outputs/mage_small_calibrated_selective/coverage_risk.png`
- `outputs/mage_small_calibrated_selective/calibration_report.md`

This run uses a small sampled MAGE subset. It is still a pre-submission pipeline validation, not the final manuscript experiment.

## 9. Run robustness evaluation

```powershell
# ROBUSTNESS_SCRIPT_NEEDED: implement scripts\evaluate_robustness.py
# .\.venv\Scripts\python.exe scripts\evaluate_robustness.py --dataset raid --split test
```

## 10. Prepare SemEval-2024 Task 8 Subtask A

```powershell
python scripts\prepare_semeval_subtaskA.py --track monolingual --max-train-per-group 500 --max-dev-per-group 500
python scripts\train_baselines.py --config configs\baseline.semeval_subtaskA_mono_small.json
python scripts\run_calibrated_selective.py --config configs\calibrated_selective.semeval_subtaskA_mono_small.json
W:\PY311\python.exe scripts\train_transformer_probe.py --config configs\transformer_probe.semeval_subtaskA_mono_small.json
W:\PY311\python.exe scripts\train_transformer_finetune.py --config configs\transformer_finetune.semeval_subtaskA_mono_small.json
```

Expected local outputs after official data are available:

- `data/processed/semeval2024_task8/semeval_subtaskA_monolingual_train.csv`
- `data/processed/semeval2024_task8/semeval_subtaskA_monolingual_dev.csv`
- `outputs/semeval_subtaskA_mono_small_baseline/metrics.json`
- `outputs/semeval_subtaskA_mono_small_calibrated_selective/metrics.json`
- `outputs/semeval_subtaskA_mono_small_transformer_probe/metrics.json`
- `outputs/semeval_subtaskA_mono_small_transformer_finetune/metrics.json`

This step is the bridge from MAGE/RAID pre-experiments to the planned SemEval/M4 benchmark evidence.

## 11. Run frozen Transformer probe baselines

Use the Python 3.11 environment with CUDA-enabled PyTorch when available:

```powershell
W:\PY311\python.exe scripts\train_transformer_probe.py --config configs\transformer_probe.raid_small.json
W:\PY311\python.exe scripts\train_transformer_probe.py --config configs\transformer_probe.mage_small.json
```

Expected local outputs:

- `outputs/raid_small_transformer_probe/metrics.json`
- `outputs/mage_small_transformer_probe/metrics.json`

This is a frozen encoder baseline, not full Transformer fine-tuning.

## 12. Run fine-tuned Transformer baselines

Use the same Python 3.11 environment:

```powershell
W:\PY311\python.exe scripts\train_transformer_finetune.py --config configs\transformer_finetune.raid_small.json
W:\PY311\python.exe scripts\train_transformer_finetune.py --config configs\transformer_finetune.mage_small.json
```

Expected local outputs:

- `outputs/raid_small_transformer_finetune/metrics.json`
- `outputs/mage_small_transformer_finetune/metrics.json`

These are small-sample fine-tuning runs; full submission-grade runs still need larger stratified samples and repeated seeds.

## 13. Summarize preliminary metrics

```powershell
python scripts\summarize_preliminary_results.py
```

Expected local outputs:

- `outputs/experiment_summary/preliminary_metrics.csv`
- `docs/preliminary_experiment_summary.md`

## 14. Run SemEval repeated-seed stability sweep

```powershell
python scripts\run_semeval_seed_sweep.py --seeds 13,42,2026
```

Expected local outputs:

- `outputs/semeval_seed_sweep/seed_sweep_metrics.csv`
- `outputs/semeval_seed_sweep/seed_sweep_summary.csv`
- `docs/semeval_seed_sweep_report.md`

This is preliminary stability evidence for low-cost SemEval TF-IDF and calibration runs. It complements, but does not replace, the neural repeated-seed sweep.

## 15. Run SemEval neural repeated-seed stability sweep

```powershell
W:\PY311\python.exe scripts\run_neural_seed_sweep.py --python W:\PY311\python.exe --device auto --seeds 13,42,2026
```

Expected local outputs:

- `outputs/neural_seed_sweep/neural_seed_sweep_metrics.csv`
- `outputs/neural_seed_sweep/neural_seed_sweep_summary.csv`
- `docs/neural_seed_sweep_report.md`

This is preliminary stability evidence for frozen and fine-tuned DistilBERT on the SemEval Subtask A monolingual small-sample setup.

## 16. Bootstrap confidence intervals and error analysis

```powershell
python scripts\bootstrap_confidence_intervals.py
python scripts\analyze_errors.py
```

Expected local outputs:

- `outputs/statistical_analysis/bootstrap_confidence_intervals.csv`
- `docs/bootstrap_confidence_intervals.md`
- `outputs/error_analysis/subgroup_error_summary.csv`
- `outputs/error_analysis/high_confidence_errors.csv`
- `docs/error_analysis.md`

## 17. Audit submission readiness

```powershell
python scripts\audit_submission_readiness.py
```

Expected local outputs:

- `outputs/readiness/submission_readiness.json`
- `outputs/readiness/submission_readiness_checks.csv`
- `outputs/readiness/submission_placeholders.csv`
- `docs/submission_readiness_report.md`

The audit must report `PASS` before the package can be treated as ready for submission.

## 18. Generate manuscript tables and figures

```powershell
python scripts\make_manuscript_assets.py
```

Expected local outputs:

- `manuscript/tables/table1_preliminary_metrics_with_ci.md`
- `manuscript/tables/table2_worst_subgroups.md`
- `manuscript/figures/figure1_accuracy_with_ci.png`
- `manuscript/figures/figure2_ece_with_ci.png`
- `manuscript/figures/figure3_worst_subgroups.png`
- `manuscript/figures/figure_captions.md`
- `docs/current_status_index.md`
