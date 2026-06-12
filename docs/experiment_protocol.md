# 实验方案

## 研究问题

RQ1: 在跨领域、跨生成器和跨数据集设置下，常见 MGT 检测器的准确率与校准可靠性是否一致？

RQ2: 温度校准、保序校准或 conformal selective prediction 是否能降低高置信误判，并在保持覆盖率的同时改善 selective risk？

RQ3: 改写式分布偏移、未知生成模型、跨语言和跨来源设置分别如何影响检测性能与置信度？

RQ4: 哪些文本类型、生成模型或扰动最容易触发失败模式？

## 数据集计划

| 数据集 | 用途 | 当前状态 | 备注 |
|---|---|---|---|
| SemEval-2024 Task 8 / M4 | 训练、验证、任务基线 | Subtask A monolingual/multilingual 官方 train/dev 已下载；提交版锁定 TF-IDF Logistic Regression、calibration/selective prediction、bootstrap CI 和 subgroup analysis 已完成 | Subtask A 官方 schema 已核验：JSONL, label 0=human, 1=machine；见 `docs/semeval_subtaskA_data_note.md` 和 `docs/empirical_experiment_summary.md` |
| MAGE | OOD 测试、跨场景评估 | full raw train/valid/test、GPT-4 OOD、paraphrase OOD 已下载；提交版锁定 TF-IDF Logistic Regression、calibration/selective prediction、bootstrap CI 和 subgroup analysis 已完成 | Hugging Face license 为 Apache-2.0；已确认 label=1 human, label=0 machine |
| RAID | 背景文献和数据源记录 | 已核验数据源和许可；不作为当前稿件主实验证据 | Hugging Face license 为 MIT；公开 test 无标签，监督实验需从 train/extra 抽样 |

## 模型与方法

### Baselines

1. TF-IDF + Logistic Regression / Linear SVM
2. Statistical feature model: length, entropy, burstiness, compression ratio, punctuation profile
3. Historical development baseline: frozen and fine-tuned DistilBERT small-sample runs.
4. Stronger neural and commercial detectors are outside the submitted empirical scope and are listed as future comparison targets.

### Proposed framework

暂名：**Calibrated Selective MGT Detector (CS-MGT)**

模块：

1. Base detector: outputs class probability for human vs MGT.
2. Calibration layer: temperature scaling, isotonic calibration, or vector scaling on validation set.
3. Selective decision layer: abstains when confidence or conformal p-value fails threshold.
4. Robustness evaluator: measures performance under OOD generator/domain, cross-lingual transfer, and paraphrase-style shifts.

### Current implementation status

- Implemented: `scripts/train_baselines.py`
- Implemented: `scripts/run_calibrated_selective.py`
- Implemented: `scripts/train_transformer_probe.py`
- Implemented: `scripts/train_transformer_finetune.py`
- Implemented calibration methods: identity baseline, temperature scaling, isotonic calibration.
- Implemented selective outputs: confidence-based coverage-risk curves and split-conformal summary.
- Implemented Transformer baseline: frozen `distilbert-base-uncased` encoder embeddings plus Logistic Regression.
- Implemented fine-tuned Transformer baseline: `distilbert-base-uncased` sequence classifier.
- Current submitted scope: locked full SemEval-A monolingual/multilingual train-dev experiments and locked full MAGE train/test/OOD experiments with TF-IDF Logistic Regression, calibration/selective prediction, bootstrap intervals, and subgroup analysis. RAID is recorded as background and data-source context, not as main empirical evidence.

## Evaluation metrics

- Accuracy
- Macro-F1
- AUROC
- AUPRC
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE), optional
- Brier score
- Selective risk at fixed coverage
- Coverage at fixed risk
- Reliability diagram
- Error rate by domain, generator, decoding strategy, and perturbation

## Experimental design

### Experiment 1: In-distribution baseline

Train and validate on official training/validation split, report baseline performance.

Required output:

- Table 1: Baseline performance
- Figure 1: Reliability diagram
- Locked submission numbers are reported in `docs/empirical_experiment_summary.md` and generated from `outputs/locked/`.

### Experiment 2: Cross-generator generalization

Train on known generators, test on held-out generator(s).

Required output:

- Table 2: Seen vs unseen generator performance
- Figure 2: Confidence distribution for correct vs incorrect predictions
- Locked cross-generator and cross-source subgroup results are reported in `docs/empirical_experiment_summary.md`.

### Experiment 3: Cross-domain generalization

Train on selected domains, test on held-out domains.

Required output:

- Table 3: Domain transfer results
- Locked cross-domain and cross-source subgroup results are reported in `docs/empirical_experiment_summary.md`.

### Experiment 4: Paraphrase-style OOD robustness

Evaluate MAGE GPT-4 OOD and paraphrase OOD subsets under the locked protocol.

Required output:

- Table 4: Robustness under OOD and paraphrase-style shifts
- Figure 3: Coverage-risk curves
- Locked OOD and paraphrase-style results are reported in `docs/empirical_experiment_summary.md`.

### Experiment 5: Ablation study

Compare:

- Base detector only
- Base + calibration
- Base + selective prediction
- Base + calibration + selective prediction
- Optional: feature fusion variants

Required output:

- Table 5: Ablation results
- Calibration and selective prediction outputs are reported in `docs/empirical_experiment_summary.md`.

## Reproducibility

- Fixed random seeds: 13, 42, 2026
- Report hardware: captured in `docs/environment_snapshot.md`.
- Report package versions: captured in `docs/environment_snapshot.md` and `outputs/reproducibility/environment_snapshot.json`.
- Save train/validation/test split manifests
- Save model checkpoints or provide deterministic training scripts
- Avoid redistributing datasets if licenses prohibit it; provide download scripts instead

## Statistical analysis

- Use bootstrap confidence intervals for AUROC/F1 where feasible.
- Use paired tests for method comparison only when predictions are aligned on identical test examples.
- Correct for multiple comparisons if many perturbation types are tested.
- Implemented locked empirical bootstrap confidence intervals in `outputs/locked/statistical_analysis/empirical_metrics_with_ci.csv`.
- Implemented locked subgroup/high-confidence error analysis in `outputs/locked/statistical_analysis/empirical_subgroup_errors.csv`.

## 投稿前不可跳过

- 所有结果表格必须来自脚本输出。
- 所有 figures 必须有生成脚本或 notebook。
- 数据许可和可用性声明必须真实可核验。

## 当前提交版证据状态

提交版稿件使用 `docs/empirical_experiment_summary.md`、`outputs/locked/statistical_analysis/empirical_metrics_with_ci.csv`、`outputs/locked/statistical_analysis/empirical_selective_metrics.csv` 和 `outputs/locked/statistical_analysis/empirical_subgroup_errors.csv` 作为主证据来源。早期 smoke tests、small-sample transformer probes 和 RAID offset checks 仅作为开发记录保留，不作为当前稿件的主实验依据。

当前主稿件的实证边界如下：

- SemEval-2024 Task 8 Subtask A monolingual full train-dev: 119,757 train / 5,000 dev。
- SemEval-2024 Task 8 Subtask A multilingual full train-dev: 172,417 train / 4,000 dev。
- MAGE full train/test/OOD: 319,071 train、56,819 in-distribution test、1,562 GPT-4 OOD、2,362 paraphrase OOD。
- Main model: TF-IDF Logistic Regression with calibration/selective prediction evaluation。
- Statistical outputs: 500-resample bootstrap confidence intervals and subgroup error analysis。
- Out-of-scope for submitted empirical claims: full RAID evaluation, stronger neural baselines on full benchmark splits, and commercial detector comparisons。
