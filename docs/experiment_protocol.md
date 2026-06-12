# 实验方案

## 研究问题

RQ1: 在跨领域、跨生成器和跨数据集设置下，常见 MGT 检测器的准确率与校准可靠性是否一致？

RQ2: 温度校准、保序校准或 conformal selective prediction 是否能降低高置信误判，并在保持覆盖率的同时改善 selective risk？

RQ3: 对抗扰动、改写、随机空格、解码策略和未知生成模型分别如何影响检测性能与置信度？

RQ4: 哪些文本类型、生成模型或扰动最容易触发失败模式？

## 数据集计划

| 数据集 | 用途 | 当前状态 | 备注 |
|---|---|---|---|
| SemEval-2024 Task 8 / M4 | 训练、验证、任务基线 | Subtask A monolingual/multilingual 官方 train/dev 已下载；monolingual 小样本 TF-IDF/calibration/DistilBERT 已跑通；完整多语言实验仍待完成 | Subtask A 官方 schema 已核验：JSONL, label 0=human, 1=machine；见 `docs/semeval_subtaskA_data_note.md` 和 `docs/semeval_preliminary_report.md` |
| MAGE | OOD 测试、跨场景评估 | 小样本已跑通；full raw train/valid/test 已下载；完整处理仍待完成 | Hugging Face license 为 Apache-2.0；已确认 label=1 human, label=0 machine |
| RAID | 鲁棒测试、adversarial perturbation、decoding variation | 有标签小样本已跑通；全量仍待完成 | Hugging Face license 为 MIT；公开 test 无标签，监督实验需从 train/extra 抽样 |

## 模型与方法

### Baselines

1. TF-IDF + Logistic Regression / Linear SVM
2. Statistical feature model: length, entropy, burstiness, compression ratio, punctuation profile
3. Pre-trained encoder classifier: RoBERTa / DeBERTa / ModernBERT variant, depending on local GPU and model availability
4. Existing detector baseline where license permits: pending license-compatible implementation

### Proposed framework

暂名：**Calibrated Selective MGT Detector (CS-MGT)**

模块：

1. Base detector: outputs class probability for human vs MGT.
2. Calibration layer: temperature scaling, isotonic calibration, or vector scaling on validation set.
3. Selective decision layer: abstains when confidence or conformal p-value fails threshold.
4. Robustness evaluator: measures performance under OOD generator/domain and adversarial perturbations.

### Current implementation status

- Implemented: `scripts/train_baselines.py`
- Implemented: `scripts/run_calibrated_selective.py`
- Implemented: `scripts/train_transformer_probe.py`
- Implemented: `scripts/train_transformer_finetune.py`
- Implemented calibration methods: identity baseline, temperature scaling, isotonic calibration.
- Implemented selective outputs: confidence-based coverage-risk curves and split-conformal summary.
- Implemented Transformer baseline: frozen `distilbert-base-uncased` encoder embeddings plus Logistic Regression.
- Implemented fine-tuned Transformer baseline: `distilbert-base-uncased` sequence classifier.
- Current limitation: MAGE/RAID remain small-sample validations; SemEval now has low-cost and neural three-seed small-sample sweeps.

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
- Final locked numbers from the complete protocol remain pending.

### Experiment 2: Cross-generator generalization

Train on known generators, test on held-out generator(s).

Required output:

- Table 2: Seen vs unseen generator performance
- Figure 2: Confidence distribution for correct vs incorrect predictions
- Final locked numbers from the complete protocol remain pending.

### Experiment 3: Cross-domain generalization

Train on selected domains, test on held-out domains.

Required output:

- Table 3: Domain transfer results
- Final locked numbers from the complete protocol remain pending.

### Experiment 4: Adversarial and decoding robustness

Evaluate paraphrase, spacing, adversarial perturbation, and decoding variation subsets where available.

Required output:

- Table 4: Robustness under perturbations
- Figure 3: Coverage-risk curves
- Final locked numbers from the complete protocol remain pending.

### Experiment 5: Ablation study

Compare:

- Base detector only
- Base + calibration
- Base + selective prediction
- Base + calibration + selective prediction
- Optional: feature fusion variants

Required output:

- Table 5: Ablation results
- Final locked numbers from the complete protocol remain pending.

## Reproducibility

- Fixed random seeds: 13, 42, 2026
- Report hardware: pending final environment capture
- Report package versions: pending final environment capture
- Save train/validation/test split manifests
- Save model checkpoints or provide deterministic training scripts
- Avoid redistributing datasets if licenses prohibit it; provide download scripts instead

## Statistical analysis

- Use bootstrap confidence intervals for AUROC/F1 where feasible.
- Use paired tests for method comparison only when predictions are aligned on identical test examples.
- Correct for multiple comparisons if many perturbation types are tested.
- Implemented preliminary bootstrap confidence intervals: `scripts/bootstrap_confidence_intervals.py`.
- Implemented preliminary subgroup/high-confidence error analysis: `scripts/analyze_errors.py`.

## 投稿前不可跳过

- 所有结果表格必须来自脚本输出。
- 所有 figures 必须有生成脚本或 notebook。
- 数据许可和可用性声明必须真实可核验。

## 当前预实验状态

已完成一次 MAGE 小样本真实数据 smoke test，见 `docs/preliminary_mage_smoke_report.md`。该结果只能证明数据转换、baseline 训练和指标输出管线可运行，不能作为投稿结果直接使用。

已完成一次 MAGE 小样本校准与选择性预测 smoke test，见 `docs/calibrated_selective_preliminary_report.md`。该结果显示方法管线可以产生 ECE、Brier、coverage-risk、reliability diagram 和 conformal summary，但仍不是最终投稿证据。

已完成一次 RAID 有标签 offset 小样本 smoke test，见 `docs/raid_preliminary_report.md`。该结果增加了 attack/domain robustness 证据链，但仍不是最终投稿证据。

已完成一次 frozen Transformer probe baseline，见 `docs/transformer_probe_preliminary_report.md`。该结果提高了 baseline 强度，但仍不是完整 fine-tuning 或投稿级主实验。

已完成一次 small-sample DistilBERT fine-tuning baseline，见 `docs/transformer_finetune_preliminary_report.md`。该结果进一步补强主模型证据，但仍需更大样本、重复种子和置信区间。

当前所有预实验主指标已汇总到 `docs/preliminary_experiment_summary.md` 和 `outputs/experiment_summary/preliminary_metrics.csv`，便于后续生成论文表格。

当前所有预实验预测已生成 bootstrap 置信区间，见 `docs/bootstrap_confidence_intervals.md`。当前 Transformer baseline 的 subgroup 和 high-confidence error 分析见 `docs/error_analysis.md`。

当前预实验论文资产已由 `scripts/make_manuscript_assets.py` 生成，见 `manuscript/tables/`、`manuscript/figures/` 和 `docs/current_status_index.md`。这些资产可用于草稿组织和投稿前路线图，但最终投稿表图仍需在完整数据、重复种子和锁定评估协议后重新生成。

SemEval-2024 Task 8 Subtask A 转换入口已实现为 `scripts/prepare_semeval_subtaskA.py`，并新增 `configs/baseline.semeval_subtaskA_mono_small.json`、`configs/calibrated_selective.semeval_subtaskA_mono_small.json`、`configs/transformer_probe.semeval_subtaskA_mono_small.json` 与 `configs/transformer_finetune.semeval_subtaskA_mono_small.json`。当前已完成官方 monolingual train/dev 小样本转换、baseline、calibration/selective、frozen DistilBERT、fine-tuned DistilBERT、bootstrap CI、低成本三 seed 复跑和 neural baseline 三 seed 复跑；完整多语言处理和官方 test/gold 处理仍需后续完成。

SemEval 低成本 TF-IDF/calibration 已完成 seeds 13、42、2026 的稳定性复跑，见 `docs/semeval_seed_sweep_report.md`。SemEval neural baseline 已完成 seeds 13、42、2026 的 frozen DistilBERT probe 与 fine-tuned DistilBERT 复跑，见 `docs/neural_seed_sweep_report.md`。
