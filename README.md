# SCI Q1 AI Manuscript Project / SCI 一区 AI 论文项目

## 中文说明

本项目用于准备一篇具备 SCI 一区投稿潜力的人工智能方向论文。当前选题暂定为：

**面向跨领域与对抗扰动场景的校准式 AI 生成文本检测方法**

论文目标不是凭空生成“完整结果”，而是建立一套可执行、可复现、可投稿前逐步填证据的研究闭环。当前版本已经从小样本预实验推进到锁定版真实数据实验：SemEval-2024 Task 8 Subtask A 使用官方 monolingual/multilingual 全量 train-dev，MAGE 使用本地 full train/test/OOD 文件。

### 当前目录

- `docs/`: 中文为主的研究设计、期刊定位、文献矩阵、实验方案和证据门控。
- `manuscript/`: 英文投稿稿件草稿、cover letter、highlights 和声明材料。
- `references/`: 初始 BibTeX 与来源记录。
- `scripts/`: 后续实验脚本与命令说明。
- `data/`: 数据下载说明与本地数据占位目录，不直接提交大数据。
- `logs/`: 分阶段操作记录。
- `submission_package/`: 生成的 Word 投稿草稿、作者信息模板和 DOCX manifest。
- `operation_log.md`: 当前项目操作日志。

### 复现实验入口

1. 准备 SemEval 全量规范化 CSV：`python scripts/prepare_semeval_subtaskA.py --max-train-per-group 0 --max-dev-per-group 0`。
2. 准备 MAGE 锁定版 CSV：`python scripts/prepare_mage_locked.py --max-train-per-src 0`。
3. 训练 TF-IDF Logistic Regression：`python scripts/train_baselines.py`，指定对应 train/test 和 `--models logreg`。
4. 运行校准与选择性预测：`python scripts/run_calibrated_selective.py`。
5. 生成投稿表图：`python scripts/make_empirical_manuscript_assets.py`。

### 当前进展

- 已完成 SemEval-2024 Task 8 Subtask A monolingual 全量 train-dev 实验：119,757 train / 5,000 dev。
- 已完成 SemEval-2024 Task 8 Subtask A multilingual 全量 train-dev 实验：172,417 train / 4,000 dev。
- 已完成 MAGE full train/test/OOD 实验：319,071 train、56,819 in-distribution test、1,562 GPT-4 OOD、2,362 paraphrase OOD。
- 已完成 TF-IDF Logistic Regression baseline、post-hoc calibration、selective prediction、conformal summary、bootstrap 置信区间和 subgroup error analysis。
- 已生成 locked empirical 表图：`manuscript/tables/table1_empirical_metrics_with_ci.md`、`table2_empirical_calibration_selective.md`、`table3_empirical_worst_subgroups.md` 以及三张 empirical figures。
- 已更新英文投稿正文、cover letter、declarations 和 Word 投稿草稿，去除未完成实验痕迹。
- 已生成本地代码归档、环境快照和投稿包 checklist。
- 已创建公开代码仓库：https://github.com/WUMIKE233/sci-ai-mgt-detection
- 已新增投稿 readiness audit，当前状态为 `WARN`；唯一警告是期刊分区仍建议补充官方 Web of Science JCR/CAS 导出。

## English Overview

This workspace prepares an AI-related manuscript with potential for submission to an SCI Q1 journal. The provisional topic is:

**Calibrated detection of AI-generated text under cross-domain and adversarial distribution shifts**

The project is evidence-gated: missing experiments are explicitly tracked as evidence gaps rather than written as completed findings. The current version has moved from small-sample smoke tests to locked empirical experiments: SemEval-2024 Task 8 Subtask A uses the official full monolingual/multilingual train-dev files, and MAGE uses the locally available full train/test/OOD files.

### Project Structure

- `docs/`: Research design, journal targeting, literature matrix, experimental protocol, and claim-evidence registry.
- `manuscript/`: English manuscript draft, cover letter, highlights, and declaration materials.
- `references/`: Seed BibTeX and source notes.
- `scripts/`: Experiment commands and future executable scripts.
- `data/`: Dataset acquisition notes and local data placeholder.
- `logs/`: Stage-by-stage operation notes.
- `submission_package/`: Generated Word submission drafts, author metadata template, and DOCX manifest.
- `operation_log.md`: Running project log.

### Reproduction Entry Points

1. Prepare full SemEval CSV files: `python scripts/prepare_semeval_subtaskA.py --max-train-per-group 0 --max-dev-per-group 0`.
2. Prepare locked MAGE CSV files: `python scripts/prepare_mage_locked.py --max-train-per-src 0`.
3. Train TF-IDF Logistic Regression: run `python scripts/train_baselines.py` with the desired train/test files and `--models logreg`.
4. Run calibration and selective prediction: `python scripts/run_calibrated_selective.py`.
5. Generate manuscript tables and figures: `python scripts/make_empirical_manuscript_assets.py`.

### Current Progress

- Full SemEval-2024 Task 8 Subtask A monolingual train-dev experiments are complete: 119,757 train / 5,000 dev.
- Full SemEval-2024 Task 8 Subtask A multilingual train-dev experiments are complete: 172,417 train / 4,000 dev.
- Full MAGE train/test/OOD experiments are complete: 319,071 train, 56,819 in-distribution test, 1,562 GPT-4 OOD, and 2,362 paraphrase OOD rows.
- TF-IDF Logistic Regression baseline, post-hoc calibration, selective prediction, conformal summaries, bootstrap confidence intervals, and subgroup error analysis have been generated.
- Locked empirical manuscript assets are available as `manuscript/tables/table1_empirical_metrics_with_ci.md`, `table2_empirical_calibration_selective.md`, `table3_empirical_worst_subgroups.md`, and three empirical figures.
- The English manuscript, cover letter, declarations, and Word submission drafts have been updated to remove incomplete-experiment language.
- A local code archive, environment snapshot, and submission package checklist have been generated.
- A public code repository has been created: https://github.com/WUMIKE233/sci-ai-mgt-detection
- A submission readiness audit has been added; the current status is `WARN`, with the remaining warning limited to official Web of Science JCR/CAS partition proof if institutional verification requires it.
