# SCI Q1 AI Manuscript Project / SCI 一区 AI 论文项目

## 中文说明

本项目用于准备一篇具备 SCI 一区投稿潜力的人工智能方向论文。当前选题暂定为：

**面向跨领域与对抗扰动场景的校准式 AI 生成文本检测方法**

论文目标不是凭空生成“完整结果”，而是建立一套可执行、可复现、可投稿前逐步填证据的研究闭环。尚未完成的内容会保留明确的证据缺口说明，避免把计划写成结论。

### 当前目录

- `docs/`: 中文为主的研究设计、期刊定位、文献矩阵、实验方案和证据门控。
- `manuscript/`: 英文投稿稿件草稿、cover letter、highlights 和声明材料。
- `references/`: 初始 BibTeX 与来源记录。
- `scripts/`: 后续实验脚本与命令说明。
- `data/`: 数据下载说明与本地数据占位目录，不直接提交大数据。
- `logs/`: 分阶段操作记录。
- `submission_package/`: 生成的 Word 投稿草稿、作者信息模板和 DOCX manifest。
- `operation_log.md`: 当前项目操作日志。

### 下一步

1. 确认是否采用当前选题，或改为医学影像、遥感生态、金融风控等 AI 应用方向。
2. 下载并核验 SemEval/M4、MAGE、RAID 等公开数据集的许可和数据结构。
3. 跑基线模型、校准模块、跨域测试、对抗扰动测试和消融实验。
4. 将真实结果填入 `manuscript/manuscript_draft.md` 中的证据缺口位置。
5. 按目标期刊格式生成 DOCX/LaTeX 投稿包。

### 当前进展

- 已接入 MAGE 小样本真实数据并跑通 TF-IDF baseline。
- 已实现并跑通 post-hoc calibration 与 selective prediction 预实验。
- 已接入 RAID 有标签小样本，并跑通 attack/domain robustness 预实验。
- 已跑通 frozen DistilBERT embedding probe，补强 Transformer baseline。
- 已跑通 fine-tuned DistilBERT sequence classifier，进一步补强主模型 baseline。
- 已生成 bootstrap 置信区间和错误分析报告。
- 已生成 reliability diagram 与 coverage-risk curve，但目前仍属于小样本预实验。
- 已下载 SemEval-2024 Task 8 Subtask A 官方 monolingual 与 multilingual train/dev 文件，并跑通 monolingual train/dev 小样本 TF-IDF、calibration、frozen DistilBERT 与 fine-tuned DistilBERT 预实验；完整多语言/全量实验仍待完成。
- 已完成 SemEval 低成本 TF-IDF/calibration 三 seed 稳定性复跑。
- 已完成 SemEval neural baseline 三 seed 复跑，覆盖 frozen DistilBERT probe 与 fine-tuned DistilBERT。
- 已下载 MAGE full raw `train.csv`、`valid.csv`、`test.csv`，完整处理与最终实验仍待完成。
- 已生成本地代码归档、环境快照和投稿包 checklist。
- 已生成 Word 投稿草稿：主文档、cover letter、declarations 和 author metadata template，并完成结构性 DOCX QA。
- 已新增投稿 readiness audit，当前状态为 `BLOCKED`，用于跟踪未完成的实验、声明、参考文献、代码归档和期刊核验门禁。

## English Overview

This workspace prepares an AI-related manuscript with potential for submission to an SCI Q1 journal. The provisional topic is:

**Calibrated detection of AI-generated text under cross-domain and adversarial distribution shifts**

The project is evidence-gated: missing experiments are explicitly tracked as evidence gaps rather than written as completed findings. The goal is to build a reproducible path from topic selection to manuscript, experiments, and submission materials.

### Project Structure

- `docs/`: Research design, journal targeting, literature matrix, experimental protocol, and claim-evidence registry.
- `manuscript/`: English manuscript draft, cover letter, highlights, and declaration materials.
- `references/`: Seed BibTeX and source notes.
- `scripts/`: Experiment commands and future executable scripts.
- `data/`: Dataset acquisition notes and local data placeholder.
- `logs/`: Stage-by-stage operation notes.
- `submission_package/`: Generated Word submission drafts, author metadata template, and DOCX manifest.
- `operation_log.md`: Running project log.

### Next Steps

1. Confirm the current topic or redirect to another AI application area.
2. Download and verify licenses for SemEval/M4, MAGE, RAID, and related datasets.
3. Run baseline, calibration, cross-domain, adversarial, and ablation experiments.
4. Replace every evidence gap with verified results.
5. Format the final manuscript for the selected journal.

### Current Progress

- A small real MAGE subset has been normalized and evaluated with TF-IDF baselines.
- Post-hoc calibration and selective prediction have been implemented and smoke-tested.
- A small labeled RAID subset has been normalized and evaluated for attack/domain robustness.
- A frozen DistilBERT embedding probe has been run as a lightweight Transformer baseline.
- A fine-tuned DistilBERT sequence classifier has been run as a stronger model baseline.
- Bootstrap confidence intervals and error-analysis reports have been generated.
- Reliability and coverage-risk figures are generated, but they remain preliminary.
- Official SemEval-2024 Task 8 Subtask A monolingual and multilingual train/dev files have been downloaded; monolingual train/dev small-sample TF-IDF, calibration, frozen DistilBERT, and fine-tuned DistilBERT runs have been completed, while full multilingual/full-scale experiments remain pending.
- A three-seed SemEval low-cost TF-IDF/calibration stability sweep has been completed.
- A three-seed SemEval neural baseline sweep has been completed for frozen DistilBERT probe and fine-tuned DistilBERT.
- Full raw MAGE `train.csv`, `valid.csv`, and `test.csv` have been downloaded; full processing and final experiments remain pending.
- A local code archive, environment snapshot, and submission package checklist have been generated.
- Word submission drafts have been generated for the manuscript, cover letter, declarations, and author metadata template, with structural DOCX QA completed.
- A submission readiness audit has been added; the current status is `BLOCKED` until remaining experiments, declarations, references, code archiving, and journal verification are complete.
