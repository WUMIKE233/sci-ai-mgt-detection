# 初始文献矩阵

## 检索状态

- 本轮已联网核对 ACL Anthology、PMLR、OpenReview、arXiv、IEEE DOI、Springer、Now Publishers 与公开项目页面。
- `references/seed_references.bib` 已扩展到 20 条，覆盖核心 benchmark、MGT 检测方法、校准、选择性预测、conformal prediction、水印与检测可靠性。
- 后续仍需在最终投稿前补目标期刊近两年同方向文章和引用计数。

## 核心数据与 benchmark

| 主题 | 代表来源 | 可用于本文的位置 | 当前要点 | 状态 |
|---|---|---|---|---|
| SemEval-2024 Task 8 / M4 | Wang et al., SemEval 2024; M4 EACL 2024 | 数据集、任务定义、基线比较 | 任务覆盖 human vs machine、模型归因、mixed text boundary；数据来自 M4 扩展 | 已核到公开任务页，需下载数据 |
| MAGE | Li et al., ACL 2024 | OOD 泛化、跨场景评估 | 论文强调现有检测方法在 wide-ranging testbed 尤其 OOD 场景存在困难 | 已核到 ACL 页面和 DOI |
| RAID | Dugan et al., arXiv/ACL 2024; GitHub | 鲁棒评估、对抗扰动、解码策略变化 | RAID 覆盖多模型、多领域、多解码和对抗攻击；仓库说明与论文版本统计存在差异，需以论文/数据版本固定 | 已核到 arXiv/GitHub |
| MGTBench | He et al., arXiv 2023/2024 | benchmark 框架与攻击类型 | 强调不同架构、数据集和设置导致评估不可比，并包含多类检测器和数据设置 | 已补 BibTeX |
| TrustAI at SemEval-2024 Task 8 | Urlana et al., SemEval 2024 | baseline 与 error analysis 参照 | 报告 statistical、neural、pre-trained 方法组合，并给出 subtask-A mono accuracy 86.9% 等结果 | 已核到 ACL 页面 |

## 拟写 Related Work 结构

### 1. Benchmarking machine-generated text detection

要点：早期检测研究常在单域或有限生成器上验证；M4/SemEval、MAGE、RAID、MGTBench 将问题推进到多生成器、多领域、多语言和对抗场景。

### 2. Detector architectures and feature families

要点：统计特征、困惑度/熵/压缩特征、传统机器学习、预训练 Transformer、黑盒检测器、白盒/概率访问检测器应分组讨论。本文更适合定位为“可靠性与校准层”而不是替代所有检测器。

### 3. Robustness under distribution shift

要点：核心缺口是 seen-domain/seen-generator 上的高分不能说明实际部署可靠性。需要跨数据集、跨模型、跨领域、跨解码、对抗扰动测试。

### 4. Calibration and selective prediction

要点：分类准确率不能说明模型置信度是否可信；本文用 ECE、selective risk、coverage-risk curve 和可选 conformal prediction 评估“什么时候不应给出确定判断”。

## 已补关键文献组

- Calibration / selective / conformal: Guo et al. 2017; Chow 1970; Geifman and El-Yaniv 2017; Vovk et al. 2005; Angelopoulos and Bates 2023.
- MGT detectors and benchmarks: GLTR; DetectGPT; Fast-DetectGPT; MGTBench; HC3; Ghostbuster; MAGE; RAID; SemEval-2024 Task 8.
- Robustness and evasion: Sadasivan et al. 2023; Krishna et al. 2023; Kirchenbauer et al. 2023.

## 投稿前仍需补充

- 目标期刊近两年同方向论文，尤其 ESWA/KBS/IPM 中 AI-generated text detection、misinformation、trustworthy AI 相关实证文章。
- 参考文献的最终 DOI、卷期页码和引用顺序需要在目标期刊格式确定后统一核对。
