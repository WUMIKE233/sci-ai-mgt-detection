# 选题决策：校准式 AI 生成文本检测

## 暂定题目

中文：面向跨领域与对抗扰动场景的校准式 AI 生成文本检测方法

英文：Calibrated Detection of AI-Generated Text Under Cross-Domain and Adversarial Distribution Shifts

## 一句话论证

In machine-generated text detection, we show that calibration-aware uncertainty estimation and selective prediction can improve the reliability of detectors under unseen domains, generators, decoding strategies, and adversarial perturbations, supported by cross-dataset experiments on SemEval/M4, MAGE, and RAID, with the boundary that the method detects distributional evidence rather than proving authorship.

## 为什么选这个题

1. **AI 相关性强**：LLM 生成内容检测直接属于 NLP、可信 AI、AI 安全和信息治理交叉方向。
2. **可复现实验可行**：SemEval/M4、MAGE、RAID 等公开资源可以支撑跨域、跨模型、对抗扰动和校准实验。
3. **SCI 一区期刊匹配度较高**：Expert Systems with Applications、Knowledge-Based Systems、Information Processing & Management 等期刊均覆盖 AI、机器学习、文本挖掘、智能系统或信息处理。
4. **容易形成可审稿贡献**：不仅做分类准确率，还评估校准、选择性预测、失效边界、跨数据集泛化和对抗鲁棒性。
5. **风险可控**：不需要敏感个人数据；重点是公开 benchmark 与透明实验。

## 拟定贡献

1. 提出一个 calibration-aware 的 AI 生成文本检测框架，将文本分类置信度、温度校准、保序校准或 conformal selective prediction 结合起来。
2. 构建跨数据集评估协议，覆盖 seen/unseen generator、seen/unseen domain、decoding variation 和 adversarial perturbation。
3. 系统比较统计特征、传统机器学习、预训练语言模型检测器和校准式检测器。
4. 用 Expected Calibration Error、selective risk、coverage-risk curve、AUROC、macro-F1 等指标报告“检测是否可靠”，而不仅是“检测是否准确”。
5. 给出失败模式分析，明确哪些生成模型、领域或扰动会让检测器失效。

## 术语表

| Canonical term | First-use definition | 中文说明 | 使用决定 |
|---|---|---|---|
| Large language model (LLM) | large language model (LLM) | 大语言模型 | 首次写全称，之后用 LLM |
| Machine-generated text (MGT) | machine-generated text (MGT) | 机器生成文本 / AI 生成文本 | 英文稿统一用 MGT |
| AI-generated text | AI-generated text | AI 生成文本 | 标题和摘要可用，正文与 MGT 对齐 |
| Out-of-distribution (OOD) | out-of-distribution (OOD) | 分布外 | 首次写全称 |
| Expected Calibration Error (ECE) | Expected Calibration Error (ECE) | 期望校准误差 | 用于校准评估 |
| Selective prediction | selective prediction | 选择性预测 / 拒判机制 | 表示低置信样本 abstain |
| Conformal prediction | conformal prediction | 保形预测 | 如果采用，需要单独定义置信集 |
| Adversarial perturbation | adversarial perturbation | 对抗扰动 | 包括 paraphrase、spacing、style transfer 等 |

## 缺口与边界

- 需要真实下载数据并运行实验后，才能写入任何提升比例、显著性、排名或统计结论。
- 需要核验每个数据集的许可、可再分发限制和训练/测试划分。
- 需要确认目标期刊采用 JCR Q1 还是中科院一区口径，并以投稿当年的官方数据为准。
- 边界：该研究只能给出文本来源的概率性判别，不应被表述为作者身份的法律证明。
