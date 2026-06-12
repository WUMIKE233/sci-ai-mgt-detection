# Data Sources

This file records data sources used or consulted by the project. Do not commit large downloaded datasets unless the license explicitly permits redistribution.

## Dataset records

1. SemEval-2024 Task 8 / M4
   - URL: https://github.com/mbzuai-nlp/SemEval2024-task8
   - Purpose: task definition, training/evaluation data where available.
   - License: Apache-2.0 according to the official GitHub repository.
   - Official download method: `gdown --folder https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc` for Subtask A.
   - Official Subtask A files: `subtaskA_train_monolingual.jsonl`, `subtaskA_dev_monolingual.jsonl`, `subtaskA_train_multilingual.jsonl`, `subtaskA_dev_multilingual.jsonl`.
   - Official Subtask A label schema: `0=human`, `1=machine`.
   - Current converter: `scripts/prepare_semeval_subtaskA.py`.
   - Status: official Subtask A monolingual train/dev and multilingual train/dev files downloaded; locked full TF-IDF Logistic Regression, calibration/selective prediction, bootstrap confidence intervals, and subgroup analysis completed for the manuscript.

2. MAGE
   - URL: https://aclanthology.org/2024.acl-long.3/
   - Repository: https://github.com/yafuly/MAGE
   - Hugging Face: https://huggingface.co/datasets/yaful/MAGE
   - License: Apache-2.0 according to the Hugging Face dataset card.
   - Current verified repo SHA: `342663f0a2b775455c023f5d36a1341ff0ec5402`
   - Current verified files:
     - `train.csv` (403,744,528 bytes)
     - `valid.csv` (72,276,577 bytes)
     - `test.csv` (71,739,623 bytes)
     - `test_ood_set_gpt.csv` (2,191,864 bytes)
     - `test_ood_set_gpt_para.csv` (3,614,100 bytes)
   - Purpose: out-of-distribution and wide testbed evaluation.
   - Status: full raw `train.csv`/`valid.csv`/`test.csv`, `test_ood_set_gpt.csv`, and `test_ood_set_gpt_para.csv` have been downloaded to `data/raw/mage/`; locked full TF-IDF Logistic Regression, calibration/selective prediction, bootstrap confidence intervals, and subgroup analysis completed for the manuscript.

3. RAID
   - Paper: https://arxiv.org/abs/2405.07940
   - Repository: https://github.com/liamdugan/raid
   - Hugging Face: https://huggingface.co/datasets/liamdugan/raid
   - License: MIT according to the Hugging Face dataset card.
   - Current verified repo SHA: `865cac74188466cb0c3b7574a10204007b57a459`
   - Current verified files:
     - `train.csv` (11,779,491,051 bytes)
     - `test.csv` (1,221,088,550 bytes)
     - `extra.csv` (3,707,337,095 bytes)
   - Purpose: related-work framing and repository/source notes. RAID is not used for the manuscript's main empirical claims.
   - Label note: Hugging Face README states `RAID-train` and `RAID-extra` include labels/metadata, while `RAID-test` is unlabeled and contains only `id` and `generation`.
   - Status: metadata verified; full locked RAID evaluation is outside the submitted manuscript scope. Earlier small labeled verification runs are historical development checks, not main evidence.

## Required checks before experiments

- Dataset license
- Allowed redistribution scope
- Exact version or commit hash
- Split definitions
- Label schema
- Language/domain/generator fields
- Known leakage risks
- Dataset size and local storage requirements
