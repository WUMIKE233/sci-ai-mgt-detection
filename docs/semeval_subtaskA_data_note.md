# SemEval-2024 Task 8 Subtask A Data Note

## Purpose

This note records how SemEval-2024 Task 8 Subtask A will enter the manuscript pipeline. It is focused on the binary human-written versus machine-generated classification track, which is the part most directly aligned with the current paper.

## Official Source

- Official repository: https://github.com/mbzuai-nlp/SemEval2024-task8
- License shown by the official repository: Apache-2.0
- Dataset relationship: the task data are an extension of M4.
- Official download method: `gdown --folder https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc`

## Official Subtask A Files

After downloading the official Subtask A folder, place the files under:

- `data/raw/semeval2024_task8/subtaskA/data/subtaskA_train_monolingual.jsonl`
- `data/raw/semeval2024_task8/subtaskA/data/subtaskA_dev_monolingual.jsonl`
- `data/raw/semeval2024_task8/subtaskA/data/subtaskA_train_multilingual.jsonl`
- `data/raw/semeval2024_task8/subtaskA/data/subtaskA_dev_multilingual.jsonl`

The official README reports these Subtask A sizes:

- Monolingual train: 119,757 rows
- Monolingual dev: 5,000 rows
- Multilingual train: 172,417 rows
- Multilingual dev: 4,000 rows

## Label Mapping

Official Subtask A JSONL rows use:

- `label = 0`: human-written text
- `label = 1`: machine-generated text

The project converter maps these to:

- `human`
- `machine`

This keeps `machine` as the positive class across MAGE, RAID, and SemEval experiments.

## Converter

Run a small stratified conversion for pipeline validation:

```powershell
python scripts\prepare_semeval_subtaskA.py --track monolingual --max-train-per-group 500 --max-dev-per-group 500
```

Run a full conversion for final experiments:

```powershell
python scripts\prepare_semeval_subtaskA.py --track monolingual --max-train-per-group 0 --max-dev-per-group 0
```

Expected outputs:

- `data/processed/semeval2024_task8/semeval_subtaskA_monolingual_train.csv`
- `data/processed/semeval2024_task8/semeval_subtaskA_monolingual_dev.csv`
- `data/processed/semeval2024_task8/semeval_subtaskA_monolingual_metadata.json`

## Current Local Status

- Official monolingual train and dev JSONL files have been downloaded locally.
- The monolingual subset has been converted and evaluated with TF-IDF baselines and calibration/selective prediction.
- The multilingual train file has been downloaded and lightly verified: 172,417 JSONL rows with fields `id`, `label`, `model`, `source`, and `text`.
- Toy fixture validation remains available under `data/toy_semeval/` for interface testing.

Final manuscript claims require a locked full-processing protocol, official test/gold handling where permitted, and regenerated result tables.
