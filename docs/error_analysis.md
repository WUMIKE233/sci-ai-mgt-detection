# Error Analysis

This report summarizes subgroup failures and high-confidence errors for the current Transformer baselines. It is preliminary because the data are small sampled subsets.

Subgroup CSV: `outputs\error_analysis\subgroup_error_summary.csv`
High-confidence error CSV: `outputs\error_analysis\high_confidence_errors.csv`

## Worst Subgroups

| Dataset | Model | Group | Value | N | Accuracy | Error Rate | High-Conf Error Rate | FP machine | FN machine |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| MAGE small | fine_tuned_distilbert | original_src | pubmed_human_para | 80 | 0.1250 | 0.8750 | 0.7875 | 0 | 70 |
| MAGE small | frozen_distilbert_probe | original_src | pubmed_human_para | 80 | 0.2625 | 0.7375 | 0.3125 | 0 | 59 |
| SemEval Subtask A mono small | frozen_distilbert_probe | generator | bloomz | 2500 | 0.2804 | 0.7196 | 0.3688 | 0 | 1799 |
| SemEval Subtask A mono small | fine_tuned_distilbert | generator | bloomz | 2500 | 0.4116 | 0.5884 | 0.3476 | 0 | 1471 |
| MAGE small | frozen_distilbert_probe | original_src | imdb_human_para | 80 | 0.4500 | 0.5500 | 0.1250 | 0 | 44 |
| MAGE small | frozen_distilbert_probe | generator | machine_paraphrase | 320 | 0.4781 | 0.5219 | 0.1562 | 0 | 167 |
| SemEval Subtask A mono small | fine_tuned_distilbert | domain | peerread | 1000 | 0.5260 | 0.4740 | 0.3550 | 24 | 450 |
| SemEval Subtask A mono small | fine_tuned_distilbert | original_source | peerread | 1000 | 0.5260 | 0.4740 | 0.3550 | 24 | 450 |
| SemEval Subtask A mono small | frozen_distilbert_probe | domain | wikihow | 1000 | 0.5360 | 0.4640 | 0.2730 | 52 | 412 |
| SemEval Subtask A mono small | frozen_distilbert_probe | original_source | wikihow | 1000 | 0.5360 | 0.4640 | 0.2730 | 52 | 412 |
| RAID small | fine_tuned_distilbert | generator | mistral-chat | 22 | 0.5455 | 0.4545 | 0.1818 | 0 | 10 |
| SemEval Subtask A mono small | frozen_distilbert_probe | domain | peerread | 1000 | 0.5500 | 0.4500 | 0.2100 | 20 | 430 |
| SemEval Subtask A mono small | frozen_distilbert_probe | original_source | peerread | 1000 | 0.5500 | 0.4500 | 0.2100 | 20 | 430 |
| MAGE small | fine_tuned_distilbert | generator | machine_paraphrase | 320 | 0.5656 | 0.4344 | 0.3406 | 0 | 139 |
| SemEval Subtask A mono small | frozen_distilbert_probe | domain | reddit | 1000 | 0.5680 | 0.4320 | 0.2520 | 19 | 413 |
| SemEval Subtask A mono small | frozen_distilbert_probe | original_source | reddit | 1000 | 0.5680 | 0.4320 | 0.2520 | 19 | 413 |
| SemEval Subtask A mono small | fine_tuned_distilbert | domain | reddit | 1000 | 0.5760 | 0.4240 | 0.2770 | 22 | 402 |
| SemEval Subtask A mono small | fine_tuned_distilbert | original_source | reddit | 1000 | 0.5760 | 0.4240 | 0.2770 | 22 | 402 |
| SemEval Subtask A mono small | frozen_distilbert_probe | track | monolingual | 5000 | 0.5796 | 0.4204 | 0.1924 | 303 | 1799 |
| SemEval Subtask A mono small | frozen_distilbert_probe | domain | wikipedia | 1000 | 0.5890 | 0.4110 | 0.1740 | 100 | 311 |

## Main Observations

- MAGE paraphrased human-source examples remain the most severe observed failure mode.
- RAID synonym and some model-specific groups remain challenging despite strong aggregate metrics.
- SemEval Subtask A monolingual dev shows persistent unseen-generator difficulty for both frozen and fine-tuned DistilBERT baselines.
- High-confidence errors should be reviewed before writing final claims about reliability.

## Required Before Submission

- Repeat error analysis on larger samples and final model predictions.
- Include representative examples only when dataset license and journal policy permit text excerpts.
- Avoid treating detector outputs as proof of authorship.