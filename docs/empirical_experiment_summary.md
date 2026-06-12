# Locked Empirical Experiment Summary

Bootstrap resamples per metric interval: `500`.

## Detector Performance

| Dataset | Protocol | Method | N | Accuracy | Macro-F1 | AUROC | Brier | ECE-10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SemEval-A monolingual full | official train-dev | TF-IDF LR | 5000 | 0.6518 [0.6382, 0.6635] | 0.6289 [0.6128, 0.6407] | 0.7444 [0.7311, 0.7571] | 0.2440 [0.2364, 0.2522] | 0.1896 [0.1776, 0.2016] |
| SemEval-A multilingual full | official train-dev | TF-IDF LR | 4000 | 0.5092 [0.4947, 0.5261] | 0.4590 [0.4440, 0.4748] | 0.4759 [0.4575, 0.4941] | 0.3346 [0.3251, 0.3445] | 0.2318 [0.2162, 0.2483] |
| MAGE in-distribution full | MAGE train-test | TF-IDF LR | 56819 | 0.7261 [0.7226, 0.7304] | 0.7261 [0.7226, 0.7304] | 0.8115 [0.8079, 0.8152] | 0.1790 [0.1770, 0.1808] | 0.0426 [0.0391, 0.0460] |
| MAGE GPT-4 OOD full | MAGE train to OOD GPT-4 | TF-IDF LR | 1562 | 0.6370 [0.6149, 0.6613] | 0.6368 [0.6147, 0.6608] | 0.7098 [0.6860, 0.7361] | 0.2256 [0.2133, 0.2368] | 0.1011 [0.0798, 0.1231] |
| MAGE paraphrase OOD full | MAGE train to paraphrase OOD | TF-IDF LR | 2362 | 0.5813 [0.5629, 0.6006] | 0.5682 [0.5486, 0.5870] | 0.6432 [0.6190, 0.6668] | 0.2610 [0.2511, 0.2712] | 0.2015 [0.1851, 0.2211] |
| SemEval-A monolingual full | fit-calibration-dev | Uncalibrated LR | 5000 | 0.6476 [0.6338, 0.6602] | 0.6237 [0.6082, 0.6367] | 0.7349 [0.7212, 0.7477] | 0.2458 [0.2383, 0.2538] | 0.1862 [0.1745, 0.1987] |
| SemEval-A monolingual full | fit-calibration-dev | Temp-scaled LR | 5000 | 0.6476 [0.6338, 0.6602] | 0.6237 [0.6082, 0.6367] | 0.7349 [0.7212, 0.7477] | 0.2779 [0.2684, 0.2885] | 0.2378 [0.2256, 0.2508] |
| SemEval-A monolingual full | fit-calibration-dev | Isotonic LR | 5000 | 0.6408 [0.6264, 0.6532] | 0.6132 [0.5979, 0.6261] | 0.7339 [0.7203, 0.7471] | 0.2796 [0.2700, 0.2903] | 0.2403 [0.2293, 0.2541] |
| SemEval-A multilingual full | fit-calibration-dev | Uncalibrated LR | 4000 | 0.5038 [0.4886, 0.5194] | 0.4601 [0.4439, 0.4760] | 0.4705 [0.4517, 0.4888] | 0.3319 [0.3227, 0.3415] | 0.2246 [0.2084, 0.2410] |
| SemEval-A multilingual full | fit-calibration-dev | Temp-scaled LR | 4000 | 0.5038 [0.4886, 0.5194] | 0.4601 [0.4439, 0.4760] | 0.4705 [0.4517, 0.4888] | 0.3716 [0.3598, 0.3834] | 0.3063 [0.2905, 0.3246] |
| SemEval-A multilingual full | fit-calibration-dev | Isotonic LR | 4000 | 0.5142 [0.4990, 0.5299] | 0.4425 [0.4282, 0.4587] | 0.4712 [0.4526, 0.4898] | 0.3676 [0.3559, 0.3792] | 0.2810 [0.2662, 0.3008] |
| MAGE in-distribution full | fit-calibration-test | Uncalibrated LR | 56819 | 0.7207 [0.7168, 0.7246] | 0.7207 [0.7167, 0.7245] | 0.8060 [0.8024, 0.8097] | 0.1815 [0.1796, 0.1834] | 0.0417 [0.0387, 0.0453] |
| MAGE in-distribution full | fit-calibration-test | Temp-scaled LR | 56819 | 0.7207 [0.7168, 0.7246] | 0.7207 [0.7167, 0.7245] | 0.8060 [0.8024, 0.8097] | 0.1927 [0.1903, 0.1950] | 0.1034 [0.0997, 0.1070] |
| MAGE in-distribution full | fit-calibration-test | Isotonic LR | 56819 | 0.6999 [0.6961, 0.7037] | 0.6944 [0.6906, 0.6983] | 0.8058 [0.8022, 0.8095] | 0.2098 [0.2073, 0.2123] | 0.1446 [0.1412, 0.1482] |
| MAGE GPT-4 OOD full | fit-calibration-OOD | Uncalibrated LR | 1562 | 0.6492 [0.6274, 0.6735] | 0.6489 [0.6269, 0.6726] | 0.7073 [0.6833, 0.7341] | 0.2249 [0.2125, 0.2363] | 0.0839 [0.0667, 0.1074] |
| MAGE GPT-4 OOD full | fit-calibration-OOD | Temp-scaled LR | 1562 | 0.6492 [0.6274, 0.6735] | 0.6489 [0.6269, 0.6726] | 0.7073 [0.6833, 0.7341] | 0.2444 [0.2279, 0.2584] | 0.1570 [0.1363, 0.1828] |
| MAGE GPT-4 OOD full | fit-calibration-OOD | Isotonic LR | 1562 | 0.6364 [0.6133, 0.6613] | 0.6257 [0.6030, 0.6518] | 0.7069 [0.6830, 0.7338] | 0.2549 [0.2383, 0.2712] | 0.1755 [0.1552, 0.2005] |
| MAGE paraphrase OOD full | fit-calibration-OOD | Uncalibrated LR | 2362 | 0.5893 [0.5707, 0.6075] | 0.5758 [0.5549, 0.5930] | 0.6451 [0.6215, 0.6689] | 0.2542 [0.2450, 0.2642] | 0.1905 [0.1753, 0.2117] |
| MAGE paraphrase OOD full | fit-calibration-OOD | Temp-scaled LR | 2362 | 0.5893 [0.5707, 0.6075] | 0.5758 [0.5549, 0.5930] | 0.6451 [0.6215, 0.6689] | 0.2793 [0.2673, 0.2922] | 0.2353 [0.2194, 0.2563] |
| MAGE paraphrase OOD full | fit-calibration-OOD | Isotonic LR | 2362 | 0.6503 [0.6321, 0.6693] | 0.6042 [0.5848, 0.6245] | 0.6450 [0.6213, 0.6689] | 0.2440 [0.2317, 0.2553] | 0.1700 [0.1558, 0.1919] |

## Calibration and Selective Prediction

| Dataset | Method | ECE-10 | Brier | Risk at 80% coverage | Conformal coverage at alpha=0.10 | Singleton rate |
|---|---|---:|---:|---:|---:|---:|
| SemEval-A monolingual full | Uncalibrated LR | 0.1862 | 0.2458 | 0.3255 | 0.6076 | 0.9256 |
| SemEval-A monolingual full | Temp-scaled LR | 0.2378 | 0.2779 | 0.3255 | 0.6076 | 0.9256 |
| SemEval-A monolingual full | Isotonic LR | 0.2403 | 0.2796 | 0.3265 | 0.6092 | 0.9310 |
| SemEval-A multilingual full | Uncalibrated LR | 0.2246 | 0.3319 | 0.4941 | 0.5118 | 0.9832 |
| SemEval-A multilingual full | Temp-scaled LR | 0.3063 | 0.3716 | 0.4941 | 0.5118 | 0.9832 |
| SemEval-A multilingual full | Isotonic LR | 0.2810 | 0.3676 | 0.4941 | 0.5140 | 0.9998 |
| MAGE in-distribution full | Uncalibrated LR | 0.0417 | 0.1815 | 0.2325 | 0.7846 | 0.8658 |
| MAGE in-distribution full | Temp-scaled LR | 0.1034 | 0.1927 | 0.2325 | 0.7846 | 0.8658 |
| MAGE in-distribution full | Isotonic LR | 0.1446 | 0.2098 | 0.2606 | 0.7512 | 0.8977 |
| MAGE GPT-4 OOD full | Uncalibrated LR | 0.0839 | 0.2249 | 0.3288 | 0.7228 | 0.8297 |
| MAGE GPT-4 OOD full | Temp-scaled LR | 0.1570 | 0.2444 | 0.3288 | 0.7228 | 0.8297 |
| MAGE GPT-4 OOD full | Isotonic LR | 0.1755 | 0.2549 | 0.3352 | 0.6908 | 0.8854 |
| MAGE paraphrase OOD full | Uncalibrated LR | 0.1905 | 0.2542 | 0.3958 | 0.6765 | 0.8133 |
| MAGE paraphrase OOD full | Temp-scaled LR | 0.2353 | 0.2793 | 0.3958 | 0.6765 | 0.8133 |
| MAGE paraphrase OOD full | Isotonic LR | 0.1700 | 0.2440 | 0.3190 | 0.7091 | 0.8654 |

## Highest Error Subgroups

| Dataset | Method | Group | Value | N | Accuracy | Error rate | High-conf error rate |
|---|---|---|---|---:|---:|---:|---:|
| SemEval-A multilingual full | TF-IDF LR | generator | davinci | 500 | 0.0540 | 0.9460 | 0.5060 |
| SemEval-A multilingual full | TF-IDF LR | original_model | davinci | 500 | 0.0540 | 0.9460 | 0.5060 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_flan_t5_xxl | 98 | 0.1327 | 0.8673 | 0.1429 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_flan_t5_small | 99 | 0.1414 | 0.8586 | 0.1212 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_flan_t5_xl | 96 | 0.1562 | 0.8438 | 0.1562 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_text-davinci-002 | 99 | 0.1616 | 0.8384 | 0.1717 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_flan_t5_large | 99 | 0.1717 | 0.8283 | 0.1818 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_flan_t5_base | 98 | 0.1939 | 0.8061 | 0.1224 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_65B | 98 | 0.2143 | 0.7857 | 0.1633 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_7B | 96 | 0.2188 | 0.7812 | 0.1458 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_30B | 98 | 0.2245 | 0.7755 | 0.2347 |
| SemEval-A multilingual full | TF-IDF LR | generator | chatGPT | 1500 | 0.2547 | 0.7453 | 0.1793 |
| SemEval-A multilingual full | TF-IDF LR | original_model | chatGPT | 1500 | 0.2547 | 0.7453 | 0.1793 |
| MAGE in-distribution full | TF-IDF LR | original_src | yelp_machine_continuation_13B | 98 | 0.2755 | 0.7245 | 0.1327 |
| MAGE GPT-4 OOD full | TF-IDF LR | original_src | imdb_gpt4 | 200 | 0.2800 | 0.7200 | 0.0600 |
