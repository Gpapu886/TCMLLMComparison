````markdown
# TCMLLMComparison

This repository contains the code, processed evaluation summaries, and result tables for the manuscript:

**A benchmark and leakage-controlled evaluation of LoRA fine-tuning for traditional Chinese medicine language models**

## Overview

This study evaluates low-rank adaptation (LoRA) fine-tuning for traditional Chinese medicine (TCM) large language models from two complementary perspectives:

1. **Benchmark task-composition evaluation**  
   We evaluated five open-source foundation models on MedBench and selected Qwen3-8B-Instruct as the base model for subsequent LoRA fine-tuning. Two task-composition settings were compared:
   - A four-category prescription-related LoRA setting
   - A seven-category heterogeneous LoRA setting

2. **Leakage-controlled medical-case evaluation**  
   We evaluated a focused Qwen3-8B-Instruct + LoRA r16 model for structured TCM medical-case generation. Exact and near-duplicate train-test overlap was audited, and a leakage-filtered clean test set was constructed.

The goal of this repository is to provide scripts and processed non-identifying result files that support the numerical findings reported in the manuscript.

## Data sources

The JSON task files used in this study were derived from publicly released resources associated with the SmartSage-ZYLLM-32B / 智医灵枢 TCM LLM project:

- Source project: https://ai.gitee.com/ljt365fir/SmartSage-ZYLLM-32B

We did not construct a new clinical database from hospital records. Instead, we selected task-specific JSON files from the publicly released integrated resources and used them for controlled LoRA fine-tuning and evaluation.

## Task-composition experiment

The four-category setting used the following JSON files:

- `choice_herb_formula.json`
- `entity_extraction.json`
- `knowledge.json`
- `recommend_formula.json`

The seven-category setting used the same four files plus:

- `admet.json`
- `medical_case.json`
- `recommend_disease.json`

The MedBench task-composition results were:

| Model setting | Overall | MKQA | MLG | CMR | MLU | MSE |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B-Instruct base | 53.7 | 62.2 | 69.8 | 57.0 | 59.4 | 20.0 |
| Four-category LoRA | 40.6 | 43.0 | 56.9 | 38.8 | 42.8 | 21.7 |
| Seven-category LoRA | 39.7 | 44.6 | 56.8 | 37.7 | 42.8 | 16.7 |

These results suggest that broader heterogeneous task mixing did not improve MedBench performance under the tested LoRA configuration.

## Leakage-controlled medical-case experiment

The leakage-controlled medical-case experiment used the integrated `medical_case.json` file from the same public resource.

The dataset was split into:

| Split | Number of cases |
|---|---:|
| Training set | 38,432 |
| Validation set | 4,804 |
| Original test set | 4,804 |

Train-test leakage auditing identified exact and near-duplicate overlap between the training and original test sets.

| Item | Value |
|---|---:|
| Original test cases | 4,804 |
| Exact duplicate cases removed | 387 |
| Additional near-duplicate cases removed | 1,231 |
| Total removed cases | 1,618 |
| Near-duplicate threshold | 0.90 |
| Clean test cases retained | 3,186 |
| Removed rate | 33.68% |
| Clean-test retention rate | 66.32% |

## Main structured medical-case results

| Setting | n | Syndrome exact | Diagnosis exact | Treatment exact | Recommendation hit | Full-output exact |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen3 clean test | 3,186 | 0.3424 | 0.0000 | 0.0000 | 0.0832 | 0.0000 |
| LoRA r16 clean test | 3,186 | 0.7235 | 0.6431 | 0.7191 | 0.7869 | 0.5041 |
| LoRA r16 original test | 4,804 | 0.5175 | 0.6534 | 0.5183 | 0.5776 | 0.4419 |
| LoRA r16 clean minus base clean | 3,186 | +0.3810 | +0.6431 | +0.7191 | +0.7037 | +0.5041 |

## Bootstrap 95% confidence intervals

Bootstrap confidence intervals were calculated using 1,000 non-parametric bootstrap resamples.

| Setting | Metric | Mean | 95% CI low | 95% CI high |
|---|---|---:|---:|---:|
| Base clean test | Syndrome exact | 0.3424 | 0.3258 | 0.3597 |
| Base clean test | Diagnosis exact | 0.0000 | 0.0000 | 0.0000 |
| Base clean test | Treatment exact | 0.0000 | 0.0000 | 0.0000 |
| Base clean test | Recommendation hit | 0.0832 | 0.0744 | 0.0926 |
| Base clean test | Full-output exact | 0.0000 | 0.0000 | 0.0000 |
| LoRA r16 clean test | Syndrome exact | 0.7235 | 0.7094 | 0.7385 |
| LoRA r16 clean test | Diagnosis exact | 0.6431 | 0.6262 | 0.6594 |
| LoRA r16 clean test | Treatment exact | 0.7191 | 0.7028 | 0.7351 |
| LoRA r16 clean test | Recommendation hit | 0.7869 | 0.7718 | 0.8004 |
| LoRA r16 clean test | Full-output exact | 0.5041 | 0.4871 | 0.5207 |
| LoRA r16 original test | Syndrome exact | 0.5175 | 0.5035 | 0.5331 |
| LoRA r16 original test | Diagnosis exact | 0.6534 | 0.6399 | 0.6667 |
| LoRA r16 original test | Treatment exact | 0.5183 | 0.5037 | 0.5327 |
| LoRA r16 original test | Recommendation hit | 0.5776 | 0.5635 | 0.5924 |
| LoRA r16 original test | Full-output exact | 0.4419 | 0.4280 | 0.4557 |

## Repository structure

```text
TCMLLMComparison/
  README.md
  requirements.txt

  scripts/
    infer_eval_medical_case_multifield.py
    infer_medbench_jsonl.py
    train_qwen3_lora_ablation.py

  results/
    final_main_results_table.csv
    final_main_results_table.json
    bootstrap_95ci_main_metrics.csv
    clean_test_no_overlap_090_summary.json
    medbench_task_composition_results.csv

  docs/
    data_sources.md
    data_availability_note.md
````

## Files included in this repository

This repository provides non-identifying processed outputs and reproducibility materials, including:

* Evaluation scripts
* Task-file list
* Main result tables
* Leakage-audit summary statistics
* Bootstrap confidence intervals
* MedBench task-composition results
* Documentation of data sources and evaluation settings

## Files not included in this repository

This repository does **not** redistribute the complete raw medical-case text or full model-output files, including:

* `medical_case.json`
* `train.jsonl`
* `val.jsonl`
* `test.jsonl`
* `test_clean_no_overlap_090.jsonl`
* full raw model prediction files
* files containing clinical case previews or full case text

The reason is that the source data are released by a third-party project, and the medical-case text may contain clinical case descriptions. To avoid inappropriate redistribution of third-party clinical text, this repository only provides processed non-identifying summaries and evaluation results.

## Reproducing the medical-case evaluation

After obtaining the original JSON resources from the public SmartSage-ZYLLM-32B project, the structured medical-case evaluation can be reproduced using:

```bash
python scripts/infer_eval_medical_case_multifield.py \
  --base_model /path/to/Qwen3-8B-Instruct \
  --adapter_path /path/to/lora_adapter \
  --test_file /path/to/test_clean_no_overlap_090.jsonl \
  --raw_eval_file outputs/eval/model_raw_eval.json \
  --multifield_file outputs/eval/model_multifield_eval.json \
  --max_input_tokens 4096 \
  --max_new_tokens 768
```

For base-model evaluation, omit the `--adapter_path` argument:

```bash
python scripts/infer_eval_medical_case_multifield.py \
  --base_model /path/to/Qwen3-8B-Instruct \
  --test_file /path/to/test_clean_no_overlap_090.jsonl \
  --raw_eval_file outputs/eval/base_raw_eval.json \
  --multifield_file outputs/eval/base_multifield_eval.json \
  --max_input_tokens 4096 \
  --max_new_tokens 768
```

## Reproducing bootstrap confidence intervals

Bootstrap confidence intervals were computed from the per-sample binary metrics in the structured evaluation result files. The processed confidence-interval table is provided in:

```text
results/bootstrap_95ci_main_metrics.csv
```

## Environment

The main experiments were conducted using:

```text
Python 3.10
PyTorch 2.6.0 + CUDA 12.4
Transformers 4.51.3
PEFT 0.19.1
GPU: NVIDIA A800 80GB PCIe
```

A minimal `requirements.txt` can include:

```text
torch
transformers
peft
tqdm
numpy
pandas
```

Version locking is recommended when reproducing the full pipeline.

## Data availability statement

The task datasets analysed in this study were derived from publicly released JSON resources associated with the SmartSage-ZYLLM-32B / 智医灵枢 TCM LLM project:

https://ai.gitee.com/ljt365fir/SmartSage-ZYLLM-32B

This repository provides the code, task-file list, preprocessing and evaluation scripts, leakage-audit summaries, processed non-identifying result tables, bootstrap confidence-interval results, and MedBench task-composition result files supporting the findings of the manuscript.

Because the original resources are released by a third-party project and the medical-case file may contain clinical case descriptions, the complete raw medical-case text is not redistributed in this repository. Researchers who wish to reproduce the full pipeline should obtain the original JSON resources from the public SmartSage-ZYLLM-32B project and then run the scripts provided here.

## Clinical disclaimer

All model outputs and evaluation results in this repository are for computational research only. They should not be interpreted as clinical prescriptions, medical advice, diagnostic recommendations, or treatment guidance.

## Citation

If this work is useful for your research, please cite the associated manuscript:

```text
Hou C, Wang Y, Zhao D, Zhang J. A benchmark and leakage-controlled evaluation of LoRA fine-tuning for traditional Chinese medicine language models.
```

The formal citation will be updated after publication.

## License

Please check the license and usage terms of the original SmartSage-ZYLLM-32B / 智医灵枢 resources before using or redistributing any derived data. This repository only distributes code and non-identifying processed evaluation summaries generated for the manuscript.

A separate license file may be added for the code in this repository.

```
```
