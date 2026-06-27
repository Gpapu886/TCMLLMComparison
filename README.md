````markdown
# TCMLLMComparison

This repository contains code, configuration notes, and processed evaluation results for the manuscript:

**A benchmark and leakage-controlled evaluation of LoRA fine-tuning for traditional Chinese medicine language models**

## Overview

This study evaluates low-rank adaptation (LoRA) fine-tuning for traditional Chinese medicine (TCM) large language models from two complementary perspectives:

1. **Benchmark task-composition evaluation**  
   Five open-source foundation models were evaluated on MedBench. Qwen3-8B-Instruct achieved the highest overall score among the tested models and was selected as the base model for LoRA fine-tuning. Two LoRA task-composition settings were compared:
   - Four-category prescription-related setting
   - Seven-category heterogeneous setting

2. **Leakage-controlled medical-case evaluation**  
   A focused Qwen3-8B-Instruct + LoRA r16 model was evaluated on structured TCM medical-case generation. Exact and near-duplicate train-test overlap was audited, and a leakage-filtered clean test set was constructed.

## Main results

### MedBench task-composition evaluation

| Model setting | Overall | MKQA | MLG | CMR | MLU | MSE |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B-Instruct base | 53.7 | 62.2 | 69.8 | 57.0 | 59.4 | 20.0 |
| Four-category LoRA | 40.6 | 43.0 | 56.9 | 38.8 | 42.8 | 21.7 |
| Seven-category LoRA | 39.7 | 44.6 | 56.8 | 37.7 | 42.8 | 16.7 |

### Leakage-controlled medical-case evaluation

| Setting | n | Syndrome exact | Diagnosis exact | Treatment exact | Recommendation hit | Full-output exact |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen3 clean test | 3,186 | 0.3424 | 0.0000 | 0.0000 | 0.0832 | 0.0000 |
| LoRA r16 clean test | 3,186 | 0.7235 | 0.6431 | 0.7191 | 0.7869 | 0.5041 |
| LoRA r16 original test | 4,804 | 0.5175 | 0.6534 | 0.5183 | 0.5776 | 0.4419 |

## Repository structure

```text
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
  data_availability_note.md
````

## Data availability

This repository provides processed, non-identifying evaluation summaries and scripts used to reproduce the reported metrics. Raw clinical medical-case text is not publicly released in this repository because it may contain sensitive clinical information and is subject to institutional data-use restrictions. Access to restricted raw data may be requested from the corresponding author and relevant institutional data holder, subject to ethical approval and data-use agreement.

## Reproducing the evaluation

The main structured medical-case evaluation can be reproduced using:

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

For base-model evaluation, omit `--adapter_path`.

## Notes

All model outputs are for research evaluation only and should not be interpreted as clinical prescriptions or medical advice.

```
```
