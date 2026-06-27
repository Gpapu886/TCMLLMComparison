# Data sources

The JSON task files used in this study were derived from publicly released resources associated with the SmartSage-ZYLLM-32B / 智医灵枢 TCM LLM project:

- Source project: https://ai.gitee.com/ljt365fir/SmartSage-ZYLLM-32B

This study did not construct a new clinical database from hospital records. Instead, we selected task-specific JSON files from the publicly released integrated resources and used them for controlled LoRA fine-tuning and evaluation.

## Task-composition experiment

The four-category setting used:

- `choice_herb_formula.json`
- `entity_extraction.json`
- `knowledge.json`
- `recommend_formula.json`

The seven-category setting used the same four files plus:

- `admet.json`
- `medical_case.json`
- `recommend_disease.json`

## Leakage-controlled medical-case experiment

The leakage-controlled medical-case evaluation used the integrated `medical_case.json` file from the same public resource. We split the data into training, validation, original test, and leakage-filtered clean test subsets, and then performed exact-duplicate and near-duplicate leakage auditing.

## Raw data redistribution

This repository does not redistribute the complete raw `medical_case.json`, train/test JSONL files, or full model-output files because the source data are released by a third-party project and the medical-case text may contain clinical case descriptions.

To reproduce the full pipeline, researchers should obtain the original JSON resources from the SmartSage-ZYLLM-32B project and then run the scripts and evaluation pipeline provided in this repository.
