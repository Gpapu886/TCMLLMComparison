# TCM-LLM-Comparison
### Leakage-Aware Evaluation & Conflict-Aware Retrieval for Structured TCM Medical Case Generation with LLMs

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3--8B--Instruct-22a565)
![Conference](https://img.shields.io/badge/ISAIMS-2026%20(ACM%20%2F%20EI)-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A reproducible pipeline for **structured Traditional Chinese Medicine (TCM) medical-case generation** with large language models, covering (1) a **leakage-controlled evaluation** of LoRA fine-tuning and (2) **CARE-TCM**, a conflict-aware retrieval-and-evidence-gating RAG framework. All experiments are built on **Qwen3-8B-Instruct** over **48,040 real TCM cases**, with strict train/test de-duplication, hybrid sparse–dense retrieval, a programmatic (non-prompt) evidence router, and paired significance testing.



## 1. Overview

Generating *structured* TCM records — syndrome differentiation (辨证), diagnosis (诊断), treatment principle (治法) and recommendation (推荐) — is easy to over-estimate in the lab for two reasons that this repository addresses:

1. **Data leakage inflates scores.** A leakage audit found that **>33%** of test cases were near-duplicates of training cases. We therefore build a four-stage cleaning pipeline and *leakage-filtered clean tests* before reporting any number.
2. **A single fixed strategy is brittle.** Plain LoRA, single-retriever RAG and naive retrieval all behave inconsistently when retrieved evidence conflicts with the case syndrome. **CARE-TCM** groups evidence by TCM syndrome and routes each case through one of three frozen branches (`LoRA-only / BM25-RAG / CARE-RAG`).

> The evidence gate is a **programmatic router whose thresholds are frozen on the validation set** — it is not an LLM prompt, so it never learns from the test distribution.

**Main result.** On a strict 3,154-case test set, CARE-TCM raises diagnosis exact accuracy from **0.6465 → 0.6874 (+4.09 pp)**, significant under a paired McNemar test (**p = 6.75×10⁻⁹**).



## 2. Key Contributions

- **Leakage audit & controlled splits** — exact-match + 64-bit SimHash (BLAKE2b-tokenized, similarity ≥ 0.90) near-duplicate detection; a four-stage pipeline of field extraction → quality filtering → NFKC normalization → de-duplication.
- **Systematic LoRA study** — Qwen3-8B-Instruct with LoRA rank r=16 across different task compositions, evaluated on five structured outputs over a 3,186-case leakage-filtered clean test.
- **CARE-TCM framework** — BM25 sparse + BGE-M3 dense dual retrieval, Reciprocal Rank Fusion (RRF), syndrome-consistent evidence grouping, and a validation-frozen three-way gate.
- **Rigorous evaluation** — per-field exact accuracy / hit-rate / F1, Bootstrap 95% confidence intervals, paired McNemar tests, and `input-only / answer-visible / counterfactual` audits that separately quantify retrieval gain and annotation-adoption risk.



## 3. Framework

mermaid
flowchart TD
    A["48,040 raw TCM cases"] --> B["Field extraction"]
    B --> C["Quality filtering"]
    C --> D["NFKC normalization"]
    D --> E["De-duplication: exact match + 64-bit SimHash (sim >= 0.90)"]
    E --> F["Leakage-controlled clean splits<br/>3,186 / 3,154 test cases"]
    Q["Input medical case"] --> R1["BM25 sparse retrieval"]
    Q --> R2["BGE-M3 dense retrieval"]
    R1 --> RF["RRF hybrid fusion"]
    R2 --> RF
    RF --> G["Syndrome-consistent evidence grouping"]
    F -.->|"thresholds frozen on validation"| GT["3-way evidence gate"]
    G --> GT
    GT -->|"LoRA-only"| M["Qwen3-8B-Instruct + LoRA r16"]
    GT -->|"BM25-RAG"| M
    GT -->|"CARE-RAG"| M
    M --> O["Structured output<br/>syndrome / diagnosis / treatment / recommendation"]




## 4. Datasets

| Item | Detail |
|---|---|
| Raw corpus | 48,040 TCM medical cases (public TCM task data + partner-hospital cases) |
| Preprocessing | Field extraction → quality filtering → NFKC → exact + SimHash de-duplication |
| Leakage found | >33% of raw test cases near-duplicated training cases |
| Clean test (LoRA study) | 3,186 leakage-filtered cases |
| Strict test (CARE-TCM) | 3,154 cases |
| Structured fields | syndrome differentiation, diagnosis, treatment principle, recommendation, full-output |

> **Data governance:** the raw records contain clinical content and are **not redistributed** in this repository. `data/` ships only an anonymized schema example and the full preprocessing / audit code so the pipeline can be reproduced on your own TCM corpus.



## 5. Repository Structure

text
TCMLLMComparison/
├── README.md
├── requirements.txt
├── configs/
│   ├── lora_r16.yaml            # LoRA rank/alpha/lr/dropout config
│   └── retrieval.yaml           # BM25 / BGE-M3 / RRF parameters
├── data/
│   ├── raw/                     # 48,040 raw cases (not redistributed)
│   ├── processed/               # cleaned splits & clean tests
│   └── schema_example.json      # one anonymized case
├── src/
│   ├── preprocess/              # extraction, filtering, NFKC, de-duplication
│   ├── leakage_audit/           # exact + SimHash near-duplicate detection
│   ├── finetune/                # LoRA/PEFT training on Qwen3-8B-Instruct
│   ├── retrieval/              # BM25, BGE-M3 dense retrieval, RRF fusion
│   ├── care_gate/              # syndrome grouping + frozen 3-way router
│   └── evaluate/               # exact/F1, Bootstrap CI, McNemar
├── scripts/
│   ├── run_preprocess.py
│   ├── run_leakage_audit.py
│   ├── run_lora.py
│   ├── run_retrieval_gate.py
│   └── run_evaluate.py
└── results/                     # result tables and figures




## 6. Quick Start

bash
# 0. environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# download Qwen3-8B-Instruct and BGE-M3 from Hugging Face first

# 1. preprocess and build leakage-controlled splits
python scripts/run_preprocess.py --config configs/retrieval.yaml

# 2. train/test leakage audit (exact + SimHash)
python scripts/run_leakage_audit.py

# 3. LoRA r16 fine-tuning on Qwen3-8B-Instruct
python scripts/run_lora.py --config configs/lora_r16.yaml

# 4. hybrid retrieval + CARE evidence gate, then evaluate
python scripts/run_retrieval_gate.py --config configs/retrieval.yaml
python scripts/run_evaluate.py


Core dependencies: torch, transformers, peft, datasets, accelerate, rank-bm25, FlagEmbedding (BGE-M3), scikit-learn, scipy, pandas, numpy, tqdm.



## 7. Main Results

### 7.1 LoRA r16 on the 3,186-case leakage-filtered clean test

| Structured output | Metric | Score |
|---|---|---:|
| Syndrome differentiation (辨证) | Exact accuracy | **72.35%** |
| Diagnosis (诊断) | Exact accuracy | **64.31%** |
| Treatment principle (治法) | Exact accuracy | **71.91%** |
| Recommendation (推荐) | Hit rate | **78.69%** |
| Full structured output | Exact accuracy | **50.41%** |

*Finding:* mixing heterogeneous tasks does not necessarily improve the medical benchmark, whereas a **case-focused LoRA r16** adapter clearly improves structured TCM output — but only after leakage is removed.

### 7.2 CARE-TCM on the 3,154-case strict test (diagnosis)

| Method | Diagnosis exact accuracy |
|---|---:|
| LoRA-only baseline | 0.6465 |
| **CARE-TCM (frozen gated routing)** | **0.6874** |
| Absolute gain | **+4.09 pp** |
| Paired significance | McNemar **p = 6.75×10⁻⁹** |



## 8. Evaluation & Auditing Protocol

- **Metrics:** per-field exact accuracy, hit rate and F1, plus full-output exact match.
- **Uncertainty:** Bootstrap 95% confidence intervals.
- **Significance:** paired McNemar test on the *same* test cases (two related proportions).
- **Three audit protocols:**
  - *input-only* — model never sees retrieved evidence;
  - *answer-visible* — evidence shown, measures adoption;
  - *counterfactual* — evidence perturbed to test whether gains come from retrieval rather than label leakage.



## 9. Papers & Citation

Two papers from this project were accepted by **ISAIMS 2026** (The 7th International Symposium on Artificial Intelligence in Medical Sciences), **ACM publication, EI-indexed**.

bibtex
@inproceedings{hou2026leakage,
  title     = {Leakage-Aware Evaluation of {LoRA} Fine-Tuning for Structured
               Traditional Chinese Medicine Medical Case Generation},
  author    = {Hou, Chao and Wang, Yang and Zhao, Duo and Tian, Bin},
  booktitle = {Proceedings of ISAIMS 2026 (ACM, EI)},
  year      = {2026}
}

@inproceedings{hou2026care,
  title     = {{CARE-TCM}: Conflict-Aware Retrieval and Evidence Gating for
               Structured Traditional Chinese Medicine Medical Case Generation},
  author    = {Hou, Chao and Wang, Yang and Zhao, Duo and Tian, Bin},
  booktitle = {Proceedings of ISAIMS 2026 (ACM, EI)},
  year      = {2026}
}


## 10. Acknowledgement

This work was supported by the **Medical–Engineering Interdisciplinary Program of Shanghai Seventh People's Hospital** (Grant No. **C80ZK230026**) and the research group at Shanghai Polytechnic University.

## 11. License & Contact

- Code released under the **MIT License**; clinical raw data are excluded for compliance.
- **Yang Wang** — replace-with-your-email@example.com · GitHub: [@Gpapu886](https://github.com/Gpapu886)
- Advisor: **Chao Hou**, houchao@sspu.edu.cn
