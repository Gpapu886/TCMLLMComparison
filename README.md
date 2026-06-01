# Traditional Chinese Medicine Large Model Comparison

This repository provides the datasets:

**A Controlled Evaluation of Negative Transfer in LoRA-Based Fine-Tuning for Traditional Chinese Medicine Large Language Models**

## Overview

Multi-task fine-tuning is widely used to adapt large language models (LLMs) to domain-specific tasks. However, in professional domains such as Traditional Chinese Medicine (TCM), heterogeneous task data may introduce negative transfer and task interference. This repository supports a controlled benchmark-based evaluation of LoRA-based fine-tuning for TCM-oriented large language models.

The study first evaluates multiple open-source base models on the Medbench benchmark and then compares two LoRA fine-tuning settings:

1. **Four-category setting**: fine-tuning on four prescription-related TCM datasets.
2. **Seven-category setting**: fine-tuning on the same four datasets plus three heterogeneous datasets, including clinical cases, reading comprehension, and disease recommendation.

The goal is to examine whether adding heterogeneous TCM task data improves or degrades benchmark performance under the same LoRA configuration.
