# Code Repositories

Code repositories cloned for the project "Fairness-Aware Calibration of LLM Evaluators Using Tokenized Disclosures."

## FairEval

- **Directory:** `FairEval/`
- **Source:** https://github.com/i-Eval/FairEval
- **Paper:** Wang et al. (2023), "Large Language Models are not Fair Evaluators"
- **Description:** Implementation of the three-part calibration framework for LLM evaluators: Multiple Evidence Calibration (MEC), Balanced Position Calibration (BPC), and Human-in-the-Loop Calibration (HITLC). Includes code for measuring positional bias via conflict rate and entropy-based uncertainty detection (BPDE).
- **Key components:**
  - Evidence-first prompting templates
  - Position-swapping evaluation scripts
  - Entropy-based human annotation selection
- **Size:** ~1.9MB

## Prometheus Eval

- **Directory:** `prometheus-eval/`
- **Source:** https://github.com/prometheus-eval/prometheus-eval
- **Paper:** Kim et al. (2024), "Prometheus: Inducing Fine-Grained Evaluation Capability in Language Models" (ICLR 2024)
- **Description:** Open-source LLM evaluator trained on rubric-based evaluation. Includes the Feedback Collection dataset generation pipeline, model training code, and evaluation scripts. Provides a baseline for rubric-based automated evaluation.
- **Key components:**
  - Model weights and inference scripts
  - Feedback Collection dataset generation
  - Evaluation benchmarks and comparison scripts
- **Size:** ~37MB

## Calibration-Fairness-MLLM (CALIN)

- **Directory:** `calibration-fairness-mllm/`
- **Source:** https://github.com/xingbpshen/medical-calibration-fairness-mllm
- **Paper:** Shen et al. (2025), "Exposing and Mitigating Calibration Biases and Demographic Unfairness in MLLM Few-Shot ICL"
- **Description:** Implementation of the CALIN algorithm -- a training-free bi-level calibration approach that simultaneously improves prediction accuracy and demographic fairness. Operates on token-level predicted probabilities with population-level calibration (L1) followed by subgroup-level calibration (L2) with exponential decay regularization.
- **Key components:**
  - CALIN calibration algorithm implementation
  - ECE, EOR, CCEG, ESCE metric computation
  - Experiment scripts for PAPILA, HAM10000, MIMIC-CXR datasets
- **Size:** ~2.0MB

## Usage

These repositories are included as reference implementations. They are not dependencies of the main project but serve as:

1. **Baselines** for comparing calibration approaches
2. **Reference implementations** for evaluation metrics and bias measurement
3. **Templates** for adapting fairness-aware calibration to LLM text evaluation
