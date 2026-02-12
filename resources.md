# Resource Catalog: Fairness-Aware Calibration of LLM Evaluators Using Tokenized Disclosures

## Research Hypothesis

Fine-tuning LLM evaluators to avoid penalizing tokenized AI assistance disclosures and to remove demographic interaction effects will operationalize fairness by design and improve the fairness of algorithmic judgment in ranking, hiring, and review systems.

---

## Papers (19 total)

### Core Papers (5)

1. **Large Language Models are not Fair Evaluators** (Wang et al., 2023)
   - File: `papers/2305.17926_llms_not_fair_evaluators.pdf`
   - Source: arXiv:2305.17926 / ACL 2024
   - Relevance: Foundational work on positional bias in LLM evaluators. Proposes MEC+BPC+HITLC calibration framework.

2. **Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge** (Ye et al., 2024)
   - File: `papers/2410.02736_justice_or_prejudice.pdf`
   - Source: arXiv:2410.02736
   - Relevance: Comprehensive taxonomy of 12 bias types in LLM judges. CALM framework for automated bias quantification.

3. **Does the Prompt-based LLM Recognize Students' Demographics and Introduce Bias in Essay Scoring?** (Yang et al., 2025)
   - File: `papers/2504.21330_llm_demographics_essay_scoring.pdf`
   - Source: arXiv:2504.21330
   - Relevance: Demonstrates demographic inference from text tokens introduces scoring bias. Directly motivates the disclosure penalty question.

4. **Exposing and Mitigating Calibration Biases and Demographic Unfairness in MLLM Few-Shot ICL** (Shen et al., 2025)
   - File: `papers/2506.23298_calibration_demographic_unfairness.pdf`
   - Source: arXiv:2506.23298
   - Relevance: CALIN algorithm -- most directly analogous approach. Training-free bi-level calibration for simultaneous accuracy and fairness improvement.

5. **Fairness in Automated Essay Scoring** (Schaller et al., 2024)
   - File: `papers/bea2024_fairness_essay_scoring.pdf`
   - Source: ACL BEA Workshop 2024
   - Relevance: Rigorous psychometric fairness evaluation framework (OSA/OSD/CSD metrics) for automated scoring systems.

### Calibration and Confidence (4)

6. **Just Ask for Calibration** (Kadavath et al., 2023)
   - File: `papers/2305.14975_just_ask_calibration.pdf`
   - Source: arXiv:2305.14975
   - Relevance: Verbalized confidence from RLHF-tuned LLMs is better calibrated than raw token probabilities.

7. **Calibrating LLM-Based Evaluator** (2024)
   - File: `papers/2404.02655_calibrating_fidelity.pdf`
   - Source: arXiv:2404.02655
   - Relevance: Calibration techniques specific to LLM-based evaluation systems.

8. **A Survey on Confidence Calibration of LLMs** (2023)
   - File: `papers/2311.08298_survey_confidence_calibration.pdf`
   - Source: arXiv:2311.08298
   - Relevance: Comprehensive overview of calibration methods for language models.

9. **Graph-based Confidence Calibration** (2024)
   - File: `papers/2411.02454_graph_confidence_calibration.pdf`
   - Source: arXiv:2411.02454
   - Relevance: Alternative calibration approaches that may inform post-hoc score adjustment.

### Bias, Fairness, and Demographics (6)

10. **Bias and Fairness in LLMs: A Survey** (Gallegos et al., 2024)
    - File: `papers/2309.00770_bias_fairness_survey.pdf`
    - Source: Computational Linguistics (MIT Press)
    - Relevance: Theoretical grounding for operationalizing fairness in LLM systems.

11. **A Survey on Fairness in LLMs** (2023)
    - File: `papers/2308.10149_survey_fairness_llm.pdf`
    - Source: arXiv:2308.10149
    - Relevance: Complementary fairness survey covering different mitigation strategies.

12. **Bias Mitigation in Fine-tuning Pre-trained Models** (2024)
    - File: `papers/2403.00625_bias_mitigation_finetuning.pdf`
    - Source: arXiv:2403.00625
    - Relevance: Weight importance neutralization strategy using Fisher information for debiasing during fine-tuning.

13. **LLM Demographic Inference** (2025)
    - File: `papers/2506.10922_llm_demographic_inference.pdf`
    - Source: arXiv:2506.10922
    - Relevance: Examines how LLMs infer demographic attributes from text, relevant to implicit bias in evaluation.

14. **Fairness in AI Recruitment** (2024)
    - File: `papers/2405.19699_fairness_ai_recruitment.pdf`
    - Source: arXiv:2405.19699
    - Relevance: Fairness considerations specific to AI-assisted hiring and ranking systems.

15. **Reasoning Towards Fairness** (2025)
    - File: `papers/2504.05632_reasoning_towards_fairness.pdf`
    - Source: arXiv:2504.05632
    - Relevance: Reasoning-based approaches to achieving fairness in LLM outputs.

### LLM-as-Judge and Evaluation (4)

16. **A Survey on LLM-as-a-Judge** (Zheng et al., 2024)
    - File: `papers/2411.15594_survey_llm_judge.pdf`
    - Source: arXiv:2411.15594
    - Relevance: Comprehensive survey establishing diversity bias as a distinct category in LLM evaluation.

17. **Self-Preference Bias in LLM-as-a-Judge** (2024)
    - File: `papers/2410.21819_self_preference_bias.pdf`
    - Source: arXiv:2410.21819
    - Relevance: Documents systematic self-enhancement bias that fairness-aware calibration must address.

18. **Evaluating Scoring Bias in LLM Evaluators** (2025)
    - File: `papers/2506.22316_evaluating_scoring_bias.pdf`
    - Source: arXiv:2506.22316
    - Relevance: Methods for detecting and measuring scoring biases in LLM evaluation systems.

19. **Humans or LLMs as the Judge** (2024)
    - File: `papers/emnlp2024_humans_or_llms_judge.pdf`
    - Source: EMNLP 2024
    - Relevance: Comparative analysis of human vs. LLM evaluation, establishing reliability benchmarks.

---

## Datasets (4 total)

| Dataset | Directory | Source | Size | Examples | Demographics |
|---------|-----------|--------|------|----------|-------------|
| PERSUADE 2.0 | `datasets/persuade_corpus_2.0/` | [GitHub](https://github.com/scrosseern/PERSUADE_corpus) | ~364KB (metadata) | 25,000+ essays | Gender, race, language background |
| Feedback Collection | `datasets/feedback_collection/` | [HuggingFace](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection) | ~490MB | 99,952 | N/A (rubric-based) |
| MT-Bench Human Judgments | `datasets/mt_bench_human_judgments/` | [HuggingFace](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) | ~3.0MB | 5,755 | N/A |
| BBQ (HELM) | `datasets/bbq/` | [HuggingFace](https://huggingface.co/datasets/lighteval/bbq_helm) | ~306KB | 1,000 | Age, disability, gender, nationality, race, religion, SES |

### Dataset Usage Notes

- **PERSUADE 2.0** is the primary dataset for essay scoring experiments with demographic attributes. The GitHub clone contains rubrics and metadata; full essay CSVs must be obtained from Kaggle.
- **Feedback Collection** provides training data for rubric-based LLM evaluators (Prometheus baseline).
- **MT-Bench** provides human and GPT-4 pairwise judgments for evaluator agreement analysis.
- **BBQ** provides bias benchmarking examples across multiple demographic axes.

---

## Code Repositories (3 total)

| Repository | Directory | Source | Paper |
|------------|-----------|--------|-------|
| FairEval | `code/FairEval/` | [GitHub](https://github.com/i-Eval/FairEval) | Wang et al. (2023) |
| Prometheus Eval | `code/prometheus-eval/` | [GitHub](https://github.com/prometheus-eval/prometheus-eval) | Kim et al. (2024) |
| CALIN | `code/calibration-fairness-mllm/` | [GitHub](https://github.com/xingbpshen/medical-calibration-fairness-mllm) | Shen et al. (2025) |

---

## Key Metrics

| Category | Metrics | Source |
|----------|---------|--------|
| Scoring agreement | QWK, Cohen's Kappa, Pearson r | Standard psychometrics |
| Fairness | OSA, OSD, CSD (threshold: \|metric\| > 0.10) | Loukina et al. (2019), RSMTool |
| Fairness (extended) | SMD, MAED, EOR | Yang et al. (2025), Shen et al. (2025) |
| Calibration | ECE, CCEG, ESCE | Calibration literature |
| Bias detection | Conflict Rate, Robustness Rate, BPDE | Wang et al. (2023), Ye et al. (2024) |

---

## Recommended Baselines

1. **Vanilla LLM-as-judge** (GPT-4o, Claude) -- no calibration
2. **Position-swapped averaging** -- Wang et al. (2023)
3. **Evidence-first prompting** -- reasoning tokens before scores
4. **Prometheus** -- open-source rubric-based evaluator
5. **CALIN** -- training-free bi-level calibration (adapted from medical domain)

---

## Identified Research Gaps

1. No work directly addresses how tokenized AI assistance disclosures affect LLM judge scoring.
2. Demographic interaction effects in LLM evaluation are understudied (Yang et al. 2025 is the first for essay scoring).
3. Fairness-aware calibration combining accuracy and demographic equity exists only for medical imaging (CALIN), not text evaluation.
4. No work examines how LLM evaluators for hiring/review applications handle AI-assisted content.
5. Fine-tuning specifically for fairness in LLM evaluator/judge models has not been explored.

---

## File Structure

```
.
├── literature_review.md          # Comprehensive literature review and synthesis
├── resources.md                  # This file -- resource catalog
├── papers/                       # 19 downloaded research papers (PDF)
│   ├── README.md
│   └── pages/                    # PDF chunks for reading (3 pages each)
├── datasets/                     # 4 downloaded datasets
│   ├── README.md
│   ├── .gitignore
│   ├── persuade_corpus_2.0/
│   ├── feedback_collection/
│   ├── mt_bench_human_judgments/
│   └── bbq/
├── code/                         # 3 cloned code repositories
│   ├── README.md
│   ├── FairEval/
│   ├── prometheus-eval/
│   └── calibration-fairness-mllm/
└── paper_search_results/         # Raw search results from paper-finder
```
