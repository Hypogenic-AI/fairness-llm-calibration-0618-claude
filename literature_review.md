# Literature Review: Fairness-Aware Calibration of LLM Evaluators Using Tokenized Disclosures

## Research Area Overview

This literature review surveys work at the intersection of LLM-as-a-judge evaluation, calibration of language model confidence, fairness/bias in automated scoring, and AI-generated text disclosure. The research hypothesis proposes fine-tuning LLM evaluators to avoid penalizing tokenized AI assistance disclosures and to remove demographic interaction effects, thereby operationalizing fairness by design in ranking, hiring, and review systems.

The field has converged on several key themes: (1) LLM evaluators exhibit systematic biases including positional, demographic, and self-preference biases; (2) calibration techniques can partially mitigate these biases; (3) demographic attributes influence scoring in automated essay evaluation; and (4) AI-generated text detection itself carries fairness implications.

---

## Key Papers

### 1. Large Language Models are not Fair Evaluators (Wang et al., 2023)
- **Authors:** Wang, Li, Chen, Cai, Zhu, Lin, Cao, Liu, Liu, Sui
- **Year:** 2023 | **Source:** arXiv:2305.17926, ACL 2024
- **Key Contribution:** Demonstrates severe positional bias in LLM evaluators. Proposes a three-part calibration framework: Multiple Evidence Calibration (MEC), Balanced Position Calibration (BPC), and Human-in-the-Loop Calibration (HITLC).
- **Methodology:** Evidence-first prompting (reversing score-before-explanation order), position swapping, and entropy-based uncertainty detection (BPDE) to identify examples needing human review.
- **Datasets:** Vicuna Benchmark (80 questions, 9 categories). Models: GPT-4, ChatGPT.
- **Results:** Vanilla GPT-4 achieves only 52.7% accuracy on 3-way classification. Full calibration (MEC+BPC) improves by +9.8%. With 20% human annotation, GPT-4 reaches 73.8% -- matching human performance.
- **Key Metric:** Conflict Rate (proportion of examples where swapping response order reverses judgment): 46.3% for GPT-4.
- **Code:** https://github.com/i-Eval/FairEval
- **Relevance:** Foundational work establishing that LLM evaluators need calibration. The evidence-first prompting (forcing reasoning tokens before scores) is a form of tokenized disclosure that calibrates judgment.

### 2. Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge (Ye et al., 2024)
- **Authors:** Ye, Wang, Huang, Chen, Zhang, Moniz, Gao, Geyer, Huang, Chen, Chawla, Zhang
- **Year:** 2024 | **Source:** arXiv:2410.02736
- **Key Contribution:** Comprehensive taxonomy of 12 distinct bias types in LLM judges. Introduces CALM (Comprehensive Assessment of Language Model Judge Biases) framework for automated bias quantification.
- **12 Biases:** Position, Verbosity, Compassion-Fade, Bandwagon, Distraction, Fallacy-Oversight, Authority, Sentiment, Diversity (demographic), Chain-of-Thought, Self-Enhancement, Refinement-Aware.
- **Datasets:** GSM8K, MATH, ScienceQA (fact-related); DPO datasets (alignment); CommonsenseQA, TruthfulQA (refinement). 6 LLM judges tested.
- **Results:** Position bias most severe (ChatGPT RR=0.566). Demographic (diversity) bias: ChatGPT RR=0.679, Claude-3.5 most resistant at 0.914. Self-enhancement error rate up to 16.1% (Qwen2). Biases significantly worse on alignment data (subtle quality differences) than fact-related data.
- **Code:** https://llm-judge-bias.github.io/
- **Relevance:** Provides the most comprehensive bias taxonomy for LLM evaluators. The diversity bias findings directly demonstrate that demographic tokens in evaluation context shift judgments. Essential baseline for measuring calibration effectiveness.

### 3. Does the Prompt-based LLM Recognize Students' Demographics and Introduce Bias in Essay Scoring? (Yang et al., 2025)
- **Authors:** Yang, Rakovic, Gasevic, Chen (Monash University)
- **Year:** 2025 | **Source:** arXiv:2504.21330
- **Key Contribution:** Demonstrates that GPT-4o can infer demographic attributes from essay text (especially language background with ~99% coverage, ~82% accuracy) and that bias worsens when demographics are correctly identified.
- **Methodology:** Two-phase study: (1) demographic inference via prompting, (2) scoring with fairness analysis. Weighted multivariate regression with inverse probability weighting.
- **Dataset:** PERSUADE 2.0 corpus (25,000+ argumentative essays, grades 6-12, with gender, race, language background metadata). CC BY 4.0 license.
- **Results:** Scoring errors for non-native English speakers increase when GPT-4o correctly identifies them (Correctness x Language interaction coefficient 0.502, p<0.05). Gender bias was minimal. MAED for non-native speakers: -0.213 (greater errors) in "Correct" group.
- **Fairness Metrics:** OSA, OSD, CSD, MAED with inverse probability weighting.
- **Relevance:** Directly relevant -- shows that implicit demographic inference from text tokens introduces scoring bias. Raises critical question: if tokenized disclosures make demographics explicit, will bias worsen? Suggests calibration must operate post-hoc rather than within prompts.

### 4. A Survey on LLM-as-a-Judge (Zheng et al., 2024)
- **Authors:** Zheng et al.
- **Year:** 2024 | **Source:** arXiv:2411.15594
- **Key Contribution:** Comprehensive survey identifying diversity bias (race, gender, sexual orientation) as a distinct category of LLM evaluation bias alongside positional, length, concreteness, and authority biases.
- **Relevance:** Establishes the conceptual framework for understanding bias categories in LLM evaluation systems.

### 5. Exposing and Mitigating Calibration Biases and Demographic Unfairness in MLLM Few-Shot ICL (Shen et al., 2025)
- **Authors:** Shen, Szeto, Li, Huang, Arbel (McGill/Mila)
- **Year:** 2025 | **Source:** arXiv:2506.23298
- **Key Contribution:** Introduces CALIN, a training-free bi-level calibration algorithm that calibrates MLLM predictions and enforces fairness across demographic subgroups at inference time.
- **Methodology:** Population-level calibration (L1) followed by subgroup-level calibration (L2) with exponential decay regularization. Operates on token-level predicted probabilities.
- **Datasets:** PAPILA (glaucoma), HAM10000 (skin cancer), MIMIC-CXR (chest X-ray). Demographic attributes: sex, age.
- **Results:** CALIN reduces ECE from 23.70 to 2.68 (HAM10000), CCEG for age from 30.25 to 3.14, with minimal accuracy trade-off.
- **Metrics:** ECE, EOR (equalized odds ratio), CCEG (confidence calibration error gap), ESCE (equity-scaling measure).
- **Code:** https://github.com/xingbpshen/medical-calibration-fairness-mllm
- **Relevance:** Most directly analogous to the proposed research. Demonstrates that training-free calibration can simultaneously improve accuracy and fairness. The bi-level approach (population then subgroup) provides a template for fairness-aware calibration of LLM evaluators.

### 6. Fairness in Automated Essay Scoring (Schaller et al., BEA 2024)
- **Authors:** Schaller, Ding, Horbach, Meyer, Jansen
- **Year:** 2024 | **Source:** ACL BEA Workshop 2024
- **Key Contribution:** Evaluates fairness of AES (SVM, BERT, GPT-4) on German essays with demographic and psychological (cognitive ability) variables. Uses RSMTool fairness framework.
- **Dataset:** DARIUS corpus (4,589 German argumentative essays with grade, gender, language, cognitive ability metadata).
- **Results:** No model showed unfairness (OSA/OSD/CSD < 0.10) on representative training data. Training on skewed subgroups produces performance disparities. GPT-4 zero-shot was fair but inconsistent across tasks.
- **Fairness Metrics:** OSA, OSD, CSD (R^2-based regression metrics), threshold 0.10.
- **Code:** https://github.com/darius-ipn/fairness_AES
- **Relevance:** Provides a rigorous psychometric fairness evaluation framework adaptable to LLM evaluator calibration. Demonstrates that training data composition is critical for fair outcomes.

### 7. Self-Preference Bias in LLM-as-a-Judge (arXiv:2410.21819, 2024)
- **Key Contribution:** Documents that LLMs systematically overrate their own outputs when serving as evaluators.
- **Relevance:** Self-enhancement bias is a confound that fairness-aware calibration must address, especially in systems where the same model generates and evaluates.

### 8. Just Ask for Calibration (Kadavath et al., 2023)
- **Year:** 2023 | **Source:** arXiv:2305.14975
- **Key Contribution:** Shows that verbalized confidence scores from RLHF-tuned LLMs are better calibrated than raw token probabilities. Proposes strategies for eliciting calibrated confidence.
- **Relevance:** Establishes that post-RLHF calibration is degraded and that verbal/token-level confidence elicitation is a viable calibration mechanism.

### 9. Bias and Fairness in LLMs: A Survey (Gallegos et al., 2024)
- **Year:** 2024 | **Source:** Computational Linguistics (MIT Press), arXiv:2309.00770
- **Key Contribution:** Comprehensive taxonomy of bias evaluation and mitigation for LLMs. Defines distinct facets of harm and proposes three taxonomies for metrics, datasets, and mitigation.
- **Relevance:** Provides the theoretical grounding for operationalizing fairness in LLM systems.

### 10. Bias Mitigation in Fine-tuning Pre-trained Models (arXiv:2403.00625, 2024)
- **Key Contribution:** Introduces a weight importance neutralization strategy using Fisher information across demographic groups, integrated with matrix factorization for efficient debiasing during fine-tuning.
- **Relevance:** Provides a fine-tuning-based debiasing approach that could be adapted for training fairness-aware LLM evaluators.

---

## Common Methodologies

- **Position swapping and averaging:** Used in Wang et al. (2023) and standard in LLM-as-judge literature to mitigate positional bias.
- **Evidence-first prompting:** Forcing reasoning before scoring to ground judgments in generated tokens (Wang et al., 2023).
- **Regression-based fairness metrics (OSA/OSD/CSD):** From psychometrics (Loukina et al., 2019), used in educational AES fairness evaluation.
- **Robustness Rate (RR):** From CALM (Ye et al., 2024), measuring consistency under bias injection.
- **Calibration error metrics (ECE, CCEG):** From calibration literature, extended for subgroup fairness.
- **Inverse probability weighting:** For handling demographic imbalance in fairness analysis.

## Standard Baselines

- **Vanilla LLM-as-judge:** Single evaluation without calibration (Wang et al., 2023)
- **Prometheus:** Open-source LLM evaluator trained on rubric-based Feedback Collection dataset (ICLR 2024)
- **GPT-4 / GPT-4o as evaluator:** De facto standard in evaluation benchmarks
- **Human evaluation:** Gold standard, typically 3 annotators with majority vote
- **Random baseline:** For measuring above-chance performance of bias detection

## Evaluation Metrics

- **Accuracy/Agreement:** Cohen's Kappa, Pearson correlation, QWK for scoring alignment
- **Fairness:** OSA, OSD, CSD (regression R^2), SMD, MAED, CCEG, EOR
- **Calibration:** ECE (Expected Calibration Error), ESCE (equity-scaling)
- **Bias Detection:** Conflict Rate, Robustness Rate, BPDE (entropy)
- **Unfairness threshold:** |metric| > 0.10 (from educational measurement literature)

## Datasets in the Literature

| Dataset | Used In | Task | Demographics | Size |
|---------|---------|------|--------------|------|
| PERSUADE 2.0 | Yang et al. 2025 | Essay scoring | Gender, race, language | 25,000+ essays |
| Vicuna Benchmark | Wang et al. 2023 | LLM evaluation | N/A | 80 questions |
| CALM datasets | Ye et al. 2024 | Bias quantification | 6 identity groups | ~1,439 samples |
| MT-Bench | Zheng et al. 2023 | LLM evaluation | N/A | 80 questions + 3.3K judgments |
| Feedback Collection | Prometheus | Rubric-based eval | N/A | 100K feedback instances |
| DARIUS | Schaller et al. 2024 | German AES | Gender, grade, language, KFT | 4,589 essays |
| BBQ | Parrish et al. 2022 | Bias in QA | 13 demographic axes | ~58K |
| HolisticBias | Meta 2022 | Bias evaluation | 600 identity descriptors | 450K+ prompts |

## Gaps and Opportunities

1. **No work directly addresses tokenized AI disclosures in evaluation:** While watermarking and AI detection are studied, no paper examines how explicit "AI-assisted" disclosures in evaluated text affect LLM judge scoring.

2. **Demographic interaction effects in LLM evaluation are understudied:** Yang et al. (2025) is the first to examine demographic inference + scoring bias interaction in prompt-based LLMs, but only for essay scoring.

3. **Fairness-aware calibration is nascent:** CALIN (Shen et al., 2025) is the only work that explicitly combines calibration and demographic fairness, and it operates in medical imaging, not text evaluation.

4. **Hiring/ranking system evaluation:** While algorithmic fairness in hiring is studied (NYC Local Law 144, EU AI Act), no work examines how LLM evaluators for hiring/review applications handle AI-assisted content.

5. **Fine-tuning for fairness in evaluation:** Debiasing via fine-tuning is explored for general LLMs but not specifically for LLM evaluator/judge models.

## Recommendations for Experiment Design

**Recommended datasets:**
- **PERSUADE 2.0:** Primary dataset for essay scoring experiments with demographics (publicly available via Kaggle/GitHub, CC BY 4.0)
- **MT-Bench / Feedback Collection:** For LLM-as-judge evaluation experiments
- **BBQ / HolisticBias:** For measuring demographic bias

**Recommended baselines:**
- Vanilla LLM-as-judge (GPT-4o, Claude)
- Prometheus (open-source rubric-based evaluator)
- Position-swapped averaging (Wang et al. 2023)

**Recommended metrics:**
- QWK/Kappa for scoring agreement
- OSA/OSD/CSD for fairness (with 0.10 threshold)
- Robustness Rate for bias resistance
- ECE/CCEG for calibration fairness

**Methodological considerations:**
- Must test both with and without tokenized disclosures to measure penalty effect
- Should examine demographic interaction effects (Correctness x Demographics from Yang et al.)
- Calibration approaches: evidence-first prompting, position averaging, post-hoc score adjustment (CALIN-style), fine-tuning with fairness constraints
- Need to control for text quality when measuring disclosure penalty
