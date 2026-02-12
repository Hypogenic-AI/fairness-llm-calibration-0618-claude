# Fairness-Aware Calibration of LLM Evaluators Using Tokenized Disclosures

## 1. Executive Summary

This study investigates whether LLM evaluators (GPT-4.1) penalize text that includes explicit AI assistance disclosures, and whether this penalty interacts with demographic signals. We found a **statistically significant AI disclosure penalty** of -0.100 points on a 1-5 scale (p=0.003, Cohen's d=-0.31) when GPT-4.1 evaluates responses prefixed with "This response was written with AI assistance." Surprisingly, adding a "non-native English speaker" label did not produce a significant independent penalty. We tested two prompt-based calibration strategies — fairness-aware prompting and evidence-first prompting — but neither substantially reduced the disclosure penalty at the per-sample paired level. Only blind evaluation (removing the disclosure text entirely) eliminated the bias, reducing the penalty by 83%. These findings suggest that the AI disclosure penalty is a subtle, implicit bias that resists surface-level prompt engineering and may require architectural or training-level interventions to address.

## 2. Goal

### Hypothesis
Fine-tuning LLM evaluators to avoid penalizing tokenized AI assistance disclosures and to remove demographic interaction effects will operationalize fairness by design and improve the fairness of algorithmic judgment in ranking, hiring, and review systems.

### Why This Is Important
As AI-assisted writing becomes ubiquitous, regulatory frameworks (EU AI Act, NYC Local Law 144) increasingly require transparency about AI use. If LLM evaluators — used in hiring, academic review, and content ranking — systematically penalize honest AI disclosures, this creates a perverse incentive against transparency and disproportionately harms those who comply with disclosure requirements.

### Research Questions
1. Do LLM evaluators penalize text that includes explicit AI assistance disclosures?
2. Does this penalty interact with demographic signals (non-native English speaker)?
3. Can prompt-based calibration strategies mitigate identified biases?

## 3. Data Construction

### Dataset Description
- **Source**: Feedback Collection dataset from Prometheus-eval (HuggingFace)
- **Size**: 99,952 rubric-based evaluation instances; we sampled 100 diverse items
- **Characteristics**: Each item includes an instruction, a response to evaluate, scoring rubric (1-5), score descriptions, and a ground truth score
- **Stratified Sampling**: 20 items per score level (1-5) for balanced representation

### Experimental Design
We used a 2×2 within-subjects factorial design:

| Condition | Disclosure | Demographic Signal |
|-----------|:----------:|:------------------:|
| Control | No | No |
| Disclosure Only | Yes | No |
| Demographic Only | No | Yes |
| Both | Yes | Yes |

**Disclosure text**: "Note: This response was written with AI assistance."
**Demographic text**: "Note: The author is a non-native English speaker."

These were prepended to the response text before evaluation. The underlying response content was identical across all 4 conditions for each sample.

### Example Samples

**Sample 1** (Ground truth score: 2, Criteria: compassion/comprehension):
- **Control**: "Well, breakups happen and life goes on. You're feeling a lot of emotions, but that's normal after ending a long-term relationship..."
- **Both**: "Note: This response was written with AI assistance. Note: The author is a non-native English speaker.\n\nWell, breakups happen and life goes on..."

### Data Quality
- All 100 samples had complete rubrics and valid ground truth scores
- Score distribution: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20} (perfectly balanced)
- 400 total evaluation instances (100 samples × 4 conditions)

## 4. Experiment Description

### Methodology

#### High-Level Approach
We evaluate GPT-4.1 as an LLM judge under controlled conditions where we systematically add AI disclosure and demographic signals to identical response texts. We then test whether prompt-based calibration strategies can mitigate identified biases. Each condition was evaluated 3 times (different random seeds) for reliability estimation.

#### Why This Method?
- **Controlled within-subjects design**: Each sample serves as its own control, isolating the effect of disclosure/demographic signals from response quality differences
- **Real LLM API calls**: All evaluations use GPT-4.1 (not simulated), providing ecologically valid results
- **Multiple calibration strategies**: Tests both the existence and malleability of identified biases

### Implementation Details

#### Tools and Libraries
- Python 3.12.8
- OpenAI API (gpt-4.1)
- NumPy 2.3.0, Pandas 2.3.0, SciPy 1.15.3
- Matplotlib 3.10.3, Seaborn 0.13.2
- HuggingFace Datasets 3.6.0

#### Calibration Strategies Tested

| Strategy | Description | Mechanism |
|----------|-------------|-----------|
| **Vanilla** (none) | Standard evaluation prompt | No calibration |
| **Fairness-aware** | Add explicit instruction: "Evaluate content quality ONLY. Do NOT consider AI usage or author background." | Explicit debiasing instruction |
| **Evidence-first** | Require detailed evidence for each rubric criterion before scoring | Forces content-focused reasoning (adapted from Wang et al. 2023) |
| **Blind** (oracle) | Strip disclosure/demographic text before evaluation | Information removal (upper bound) |

#### Hyperparameters

| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| Model | GPT-4.1 | Latest available |
| Temperature | 0.3 | Low for consistency, >0 for variance |
| Max tokens | 500 | Sufficient for evaluation |
| N runs | 3 | Balance reliability/cost |
| Seed | 42 | Reproducibility |
| Max concurrent | 20 | Rate limit management |

### Experimental Protocol

#### Reproducibility Information
- Number of runs per instance: 3
- Random seeds: 42, 43, 44
- Hardware: 4× NVIDIA RTX A6000 (GPU not used for evaluation — API-based)
- Total API calls: ~4,800
- Total execution time: 21 minutes
- Estimated cost: ~$30-50

#### Evaluation Metrics
1. **Disclosure Penalty**: Paired mean difference (disclosure - control) per sample
2. **Demographic Penalty**: Paired mean difference (demographic - control)
3. **Interaction Effect**: (both - demographic) - (disclosure - control)
4. **Statistical tests**: Paired t-test, Wilcoxon signed-rank, sign test (Bonferroni-corrected)
5. **Effect size**: Cohen's d with bootstrap 95% CIs
6. **Reliability**: ICC(1,1) across 3 runs

### Raw Results

#### Main Effects (All Calibration Strategies)

| Strategy | Control Mean | Disclosure Mean | Penalty | Cohen's d | p-value | 95% CI |
|----------|:-----------:|:--------------:|:-------:|:---------:|:-------:|:------:|
| Vanilla | 3.123 | 3.023 | **-0.100** | -0.31 | **0.003** | [-0.167, -0.040] |
| Fairness-aware | 3.137 | 3.040 | **-0.097** | -0.27 | **0.007** | [-0.163, -0.027] |
| Evidence-first | 2.946* | 2.940* | -0.096 | -0.19 | 0.094 | [-0.214, 0.015] |
| Blind (oracle) | 3.120 | 3.137 | +0.017 | 0.09 | 0.372 | [-0.017, 0.057] |

*Evidence-first had 68/400 items with unparseable scores due to response truncation.

#### Demographic Penalties

| Strategy | Control Mean | Demographic Mean | Penalty | p-value |
|----------|:-----------:|:----------------:|:-------:|:-------:|
| Vanilla | 3.123 | 3.107 | -0.017 | 0.594 |
| Fairness-aware | 3.137 | 3.150 | +0.013 | 0.712 |
| Evidence-first | 2.946 | 2.947 | +0.077 | 0.232 |
| Blind (oracle) | 3.120 | 3.083 | -0.037 | 0.124 |

#### Interaction Effects (Disclosure × Demographic)

| Strategy | Interaction | Cohen's d | p-value | Interpretation |
|----------|:----------:|:---------:|:-------:|---------------|
| Vanilla | +0.080 | 0.19 | 0.061 | Borderline: demographic label slightly buffers disclosure penalty |
| Fairness-aware | +0.063 | 0.13 | 0.190 | Not significant |
| Evidence-first | -0.043 | -0.07 | 0.586 | Not significant |
| Blind (oracle) | +0.003 | 0.01 | 0.905 | No interaction (expected) |

#### Disclosure Penalty by Quality Level (Vanilla)

| Ground Truth Score | Penalty | n | p-value |
|:-----------------:|:-------:|:-:|:-------:|
| 1 (lowest) | -0.100 | 20 | 0.083 |
| 2 | **-0.217** | 20 | **0.012** |
| 3 | -0.083 | 20 | 0.204 |
| 4 | +0.017 | 20 | 0.825 |
| 5 (highest) | -0.117 | 20 | 0.201 |

#### Inter-Run Reliability

| Strategy | ICC | Mean Score Range | n items |
|----------|:---:|:----------------:|:-------:|
| Vanilla | 0.950 | 0.27 | 400 |
| Fairness-aware | 0.950 | 0.27 | 400 |
| Evidence-first | 0.876 | 0.35 | 245 |
| Blind (oracle) | 0.949 | 0.26 | 400 |

#### Evaluation Reasoning Analysis

| Condition | Evaluator Explicitly Mentions Factor | Rate |
|-----------|--------------------------------------|:----:|
| AI Disclosure | Mentions "AI" or "assistance" | 8.0% |
| Non-Native Speaker | Mentions "language" or "background" | 46.0% |

### Visualizations

Key plots saved to `results/plots/`:
- `disclosure_penalty_comparison.png`: Bar chart of penalties across strategies with 95% CIs
- `condition_means_heatmap.png`: Heatmap of mean scores by condition × strategy
- `interaction_effects.png`: Disclosure × Demographic interaction plots
- `score_distributions_baseline.png`: Violin plots of score distributions
- `penalty_by_quality.png`: Penalty magnitude by ground truth quality level
- `comprehensive_results.png`: Three-panel summary figure
- `disclosure_penalty_distribution.png`: Histogram of per-sample penalty distribution
- `calibration_comparison_scatter.png`: Baseline vs. calibrated penalty scatter

## 5. Result Analysis

### Key Findings

**Finding 1: LLM evaluators exhibit a statistically significant AI disclosure penalty.**
When GPT-4.1 evaluates identical text with an AI assistance disclosure prepended, scores decrease by 0.100 points on average (1-5 scale). This is statistically significant (paired t-test: t=-3.06, p=0.003; Wilcoxon: p=0.002; sign test: p=0.0002) with a small-to-medium effect size (d=-0.31). 28% of samples received lower scores with disclosure, while only 6% received higher scores. The remaining 66% were unchanged, indicating the penalty affects a substantial minority of evaluations.

**Finding 2: The non-native English speaker label alone does not produce a significant scoring penalty.**
The demographic-only condition showed a negligible -0.017 penalty (p=0.594), suggesting GPT-4.1 does not overtly penalize text labeled as coming from non-native speakers. This is consistent with Schaller et al. (2024) who found GPT-4 was generally fair across demographics with explicit labels, and suggests the model has been aligned to avoid explicit demographic bias.

**Finding 3: Prompt-based calibration strategies are largely ineffective at removing disclosure bias.**
- **Fairness-aware prompting** (explicit "do not consider AI usage" instruction) reduced the penalty by only 3.3% (from -0.100 to -0.097, still significant at p=0.007). This suggests the model's bias operates below the level of explicit instruction following.
- **Evidence-first prompting** (forced reasoning before scoring) showed a numerically similar penalty (-0.096) among items with valid scores, with loss of statistical significance (p=0.094) likely driven by reduced sample size (332/400 valid) rather than genuine debiasing. The strategy also introduced practical problems: 17% of evaluations were truncated before producing a score.
- **Blind evaluation** (disclosure text removed) eliminated the penalty entirely (+0.017, p=0.37), confirming that the penalty is causally linked to the disclosure text, not a random artifact.

**Finding 4: The disclosure penalty is concentrated in lower-quality responses.**
Responses with ground truth score 2 (below average) showed the largest penalty (-0.217, p=0.012), while score 4 responses showed no penalty (+0.017). This suggests the evaluator is more harsh about AI disclosure when the underlying content quality is already questionable, potentially applying a "if you used AI and it's still mediocre" heuristic.

**Finding 5: The bias operates implicitly — the evaluator rarely explicitly references AI assistance.**
In only 8% of evaluations for disclosure conditions did the evaluator's reasoning explicitly mention AI or assistance. By contrast, language/background was mentioned in 46% of demographic conditions. This suggests the AI disclosure penalty operates as an implicit anchoring or framing effect, not through explicit reasoning about AI use.

**Finding 6: Borderline interaction between disclosure and demographic signals.**
The Disclosure × Demographic interaction in the vanilla condition was marginal (d=0.19, p=0.061). The combined condition (both disclosure + demographic) showed a penalty of -0.037, less than disclosure alone (-0.100). This counterintuitive result suggests that when both signals are present, the evaluator may apply a "sympathetic" adjustment — perhaps reasoning that a non-native speaker's use of AI is more understandable.

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| H1: Disclosure penalty exists | **Supported** | p=0.003, d=-0.31, 95% CI [-0.167, -0.040] |
| H2: Demographic interaction amplifies penalty | **Not Supported** | p=0.061 (marginal), direction opposite to predicted |
| H3: Calibration strategies reduce penalty | **Partially Supported** | Prompt-based strategies ineffective; blind evaluation eliminates penalty |

### Surprises and Insights

1. **Fairness-aware prompting failure**: The most direct approach — telling the model to ignore AI disclosures — had almost zero effect. This is a significant finding for the field: explicit debiasing instructions don't overcome implicit biases in evaluation.

2. **Demographic resilience**: GPT-4.1 appears well-calibrated against explicit demographic labels, suggesting alignment training has been effective for this dimension. However, this doesn't address implicit demographic inference from text (as shown by Yang et al., 2025).

3. **Quality-dependent penalty**: The disclosure penalty is strongest for below-average responses. This has practical implications: in hiring/review contexts, candidates with mediocre work who disclose AI use will be disproportionately penalized.

4. **Implicit bias mechanism**: The evaluator rarely mentions AI in its reasoning but still penalizes it, suggesting the disclosure acts as a negative priming/anchoring cue rather than a consciously applied criterion.

### Error Analysis

- **Evidence-first score extraction failures**: 17% of evidence-first evaluations didn't produce parseable scores. The strategy generates verbose evidence that exceeds the 500-token max_tokens limit. Increasing max_tokens would help but increases cost.
- **Score clustering at integers**: Most mean scores are exact integers (66% of baseline pairs had identical scores), indicating that the 3-run average doesn't fully smooth discrete scoring.
- **Ceiling/floor effects**: High-quality (score 5) and low-quality (score 1) responses show smaller penalties, likely due to ceiling/floor effects where the evaluator is more confident and less influenced by framing.

### Limitations

1. **Single model tested**: Only GPT-4.1 was evaluated. Other LLM evaluators (Claude, Gemini, open-source models) may behave differently.
2. **English-only**: All evaluation text is in English. Disclosure penalties may differ in other languages.
3. **Simple disclosure format**: We tested only one disclosure phrasing. Variations in wording, placement, or detail may produce different effects.
4. **No real demographics**: We used explicit labels rather than text with genuine non-native speaker characteristics. The interaction with actual linguistic features remains unstudied.
5. **Rubric-based evaluation only**: Results may differ for open-ended evaluation without rubrics.
6. **Prompt-based calibration only**: We tested only prompt engineering approaches, not fine-tuning or post-hoc score adjustment, which the original hypothesis centers on.
7. **Sample size**: While 100 samples with 3 runs provides adequate statistical power for the main effect (observed power ~0.86 for d=0.31), subgroup analyses (by quality level) have lower power.

## 6. Conclusions

### Summary
LLM evaluators (GPT-4.1) exhibit a statistically significant bias against text with AI assistance disclosures, penalizing such text by 0.1 points on a 1-5 scale (d=-0.31). This penalty is implicit — the evaluator rarely explicitly references AI use in its reasoning — and resists prompt-based calibration strategies. Only removing the disclosure text entirely eliminates the penalty, confirming its causal nature. These findings have direct implications for fairness in AI-mediated hiring, academic review, and ranking systems where AI disclosure may be required.

### Implications

**Practical**: Organizations using LLM evaluators should be aware that requiring AI disclosure in candidate materials may introduce systematic scoring bias. Blind evaluation (stripping disclosures before evaluation) is the most effective mitigation currently available.

**Theoretical**: The failure of explicit debiasing instructions ("ignore AI use") to reduce the penalty suggests that LLM evaluation biases operate at a deeper level than instruction-following, possibly reflecting training data biases about AI-generated content quality. This challenges the "prompt engineering as debiasing" approach.

**Policy**: As AI disclosure requirements become more common (EU AI Act, institutional policies), the interaction between mandatory disclosure and algorithmic evaluation creates a fairness gap that requires attention from both regulators and AI system designers.

### Confidence in Findings
- **High confidence** in H1 (disclosure penalty exists): Replicated across multiple statistical tests, consistent effect direction.
- **Moderate confidence** in calibration ineffectiveness: Two strategies tested; more sophisticated approaches (fine-tuning, RLHF) remain untested.
- **Lower confidence** in interaction effects: Marginal significance, small effect size, interpretation uncertain.

## 7. Next Steps

### Immediate Follow-ups
1. **Test additional models**: Evaluate Claude, Gemini, and open-source evaluators (Prometheus 2, Llama-based judges) to assess generalizability.
2. **Fine-tuning approach**: Actually fine-tune an evaluator model on balanced data (with/without disclosures) to train explicit fairness, as originally hypothesized.
3. **Post-hoc score adjustment**: Adapt CALIN's bi-level calibration (Shen et al., 2025) to text evaluation: (a) calibrate population-level scores, (b) equalize across disclosure/non-disclosure subgroups.
4. **Disclosure wording variations**: Test different phrasings ("AI-assisted", "co-authored with AI", "proofread by AI") to map the disclosure penalty landscape.

### Alternative Approaches
- **Contrastive fine-tuning**: Train on triplets (text, text+disclosure, text+demographic) to learn disclosure-invariant representations.
- **Score normalization**: Collect disclosure-condition baseline scores and subtract estimated bias post-hoc.
- **Two-stage evaluation**: First evaluate quality, then separately assess disclosure — preventing cross-contamination.

### Broader Extensions
- **Real hiring context**: Test with actual job application materials and industry rubrics.
- **Cross-lingual study**: Evaluate disclosure penalties in non-English contexts.
- **Longitudinal tracking**: Monitor how disclosure bias evolves as models are updated.

### Open Questions
1. Why does fairness-aware prompting fail? Is this a fundamental limitation of instruction-following or specific to this model?
2. Does the disclosure penalty reflect genuine quality differences in AI-assisted text, or purely anchoring bias?
3. Would users who disclose AI use be better served by structured disclosure (metadata) rather than in-text statements?

## 8. References

1. Wang et al. (2023). "Large Language Models are not Fair Evaluators." arXiv:2305.17926, ACL 2024.
2. Ye et al. (2024). "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge." arXiv:2410.02736.
3. Yang et al. (2025). "Does the Prompt-based LLM Recognize Students' Demographics?" arXiv:2504.21330.
4. Shen et al. (2025). "Exposing and Mitigating Calibration Biases and Demographic Unfairness in MLLM Few-Shot ICL." arXiv:2506.23298.
5. Schaller et al. (2024). "Fairness in Automated Essay Scoring." ACL BEA Workshop 2024.
6. Zheng et al. (2024). "A Survey on LLM-as-a-Judge." arXiv:2411.15594.
7. Gallegos et al. (2024). "Bias and Fairness in LLMs: A Survey." Computational Linguistics, MIT Press.
8. Kadavath et al. (2023). "Just Ask for Calibration." arXiv:2305.14975.
