# Research Plan: Fairness-Aware Calibration of LLM Evaluators Using Tokenized Disclosures

## Motivation & Novelty Assessment

### Why This Research Matters
LLM evaluators are increasingly used as judges in hiring, ranking, and review systems. If these evaluators penalize candidates who honestly disclose AI assistance, or if their scores interact with demographic signals (e.g., non-native English speakers being more penalized for AI disclosure than native speakers), this creates a systematic unfairness that discourages transparency and disproportionately harms underrepresented groups. Operationalizing fairness in LLM evaluators is critical for trust in AI-mediated high-stakes decisions.

### Gap in Existing Work
Based on the literature review:
1. **No work studies tokenized AI disclosures in evaluation**: While AI detection and watermarking are studied, nobody has measured whether explicit "AI-assisted" disclosures in evaluated text cause LLM judges to penalize the content.
2. **Demographic interaction effects in LLM evaluation are understudied**: Yang et al. (2025) showed demographic inference introduces scoring bias in essay scoring, but no work examines whether this interacts with AI disclosure.
3. **Fairness-aware calibration exists only for medical imaging**: CALIN (Shen et al., 2025) combined calibration + fairness but in a different domain. No analogous work exists for text evaluation.
4. **No calibration approach addresses disclosure penalties**: Existing calibration (position-swapping, evidence-first prompting) doesn't address the AI disclosure bias dimension.

### Our Novel Contribution
We are the **first to measure and mitigate the AI disclosure penalty** in LLM evaluators, and the **first to examine demographic × disclosure interaction effects**. We propose and test prompt-based calibration strategies that remove these biases, adapted from the CALIN bi-level framework to text evaluation.

### Experiment Justification
- **Experiment 1 (Disclosure Penalty Measurement)**: Establishes the baseline problem — do LLM evaluators penalize text with AI assistance disclosures? Without measuring this, we cannot know if calibration is needed.
- **Experiment 2 (Demographic Interaction Effects)**: Tests whether disclosure penalties interact with demographic signals (e.g., non-native English speaker + AI disclosure). This is critical for understanding whether the bias is uniform or discriminatory.
- **Experiment 3 (Calibration Strategies)**: Tests whether prompt-based calibration strategies can remove disclosure penalties and demographic interaction effects while maintaining evaluation quality.

## Research Question
Do LLM evaluators penalize text that includes tokenized AI assistance disclosures, and do these penalties interact with demographic signals? Can prompt-based calibration strategies remove these biases while maintaining evaluation accuracy?

## Hypothesis Decomposition
- **H1**: LLM evaluators assign lower scores to text that includes explicit AI assistance disclosures compared to identical text without disclosures.
- **H2**: The disclosure penalty interacts with demographic signals — non-native English speakers (or other underrepresented groups) are penalized more for AI disclosure than majority-group writers.
- **H3**: Prompt-based calibration strategies (fairness-aware instructions, evidence-first prompting, blind evaluation) can reduce or eliminate disclosure penalties and demographic interaction effects.

## Proposed Methodology

### Approach
We use a controlled experimental design with real LLM evaluators (GPT-4.1 via OpenAI API). We take text samples from the Feedback Collection dataset (which has rubric-based scoring), systematically inject AI disclosure statements and demographic signals, and measure how these modifications affect LLM evaluator scores. We then test calibration strategies to mitigate identified biases.

### Experimental Steps

#### Step 1: Dataset Construction
1. Sample 100 response texts from the Feedback Collection dataset (diverse quality levels, scores 1-5)
2. Create 4 versions of each text:
   - **Control**: Original text (no disclosure, no demographic signal)
   - **Disclosure**: Prepend "Note: This response was written with AI assistance."
   - **Demographic**: Prepend "Note: The author is a non-native English speaker."
   - **Both**: Prepend both disclosure and demographic statements
3. This gives us a 2×2 factorial design: Disclosure (yes/no) × Demographic Signal (yes/no)
4. Total: 400 evaluation instances

#### Step 2: Baseline Evaluation (Experiment 1 & 2)
1. Use GPT-4.1 as the LLM evaluator with the original Feedback Collection rubric
2. Evaluate all 400 instances with standardized prompts
3. Each instance evaluated 3 times (different random seeds / temperature) for reliability
4. Record scores, reasoning, and any mentions of disclosure/demographics
5. Total API calls: ~1,200

#### Step 3: Calibration Strategies (Experiment 3)
Test 3 calibration approaches on the same 400 instances:
- **Fairness-aware prompting**: Add explicit instruction: "Evaluate the content quality only. Do not consider whether AI was used or the author's background."
- **Evidence-first prompting**: Require detailed reasoning about specific quality criteria before giving a score (adapted from Wang et al. 2023)
- **Blind evaluation**: Strip disclosure/demographic statements before evaluation (oracle baseline — shows maximum possible improvement)
Total additional API calls: ~3,600 (3 strategies × 400 instances × 3 runs)

### Baselines
1. **Vanilla LLM evaluator**: GPT-4.1 with standard Feedback Collection rubric (no calibration)
2. **Blind evaluation**: Disclosure statements removed before evaluation (oracle upper bound)
3. **Evidence-first prompting**: From Wang et al. (2023) — forces reasoning before scoring

### Evaluation Metrics
1. **Disclosure Penalty**: Mean score difference between disclosure and no-disclosure conditions (paired)
2. **Demographic Interaction**: Interaction coefficient in 2×2 ANOVA (Disclosure × Demographic)
3. **Fairness metrics**:
   - Overall Score Accuracy (OSA): Agreement between calibrated and blind evaluation
   - Conditional Score Difference (CSD): Max score difference across subgroups
4. **Calibration quality**: Pearson correlation and QWK between calibrated scores and blind evaluation scores
5. **Statistical significance**: Paired t-tests with Bonferroni correction, Cohen's d for effect sizes

### Statistical Analysis Plan
- 2×2 repeated-measures ANOVA: Disclosure × Demographic Signal on evaluation scores
- Paired t-tests for pairwise comparisons with Bonferroni correction (α = 0.05/6 = 0.0083)
- Cohen's d for effect sizes (small: 0.2, medium: 0.5, large: 0.8)
- Bootstrap 95% confidence intervals for key metrics (1000 resamples)
- ICC (intra-class correlation) for inter-run reliability

## Expected Outcomes
- **H1 supported**: 0.3-0.5 point decrease in scores with AI disclosure (on 1-5 scale)
- **H2 supported**: Larger penalty for non-native English speaker + AI disclosure combination
- **H3 supported**: Fairness-aware prompting reduces disclosure penalty by 50-80%; evidence-first prompting partially reduces it
- If H1 is not supported (no disclosure penalty), this is also an important finding — suggesting current LLMs are already fair on this dimension

## Timeline and Milestones
1. Data preparation + dataset construction: 15 min
2. Baseline experiments (Exp 1 & 2): 30 min
3. Calibration experiments (Exp 3): 45 min
4. Analysis and visualization: 30 min
5. Documentation: 20 min

## Potential Challenges
1. **API rate limits**: Mitigate with exponential backoff and batching
2. **High variance in scores**: Mitigate with multiple runs per instance (3 runs)
3. **No disclosure penalty found**: Still valuable — document and analyze why (model alignment may already handle this)
4. **Cost**: ~4,800 API calls at ~$0.01 each = ~$48 (manageable)

## Success Criteria
1. Clear measurement of disclosure penalty magnitude with statistical significance
2. Clear measurement of demographic interaction effects
3. At least one calibration strategy that significantly reduces bias
4. Comprehensive statistical analysis with effect sizes and confidence intervals
5. Reproducible experimental pipeline
