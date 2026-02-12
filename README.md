# Fairness-Aware Calibration of LLM Evaluators Using Tokenized Disclosures

## Overview

This project investigates whether LLM evaluators penalize text that includes AI assistance disclosures, and tests calibration strategies to mitigate identified biases. We use GPT-4.1 to evaluate 100 response samples from the Feedback Collection dataset under 4 conditions (control, AI disclosure, non-native speaker label, both) and 4 calibration strategies (vanilla, fairness-aware, evidence-first, blind).

## Key Findings

- **AI disclosure penalty is real and significant**: GPT-4.1 penalizes disclosed AI-assisted text by -0.100 points (1-5 scale, p=0.003, d=-0.31)
- **Non-native speaker label alone shows no penalty**: -0.017 (p=0.59), suggesting GPT-4.1 is well-aligned against explicit demographic bias
- **Prompt-based calibration largely fails**: Fairness-aware prompting (3.3% reduction) and evidence-first prompting (3.8% reduction) are ineffective
- **Blind evaluation works**: Removing disclosure text eliminates the penalty (83% reduction), confirming causal link
- **Penalty is implicit**: Evaluator mentions AI in only 8% of reasoning, yet still penalizes — suggesting anchoring/priming bias
- **Quality-dependent**: Penalty strongest for below-average responses (score 2: -0.217, p=0.012)

## How to Reproduce

### Environment Setup
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install pyarrow numpy pandas matplotlib scipy scikit-learn openai datasets tqdm seaborn statsmodels httpx
```

### Running Experiments
```bash
# Ensure OPENAI_API_KEY is set
export OPENAI_API_KEY=your_key_here

# Run all experiments (~21 minutes, ~4800 API calls, ~$30-50)
cd src
python run_experiments_fast.py

# Run analysis
python analysis.py
python analysis_extended.py
```

### File Structure
```
.
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan
├── literature_review.md         # Literature review
├── resources.md                 # Resource catalog
├── src/
│   ├── config.py                # Configuration and constants
│   ├── data_prep.py             # Dataset sampling and condition creation
│   ├── evaluator.py             # LLM evaluator interface (sync)
│   ├── run_experiments_fast.py  # Async experiment runner (primary)
│   ├── run_experiments.py       # Sync experiment runner (backup)
│   ├── analysis.py              # Core statistical analysis
│   └── analysis_extended.py     # Extended analysis and extra visualizations
├── results/
│   ├── experimental_conditions.json
│   ├── baseline_results.json
│   ├── fairness_results.json
│   ├── evidence_first_results.json
│   ├── blind_results.json
│   ├── experiment_metadata.json
│   ├── analysis_summary.json
│   └── plots/                   # Visualizations (8 PNG files)
├── datasets/                    # Pre-downloaded datasets
├── papers/                      # Reference papers (PDFs)
└── code/                        # Baseline code repositories
```

See [REPORT.md](REPORT.md) for the full research report.
