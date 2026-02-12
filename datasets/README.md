# Datasets

Datasets collected for the project "Fairness-Aware Calibration of LLM Evaluators Using Tokenized Disclosures."

## PERSUADE 2.0 Corpus

- **Directory:** `persuade_corpus_2.0/`
- **Source:** https://github.com/scrosseern/PERSUADE_corpus
- **Description:** 25,000+ argumentative essays from grades 6-12 with demographic metadata (gender, race, language background). Used in Yang et al. (2025) for studying demographic bias in LLM essay scoring.
- **License:** CC BY 4.0
- **Size:** ~364KB (repository metadata and rubrics only)
- **Note:** The full essay CSV files are hosted on Kaggle and are not included in the GitHub clone. To obtain the full dataset:
  1. Visit https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data
  2. Download `train.csv` and place it in `persuade_corpus_2.0/`
- **Key fields:** essay_id, full_text, score, gender, race, ell_status

## Prometheus Feedback Collection

- **Directory:** `feedback_collection/`
- **Source:** HuggingFace `prometheus-eval/Feedback-Collection`
- **Description:** 99,952 rubric-based evaluation examples for training open-source LLM evaluators. Each example includes an instruction, response, rubric, reference answer, and score with feedback.
- **License:** Apache 2.0
- **Size:** ~490MB (2 arrow files)
- **Download command:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("prometheus-eval/Feedback-Collection")
  ds.save_to_disk("datasets/feedback_collection")
  ```

## MT-Bench Human Judgments

- **Directory:** `mt_bench_human_judgments/`
- **Source:** HuggingFace `lmsys/mt_bench_human_judgments`
- **Description:** Human and GPT-4 pairwise judgments on MT-Bench. Contains two splits: `human` (3,355 examples) and `gpt4_pair` (2,400 examples).
- **License:** CC BY 4.0
- **Size:** ~3.0MB
- **Download command:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("lmsys/mt_bench_human_judgments")
  ds.save_to_disk("datasets/mt_bench_human_judgments")
  ```

## BBQ (Bias Benchmark for QA)

- **Directory:** `bbq/`
- **Source:** HuggingFace `lighteval/bbq_helm`
- **Description:** 1,000 bias benchmarking examples for question answering, covering demographic attributes including age, disability, gender, nationality, race, religion, and socioeconomic status.
- **License:** CC BY 4.0
- **Size:** ~306KB
- **Download command:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("lighteval/bbq_helm", "all")
  ds.save_to_disk("datasets/bbq")
  ```

## Data Management

Large data files are excluded from git via `.gitignore`. Only documentation and small sample files are tracked. To regenerate all datasets, run the download commands listed above with the `datasets` library installed:

```bash
pip install datasets
```
