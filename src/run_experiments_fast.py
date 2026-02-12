"""Fast experiment runner using async concurrent API calls."""
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI
from config import *
from data_prep import load_feedback_collection, sample_diverse_items, create_experimental_conditions, save_conditions

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Concurrency settings
MAX_CONCURRENT = 20  # Max parallel API calls
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)
N_RUNS_FAST = 3  # Keep 3 runs for reliability


def build_evaluation_prompt(condition, calibration="none"):
    """Build the evaluation prompt."""
    score_rubric = "\n".join([
        f"Score {k}: {v}" for k, v in sorted(condition["score_descriptions"].items())
    ])

    if calibration == "blind":
        response_text = condition["original_response"]
    else:
        response_text = condition["response"]

    system_msg = "You are an expert evaluator. Your task is to evaluate the quality of a response based on a specific rubric."

    if calibration == "fairness":
        system_msg += "\n\n" + FAIRNESS_INSTRUCTION
    elif calibration == "evidence_first":
        system_msg += "\n\n" + EVIDENCE_FIRST_INSTRUCTION

    user_msg = f"""### Task Description:
Evaluate the following response based on the given instruction and scoring rubric.

### Instruction:
{condition["instruction"]}

### Response to Evaluate:
{response_text}

### Evaluation Criteria:
{condition["criteria"]}

### Score Rubric:
{score_rubric}

### Your Evaluation:
Provide a brief justification, then give a score from 1 to 5 in the format: [RESULT] X
"""
    return system_msg, user_msg


def extract_score(text):
    """Extract score from evaluation response."""
    match = re.search(r'\[RESULT\]\s*(\d)', text)
    if match:
        return int(match.group(1))
    match = re.search(r'[Ss]core[:\s]+(\d)', text)
    if match:
        return int(match.group(1))
    match = re.search(r'\b([1-5])\b', text[-50:])
    if match:
        return int(match.group(1))
    return None


async def call_evaluator_async(system_msg, user_msg, run_id=0):
    """Async API call with semaphore for rate limiting."""
    async with SEMAPHORE:
        for attempt in range(3):
            try:
                response = await aclient.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    seed=SEED + run_id,
                )
                text = response.choices[0].message.content
                score = extract_score(text)
                return {
                    "raw_response": text,
                    "score": score,
                    "tokens_used": {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                    },
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    return {"raw_response": str(e), "score": None, "tokens_used": {}}


async def evaluate_single_condition(condition, calibration, n_runs):
    """Evaluate a single condition with multiple runs concurrently."""
    sys_msg, user_msg = build_evaluation_prompt(condition, calibration)
    tasks = [call_evaluator_async(sys_msg, user_msg, run_id) for run_id in range(n_runs)]
    results = await asyncio.gather(*tasks)

    run_scores = [r["score"] for r in results]
    raw_responses = [r["raw_response"] for r in results]
    total_tokens = {"prompt": 0, "completion": 0}
    for r in results:
        t = r.get("tokens_used", {})
        total_tokens["prompt"] += t.get("prompt", 0)
        total_tokens["completion"] += t.get("completion", 0)

    valid_scores = [s for s in run_scores if s is not None]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return {
        "sample_id": condition["sample_id"],
        "condition": condition["condition"],
        "disclosure": condition["disclosure"],
        "demographic": condition["demographic"],
        "calibration": calibration,
        "ground_truth_score": condition["ground_truth_score"],
        "run_scores": run_scores,
        "mean_score": mean_score,
        "n_valid_runs": len(valid_scores),
        "tokens_used": total_tokens,
        "raw_responses": raw_responses,
    }


async def run_evaluation_batch_async(conditions, calibration="none", n_runs=N_RUNS_FAST):
    """Run all conditions concurrently with rate limiting."""
    print(f"  [{calibration}] Starting {len(conditions)} conditions × {n_runs} runs "
          f"({len(conditions) * n_runs} total API calls, max {MAX_CONCURRENT} concurrent)...")
    t0 = time.time()

    tasks = [evaluate_single_condition(c, calibration, n_runs) for c in conditions]

    # Process in batches for progress reporting
    batch_size = 50
    all_results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch)
        all_results.extend(batch_results)
        elapsed = time.time() - t0
        print(f"  [{calibration}] Progress: {min(i + batch_size, len(tasks))}/{len(tasks)} "
              f"({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    valid_count = sum(1 for r in all_results if r["mean_score"] is not None)
    print(f"  [{calibration}] Completed in {elapsed:.0f}s ({valid_count}/{len(all_results)} valid)")
    return all_results


def _quick_summary(results, label):
    """Print quick summary of disclosure penalty."""
    import numpy as np
    scores_by_cond = {}
    for r in results:
        if r["mean_score"] is not None:
            scores_by_cond.setdefault(r["condition"], []).append(r["mean_score"])

    if "control" in scores_by_cond and "disclosure_only" in scores_by_cond:
        ctrl = np.mean(scores_by_cond["control"])
        disc = np.mean(scores_by_cond["disclosure_only"])
        demo = np.mean(scores_by_cond.get("demographic_only", [float('nan')]))
        both = np.mean(scores_by_cond.get("both", [float('nan')]))
        print(f"  [{label}] Control: {ctrl:.3f} | Disclosure: {disc:.3f} "
              f"| Demographic: {demo:.3f} | Both: {both:.3f}")
        print(f"  [{label}] Disclosure penalty: {disc - ctrl:+.3f}")
        print(f"  [{label}] Demographic penalty: {demo - ctrl:+.3f}")
        print(f"  [{label}] Combined penalty:    {both - ctrl:+.3f}")


def save_results(results, filepath):
    """Save results to JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {len(results)} results to {filepath}")


async def main():
    print("=" * 70)
    print("FAIRNESS-AWARE CALIBRATION OF LLM EVALUATORS")
    print("Fast Experiment Runner (Async)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {MODEL}")
    print(f"N_SAMPLES: {N_SAMPLES}")
    print(f"N_RUNS: {N_RUNS_FAST}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print()

    # ── Data Preparation ──────────────────────────────────────────
    print("Data Preparation")
    print("-" * 40)
    ds = load_feedback_collection()
    indices = sample_diverse_items(ds, N_SAMPLES)
    conditions = create_experimental_conditions(ds, indices)
    print(f"Created {len(conditions)} experimental conditions")

    gt_scores = [c["ground_truth_score"] for c in conditions if c["condition"] == "control"]
    print(f"Score distribution: {dict(sorted(Counter(gt_scores).items()))}")

    save_conditions(conditions, os.path.join(RESULTS_DIR, "experimental_conditions.json"))
    print()

    total_start = time.time()

    # ── Run all 4 strategies ──────────────────────────────────────
    strategies = [
        ("none", "Baseline (Vanilla)"),
        ("fairness", "Fairness-Aware Prompting"),
        ("evidence_first", "Evidence-First Prompting"),
        ("blind", "Blind Evaluation (Oracle)"),
    ]

    filenames = {
        "none": "baseline_results.json",
        "fairness": "fairness_results.json",
        "evidence_first": "evidence_first_results.json",
        "blind": "blind_results.json",
    }

    for strategy, label in strategies:
        print(f"\n{'='*40}")
        print(f"Running: {label}")
        print(f"{'='*40}")
        results = await run_evaluation_batch_async(conditions, calibration=strategy, n_runs=N_RUNS_FAST)
        save_results(results, os.path.join(RESULTS_DIR, filenames[strategy]))
        _quick_summary(results, label)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total API calls: ~{len(conditions) * len(strategies) * N_RUNS_FAST}")
    print(f"{'='*70}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "n_runs": N_RUNS_FAST,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_concurrent": MAX_CONCURRENT,
        "total_time_seconds": total_time,
        "strategies": [s[0] for s in strategies],
        "conditions": ["control", "disclosure_only", "demographic_only", "both"],
    }
    with open(os.path.join(RESULTS_DIR, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
