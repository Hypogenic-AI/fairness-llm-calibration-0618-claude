"""LLM evaluator: calls GPT-4.1 to score responses under different calibration conditions."""
import json
import os
import re
import time
from openai import OpenAI
from config import *


client = OpenAI(api_key=OPENAI_API_KEY)


def build_evaluation_prompt(condition, calibration="none"):
    """Build the evaluation prompt for a given condition and calibration strategy.

    calibration options:
    - "none": vanilla evaluation
    - "fairness": add fairness-aware instruction
    - "evidence_first": require evidence before scoring
    - "blind": strip disclosure/demographic from response (oracle)
    """
    score_rubric = "\n".join([
        f"Score {k}: {v}" for k, v in sorted(condition["score_descriptions"].items())
    ])

    # Determine response text based on calibration
    if calibration == "blind":
        response_text = condition["original_response"]
    else:
        response_text = condition["response"]

    # Build system message
    system_msg = "You are an expert evaluator. Your task is to evaluate the quality of a response based on a specific rubric."

    if calibration == "fairness":
        system_msg += "\n\n" + FAIRNESS_INSTRUCTION
    elif calibration == "evidence_first":
        system_msg += "\n\n" + EVIDENCE_FIRST_INSTRUCTION

    # Build user message
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


def call_evaluator(system_msg, user_msg, run_id=0):
    """Call the LLM evaluator and extract the score."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
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
                wait = 2 ** (attempt + 1)
                print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error (final attempt): {e}")
                return {"raw_response": str(e), "score": None, "tokens_used": {}}


def extract_score(text):
    """Extract score from [RESULT] X format."""
    # Look for [RESULT] followed by a number
    match = re.search(r'\[RESULT\]\s*(\d)', text)
    if match:
        return int(match.group(1))
    # Fallback: look for "Score: X" or "score of X"
    match = re.search(r'[Ss]core[:\s]+(\d)', text)
    if match:
        return int(match.group(1))
    # Last resort: find any standalone digit 1-5
    match = re.search(r'\b([1-5])\b', text[-50:])
    if match:
        return int(match.group(1))
    return None


def run_evaluation_batch(conditions, calibration="none", n_runs=N_RUNS):
    """Run evaluation for a batch of conditions with multiple runs.

    Returns list of result dicts with condition info + evaluation results.
    """
    results = []
    total = len(conditions)
    start_time = time.time()

    for i, cond in enumerate(conditions):
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 1)
            print(f"  [{calibration}] Evaluating {i+1}/{total} "
                  f"(condition={cond['condition']}, sample={cond['sample_id']}) "
                  f"[{elapsed:.0f}s elapsed, {rate:.1f} items/s]")

        run_scores = []
        run_responses = []
        total_tokens = {"prompt": 0, "completion": 0}

        for run_id in range(n_runs):
            sys_msg, user_msg = build_evaluation_prompt(cond, calibration)
            result = call_evaluator(sys_msg, user_msg, run_id)
            run_scores.append(result["score"])
            run_responses.append(result["raw_response"])
            if result["tokens_used"]:
                total_tokens["prompt"] += result["tokens_used"].get("prompt", 0)
                total_tokens["completion"] += result["tokens_used"].get("completion", 0)

        # Compute mean score (ignoring None)
        valid_scores = [s for s in run_scores if s is not None]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        results.append({
            "sample_id": cond["sample_id"],
            "condition": cond["condition"],
            "disclosure": cond["disclosure"],
            "demographic": cond["demographic"],
            "calibration": calibration,
            "ground_truth_score": cond["ground_truth_score"],
            "run_scores": run_scores,
            "mean_score": mean_score,
            "n_valid_runs": len(valid_scores),
            "tokens_used": total_tokens,
            "raw_responses": run_responses,
        })

    elapsed = time.time() - start_time
    valid_count = sum(1 for r in results if r["mean_score"] is not None)
    print(f"  [{calibration}] Completed {total} conditions in {elapsed:.0f}s "
          f"({valid_count}/{total} with valid scores)")

    return results


def save_results(results, filepath):
    """Save results to JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} results to {filepath}")
