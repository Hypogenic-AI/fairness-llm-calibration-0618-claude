"""Data preparation: sample from Feedback Collection and create experimental conditions."""
import json
import random
import numpy as np
from datasets import load_from_disk
from config import *


def load_feedback_collection():
    """Load the Feedback Collection dataset and return as list of dicts."""
    ds = load_from_disk(os.path.join(DATASET_DIR, "feedback_collection"))
    return ds


def sample_diverse_items(ds, n=N_SAMPLES):
    """Sample n items stratified by original score (1-5) for diversity."""
    random.seed(SEED)
    # Group by score
    by_score = {}
    for i, item in enumerate(ds):
        score = int(item["orig_score"])
        if score not in by_score:
            by_score[score] = []
        by_score[score].append(i)

    # Sample proportionally from each score level
    per_score = max(n // len(by_score), 4)
    sampled_indices = []
    for score in sorted(by_score.keys()):
        indices = by_score[score]
        k = min(per_score, len(indices))
        sampled_indices.extend(random.sample(indices, k))

    # If we need more, sample from remaining
    remaining = [i for i in range(len(ds)) if i not in set(sampled_indices)]
    if len(sampled_indices) < n:
        extra = random.sample(remaining, n - len(sampled_indices))
        sampled_indices.extend(extra)
    else:
        sampled_indices = sampled_indices[:n]

    random.shuffle(sampled_indices)
    return sampled_indices


def create_experimental_conditions(ds, indices):
    """Create 2x2 factorial conditions for each sampled item.

    Conditions:
    - control: no disclosure, no demographic
    - disclosure: AI disclosure added
    - demographic: non-native English speaker label
    - both: AI disclosure + non-native English speaker
    """
    conditions = []
    for idx in indices:
        item = ds[idx]
        base_instruction = item["orig_instruction"]
        base_response = item["orig_response"]
        base_criteria = item["orig_criteria"]
        base_score = int(item["orig_score"])
        score_descriptions = {
            1: item["orig_score1_description"],
            2: item["orig_score2_description"],
            3: item["orig_score3_description"],
            4: item["orig_score4_description"],
            5: item["orig_score5_description"],
        }

        # Create 4 conditions
        for disclosure in [False, True]:
            for demographic in [False, True]:
                prefix_parts = []
                if disclosure:
                    prefix_parts.append(DISCLOSURE_TEXT)
                if demographic:
                    prefix_parts.append(DEMOGRAPHIC_NONNATIVE)

                if prefix_parts:
                    modified_response = " ".join(prefix_parts) + "\n\n" + base_response
                else:
                    modified_response = base_response

                condition_name = _get_condition_name(disclosure, demographic)

                conditions.append({
                    "sample_id": idx,
                    "condition": condition_name,
                    "disclosure": disclosure,
                    "demographic": demographic,
                    "instruction": base_instruction,
                    "response": modified_response,
                    "original_response": base_response,
                    "criteria": base_criteria,
                    "ground_truth_score": base_score,
                    "score_descriptions": score_descriptions,
                })

    return conditions


def _get_condition_name(disclosure, demographic):
    if not disclosure and not demographic:
        return "control"
    elif disclosure and not demographic:
        return "disclosure_only"
    elif not disclosure and demographic:
        return "demographic_only"
    else:
        return "both"


def save_conditions(conditions, filepath):
    """Save conditions to JSON."""
    # Convert score_descriptions keys to strings for JSON
    for c in conditions:
        c["score_descriptions"] = {str(k): v for k, v in c["score_descriptions"].items()}
    with open(filepath, "w") as f:
        json.dump(conditions, f, indent=2)
    print(f"Saved {len(conditions)} conditions to {filepath}")


if __name__ == "__main__":
    print("Loading Feedback Collection dataset...")
    ds = load_feedback_collection()
    print(f"Total items: {len(ds)}")

    print(f"\nSampling {N_SAMPLES} diverse items...")
    indices = sample_diverse_items(ds, N_SAMPLES)

    # Check score distribution
    scores = [int(ds[i]["orig_score"]) for i in indices]
    from collections import Counter
    print(f"Score distribution: {dict(sorted(Counter(scores).items()))}")

    print(f"\nCreating experimental conditions (2x2 factorial)...")
    conditions = create_experimental_conditions(ds, indices)
    print(f"Total conditions: {len(conditions)}")

    outpath = os.path.join(RESULTS_DIR, "experimental_conditions.json")
    save_conditions(conditions, outpath)

    # Print a sample
    print("\n--- Sample condition (control) ---")
    sample = [c for c in conditions if c["condition"] == "control"][0]
    print(f"Criteria: {sample['criteria'][:100]}...")
    print(f"Response (first 200 chars): {sample['response'][:200]}...")
    print(f"Ground truth score: {sample['ground_truth_score']}")

    print("\n--- Sample condition (both) ---")
    sample_both = [c for c in conditions if c["sample_id"] == sample["sample_id"] and c["condition"] == "both"][0]
    print(f"Response (first 300 chars): {sample_both['response'][:300]}...")
