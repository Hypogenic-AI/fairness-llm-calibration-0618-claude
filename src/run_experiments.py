"""Main experiment runner: executes all experimental conditions and calibration strategies."""
import json
import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_prep import load_feedback_collection, sample_diverse_items, create_experimental_conditions, save_conditions
from evaluator import run_evaluation_batch, save_results


def main():
    print("=" * 70)
    print("FAIRNESS-AWARE CALIBRATION OF LLM EVALUATORS")
    print("Experiment Runner")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {MODEL}")
    print(f"N_SAMPLES: {N_SAMPLES}")
    print(f"N_RUNS: {N_RUNS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Seed: {SEED}")
    print()

    # ── Step 1: Data Preparation ──────────────────────────────────
    print("Step 1: Data Preparation")
    print("-" * 40)
    ds = load_feedback_collection()
    print(f"Loaded {len(ds)} items from Feedback Collection")

    indices = sample_diverse_items(ds, N_SAMPLES)
    conditions = create_experimental_conditions(ds, indices)
    print(f"Created {len(conditions)} experimental conditions (2x2 × {N_SAMPLES} samples)")

    cond_path = os.path.join(RESULTS_DIR, "experimental_conditions.json")
    save_conditions(conditions, cond_path)

    # Score distribution check
    from collections import Counter
    gt_scores = [c["ground_truth_score"] for c in conditions if c["condition"] == "control"]
    print(f"Ground truth score distribution: {dict(sorted(Counter(gt_scores).items()))}")
    print()

    # ── Step 2: Baseline Evaluation (no calibration) ──────────────
    print("Step 2: Baseline Evaluation (vanilla, no calibration)")
    print("-" * 40)
    t0 = time.time()
    baseline_results = run_evaluation_batch(conditions, calibration="none", n_runs=N_RUNS)
    baseline_time = time.time() - t0
    save_results(baseline_results, os.path.join(RESULTS_DIR, "baseline_results.json"))
    print(f"Baseline completed in {baseline_time:.0f}s")
    print()

    # Quick check: disclosure penalty in baseline
    _quick_summary(baseline_results, "Baseline")
    print()

    # ── Step 3: Fairness-Aware Prompting ──────────────────────────
    print("Step 3: Fairness-Aware Prompting Calibration")
    print("-" * 40)
    t0 = time.time()
    fairness_results = run_evaluation_batch(conditions, calibration="fairness", n_runs=N_RUNS)
    fairness_time = time.time() - t0
    save_results(fairness_results, os.path.join(RESULTS_DIR, "fairness_results.json"))
    print(f"Fairness-aware completed in {fairness_time:.0f}s")
    _quick_summary(fairness_results, "Fairness-Aware")
    print()

    # ── Step 4: Evidence-First Prompting ──────────────────────────
    print("Step 4: Evidence-First Prompting Calibration")
    print("-" * 40)
    t0 = time.time()
    evidence_results = run_evaluation_batch(conditions, calibration="evidence_first", n_runs=N_RUNS)
    evidence_time = time.time() - t0
    save_results(evidence_results, os.path.join(RESULTS_DIR, "evidence_first_results.json"))
    print(f"Evidence-first completed in {evidence_time:.0f}s")
    _quick_summary(evidence_results, "Evidence-First")
    print()

    # ── Step 5: Blind Evaluation (oracle) ─────────────────────────
    print("Step 5: Blind Evaluation (oracle baseline)")
    print("-" * 40)
    t0 = time.time()
    blind_results = run_evaluation_batch(conditions, calibration="blind", n_runs=N_RUNS)
    blind_time = time.time() - t0
    save_results(blind_results, os.path.join(RESULTS_DIR, "blind_results.json"))
    print(f"Blind completed in {blind_time:.0f}s")
    _quick_summary(blind_results, "Blind")
    print()

    # ── Summary ───────────────────────────────────────────────────
    total_time = baseline_time + fairness_time + evidence_time + blind_time
    total_tokens = _count_tokens([baseline_results, fairness_results, evidence_results, blind_results])
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total API calls: ~{len(conditions) * 4 * N_RUNS}")
    print(f"Total tokens: {total_tokens:,}")
    print("=" * 70)

    # Save experiment metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "n_runs": N_RUNS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "total_time_seconds": total_time,
        "total_tokens": total_tokens,
        "calibration_strategies": ["none", "fairness", "evidence_first", "blind"],
        "conditions": ["control", "disclosure_only", "demographic_only", "both"],
    }
    with open(os.path.join(RESULTS_DIR, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def _quick_summary(results, label):
    """Print quick summary of disclosure penalty."""
    import numpy as np
    control_scores = []
    disclosure_scores = []
    demographic_scores = []
    both_scores = []

    for r in results:
        if r["mean_score"] is None:
            continue
        if r["condition"] == "control":
            control_scores.append(r["mean_score"])
        elif r["condition"] == "disclosure_only":
            disclosure_scores.append(r["mean_score"])
        elif r["condition"] == "demographic_only":
            demographic_scores.append(r["mean_score"])
        elif r["condition"] == "both":
            both_scores.append(r["mean_score"])

    if control_scores and disclosure_scores:
        ctrl_mean = np.mean(control_scores)
        disc_mean = np.mean(disclosure_scores)
        demo_mean = np.mean(demographic_scores) if demographic_scores else float('nan')
        both_mean = np.mean(both_scores) if both_scores else float('nan')

        print(f"  [{label}] Control: {ctrl_mean:.3f} | Disclosure: {disc_mean:.3f} "
              f"| Demographic: {demo_mean:.3f} | Both: {both_mean:.3f}")
        print(f"  [{label}] Disclosure penalty: {disc_mean - ctrl_mean:.3f}")
        print(f"  [{label}] Demographic penalty: {demo_mean - ctrl_mean:.3f}")
        print(f"  [{label}] Combined penalty:    {both_mean - ctrl_mean:.3f}")


def _count_tokens(results_lists):
    """Count total tokens across all results."""
    total = 0
    for results in results_lists:
        for r in results:
            t = r.get("tokens_used", {})
            total += t.get("prompt", 0) + t.get("completion", 0)
    return total


if __name__ == "__main__":
    main()
