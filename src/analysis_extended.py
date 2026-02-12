"""Extended analysis with additional statistical tests and visualizations."""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, PLOTS_DIR, SEED
from analysis import combine_all_results, compute_disclosure_penalty, compute_demographic_penalty, compute_interaction_effect

np.random.seed(SEED)


def run_extended_analysis():
    df = combine_all_results()
    print(f"Loaded {len(df)} result rows")

    # ── 1. Paired analysis: how often does disclosure LOWER the score? ──
    print("\n" + "="*60)
    print("DETAILED DISCLOSURE PENALTY ANALYSIS (Baseline)")
    print("="*60)

    baseline = df[df["calibration"] == "none"].dropna(subset=["mean_score"])
    ctrl = baseline[baseline["condition"] == "control"].set_index("sample_id")["mean_score"]
    disc = baseline[baseline["condition"] == "disclosure_only"].set_index("sample_id")["mean_score"]
    common = ctrl.index.intersection(disc.index)

    diff = disc[common].values - ctrl[common].values
    n_lower = (diff < 0).sum()
    n_same = (diff == 0).sum()
    n_higher = (diff > 0).sum()
    print(f"Disclosure LOWERS score: {n_lower}/{len(diff)} ({100*n_lower/len(diff):.1f}%)")
    print(f"Disclosure SAME score:   {n_same}/{len(diff)} ({100*n_same/len(diff):.1f}%)")
    print(f"Disclosure HIGHER score: {n_higher}/{len(diff)} ({100*n_higher/len(diff):.1f}%)")
    print(f"Mean penalty: {np.mean(diff):.3f} (negative = penalty)")
    print(f"Median penalty: {np.median(diff):.3f}")

    # Sign test
    n_nonzero = n_lower + n_higher
    if n_nonzero > 0:
        sign_p = stats.binomtest(n_lower, n_nonzero, 0.5).pvalue
        print(f"Sign test p-value: {sign_p:.4f}")

    # Wilcoxon signed-rank test (non-parametric alternative)
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) > 10:
        w_stat, w_p = stats.wilcoxon(nonzero_diff)
        print(f"Wilcoxon signed-rank test: W={w_stat:.1f}, p={w_p:.4f}")

    # ── 2. Penalty by ground truth quality level ──
    print("\n" + "="*60)
    print("DISCLOSURE PENALTY BY QUALITY LEVEL (Baseline)")
    print("="*60)

    for gt in sorted(baseline["ground_truth_score"].unique()):
        gt_sub = baseline[baseline["ground_truth_score"] == gt]
        c = gt_sub[gt_sub["condition"] == "control"].set_index("sample_id")["mean_score"]
        d = gt_sub[gt_sub["condition"] == "disclosure_only"].set_index("sample_id")["mean_score"]
        common_gt = c.index.intersection(d.index)
        if len(common_gt) >= 3:
            diffs = d[common_gt].values - c[common_gt].values
            t, p = stats.ttest_rel(d[common_gt].values, c[common_gt].values) if len(common_gt) >= 5 else (0, 1)
            print(f"  Score {gt}: penalty={np.mean(diffs):+.3f} (n={len(common_gt)}, p={p:.4f})")

    # ── 3. Calibration effectiveness comparison ──
    print("\n" + "="*60)
    print("CALIBRATION EFFECTIVENESS")
    print("="*60)

    baseline_penalty = compute_disclosure_penalty(df, "none")["penalty_mean"]
    for strat in ["fairness", "evidence_first", "blind"]:
        strat_penalty = compute_disclosure_penalty(df, strat)["penalty_mean"]
        reduction = abs(baseline_penalty) - abs(strat_penalty)
        pct_reduction = (reduction / abs(baseline_penalty)) * 100 if baseline_penalty != 0 else 0
        print(f"  {strat:20s}: penalty={strat_penalty:+.3f}, "
              f"reduction={pct_reduction:.1f}% from baseline ({baseline_penalty:+.3f})")

    # ── 4. Create combined summary figure ──
    print("\n" + "="*60)
    print("GENERATING EXTENDED VISUALIZATIONS")
    print("="*60)

    # Figure 1: Comprehensive penalty comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Score distributions by condition (baseline)
    ax = axes[0]
    conditions_order = ["control", "disclosure_only", "demographic_only", "both"]
    colors_map = {"control": "#3498db", "disclosure_only": "#e74c3c",
                  "demographic_only": "#f39c12", "both": "#9b59b6"}
    labels_map = {"control": "Control", "disclosure_only": "AI Disclosure",
                  "demographic_only": "Non-Native", "both": "Both"}

    for cond in conditions_order:
        data = baseline[baseline["condition"] == cond]["mean_score"].dropna()
        ax.hist(data, bins=np.arange(0.5, 6.0, 0.5), alpha=0.4,
                label=labels_map[cond], color=colors_map[cond], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Mean Evaluation Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("(A) Score Distributions by Condition\n(Vanilla Evaluator)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel B: Penalties across strategies
    ax = axes[1]
    strategies = ["none", "fairness", "evidence_first", "blind"]
    strat_labels = ["Vanilla", "Fairness\nAware", "Evidence\nFirst", "Blind\n(Oracle)"]
    disc_penalties = []
    demo_penalties = []
    for strat in strategies:
        dp = compute_disclosure_penalty(df, strat)
        dmp = compute_demographic_penalty(df, strat)
        disc_penalties.append(dp["penalty_mean"])
        demo_penalties.append(dmp["penalty_mean"])

    x = np.arange(len(strategies))
    width = 0.35
    bars1 = ax.bar(x - width/2, disc_penalties, width, label="AI Disclosure", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + width/2, demo_penalties, width, label="Non-Native Speaker", color="#f39c12", alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels)
    ax.set_ylabel("Score Penalty (vs. Control)", fontsize=11)
    ax.set_title("(B) Penalties by Calibration Strategy", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: Interaction effects
    ax = axes[2]
    for i, (strat, label) in enumerate(zip(strategies, strat_labels)):
        result = compute_interaction_effect(df, strat)
        if result.get("cell_means"):
            means = result["cell_means"]
            # Plot interaction strength
            interaction_val = result["interaction_mean"]
            color = "#e74c3c" if abs(interaction_val) > 0.05 else "#3498db"
            ax.barh(i, interaction_val, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.text(interaction_val + 0.005 if interaction_val >= 0 else interaction_val - 0.005,
                    i, f"{interaction_val:+.3f}", va="center",
                    ha="left" if interaction_val >= 0 else "right", fontsize=10)

    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strat_labels)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Interaction Effect\n(Disclosure × Demographic)", fontsize=11)
    ax.set_title("(C) Interaction Effects", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "comprehensive_results.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: comprehensive_results.png")

    # Figure 2: Paired difference histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diff, bins=np.arange(-2.5, 2.6, 0.33), color="#e74c3c", alpha=0.7,
            edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=1, linestyle="--")
    ax.axvline(x=np.mean(diff), color="#e74c3c", linewidth=2, linestyle="-",
               label=f"Mean = {np.mean(diff):.3f}")
    ax.set_xlabel("Score Difference (Disclosure - Control)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Per-Sample AI Disclosure Penalty\n(Vanilla Evaluator)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "disclosure_penalty_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: disclosure_penalty_distribution.png")

    # Figure 3: Before vs After calibration (matched pairs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (strat, label) in enumerate([("evidence_first", "Evidence-First"), ("fairness", "Fairness-Aware")]):
        ax = axes[idx]
        strat_df = df[df["calibration"] == strat].dropna(subset=["mean_score"])
        base_df = df[df["calibration"] == "none"].dropna(subset=["mean_score"])

        # Compute per-sample penalty for baseline and strategy
        base_ctrl = base_df[base_df["condition"] == "control"].set_index("sample_id")["mean_score"]
        base_disc = base_df[base_df["condition"] == "disclosure_only"].set_index("sample_id")["mean_score"]
        strat_ctrl = strat_df[strat_df["condition"] == "control"].set_index("sample_id")["mean_score"]
        strat_disc = strat_df[strat_df["condition"] == "disclosure_only"].set_index("sample_id")["mean_score"]

        common_all = base_ctrl.index.intersection(base_disc.index).intersection(strat_ctrl.index).intersection(strat_disc.index)
        base_penalty = base_disc[common_all].values - base_ctrl[common_all].values
        strat_penalty = strat_disc[common_all].values - strat_ctrl[common_all].values

        ax.scatter(base_penalty, strat_penalty, alpha=0.5, s=30, c="#3498db")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
        ax.plot([-2, 2], [-2, 2], color="#e74c3c", linewidth=1, linestyle="--", alpha=0.5,
                label="Identity line")
        ax.set_xlabel("Baseline Penalty (per sample)", fontsize=11)
        ax.set_ylabel(f"{label} Penalty (per sample)", fontsize=11)
        ax.set_title(f"Penalty: Baseline vs. {label} (n={len(common_all)})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "calibration_comparison_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: calibration_comparison_scatter.png")

    # ── 5. Detailed content analysis of evaluation reasoning ──
    print("\n" + "="*60)
    print("EVALUATION REASONING ANALYSIS")
    print("="*60)

    with open(os.path.join(RESULTS_DIR, "baseline_results.json")) as f:
        baseline_raw = json.load(f)

    # Check how often the evaluator MENTIONS disclosure in its reasoning
    disclosure_mentioned = 0
    total_disc_items = 0
    for r in baseline_raw:
        if r["disclosure"]:
            total_disc_items += 1
            for resp in r["raw_responses"]:
                if resp and ("AI" in resp or "assistance" in resp.lower() or "disclosure" in resp.lower()):
                    disclosure_mentioned += 1
                    break

    print(f"In disclosure conditions, evaluator mentions AI/assistance in {disclosure_mentioned}/{total_disc_items} "
          f"({100*disclosure_mentioned/max(total_disc_items,1):.1f}%) of evaluations")

    # Check demographic mentions
    demo_mentioned = 0
    total_demo_items = 0
    for r in baseline_raw:
        if r["demographic"]:
            total_demo_items += 1
            for resp in r["raw_responses"]:
                if resp and ("non-native" in resp.lower() or "language" in resp.lower() or "background" in resp.lower()):
                    demo_mentioned += 1
                    break

    print(f"In demographic conditions, evaluator mentions language/background in {demo_mentioned}/{total_demo_items} "
          f"({100*demo_mentioned/max(total_demo_items,1):.1f}%) of evaluations")

    print("\nDone!")


if __name__ == "__main__":
    run_extended_analysis()
