"""Statistical analysis and visualization of experiment results."""
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, PLOTS_DIR, SEED

np.random.seed(SEED)


def load_results(strategy):
    """Load results for a given calibration strategy."""
    filemap = {
        "none": "baseline_results.json",
        "fairness": "fairness_results.json",
        "evidence_first": "evidence_first_results.json",
        "blind": "blind_results.json",
    }
    path = os.path.join(RESULTS_DIR, filemap[strategy])
    with open(path) as f:
        return json.load(f)


def results_to_dataframe(results):
    """Convert results list to a pandas DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "sample_id": r["sample_id"],
            "condition": r["condition"],
            "disclosure": r["disclosure"],
            "demographic": r["demographic"],
            "calibration": r["calibration"],
            "ground_truth_score": r["ground_truth_score"],
            "mean_score": r["mean_score"],
            "run_scores": r["run_scores"],
            "n_valid_runs": r["n_valid_runs"],
        })
    return pd.DataFrame(rows)


def combine_all_results():
    """Load and combine all calibration strategy results into one DataFrame."""
    all_results = []
    for strategy in ["none", "fairness", "evidence_first", "blind"]:
        try:
            results = load_results(strategy)
            all_results.extend(results)
        except FileNotFoundError:
            print(f"Warning: results for '{strategy}' not found, skipping.")
    return results_to_dataframe(all_results)


# ─── Statistical Tests ────────────────────────────────────────────────

def compute_disclosure_penalty(df, calibration="none"):
    """Compute disclosure penalty: mean(disclosure) - mean(control), paired by sample_id."""
    sub = df[df["calibration"] == calibration].dropna(subset=["mean_score"])
    control = sub[sub["condition"] == "control"].set_index("sample_id")["mean_score"]
    disclosure = sub[sub["condition"] == "disclosure_only"].set_index("sample_id")["mean_score"]

    common = control.index.intersection(disclosure.index)
    ctrl = control[common].values
    disc = disclosure[common].values
    diff = disc - ctrl

    t_stat, p_val = stats.ttest_rel(disc, ctrl)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

    # Bootstrap CI
    boot_diffs = []
    rng = np.random.RandomState(SEED)
    for _ in range(1000):
        idx = rng.choice(len(diff), size=len(diff), replace=True)
        boot_diffs.append(np.mean(diff[idx]))
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "penalty_mean": np.mean(diff),
        "penalty_std": np.std(diff),
        "t_statistic": t_stat,
        "p_value": p_val,
        "cohens_d": cohens_d,
        "ci_95": (ci_lower, ci_upper),
        "n_pairs": len(common),
        "control_mean": np.mean(ctrl),
        "disclosure_mean": np.mean(disc),
    }


def compute_demographic_penalty(df, calibration="none"):
    """Compute demographic penalty: mean(demographic) - mean(control), paired."""
    sub = df[df["calibration"] == calibration].dropna(subset=["mean_score"])
    control = sub[sub["condition"] == "control"].set_index("sample_id")["mean_score"]
    demographic = sub[sub["condition"] == "demographic_only"].set_index("sample_id")["mean_score"]

    common = control.index.intersection(demographic.index)
    ctrl = control[common].values
    demo = demographic[common].values
    diff = demo - ctrl

    t_stat, p_val = stats.ttest_rel(demo, ctrl)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

    boot_diffs = []
    rng = np.random.RandomState(SEED)
    for _ in range(1000):
        idx = rng.choice(len(diff), size=len(diff), replace=True)
        boot_diffs.append(np.mean(diff[idx]))
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "penalty_mean": np.mean(diff),
        "penalty_std": np.std(diff),
        "t_statistic": t_stat,
        "p_value": p_val,
        "cohens_d": cohens_d,
        "ci_95": (ci_lower, ci_upper),
        "n_pairs": len(common),
        "control_mean": np.mean(ctrl),
        "demographic_mean": np.mean(demo),
    }


def compute_interaction_effect(df, calibration="none"):
    """2x2 ANOVA: Disclosure × Demographic interaction effect."""
    sub = df[df["calibration"] == calibration].dropna(subset=["mean_score"])

    # Pivot to get scores by sample_id and condition
    pivot = sub.pivot_table(values="mean_score", index="sample_id",
                            columns="condition", aggfunc="mean")

    common = pivot.dropna().index
    if len(common) < 10:
        return {"interaction_F": None, "interaction_p": None, "n": len(common),
                "note": "Too few complete cases"}

    # Two-way repeated measures: compute interaction as
    # (both - demographic) - (disclosure - control)
    ctrl = pivot.loc[common, "control"].values
    disc = pivot.loc[common, "disclosure_only"].values
    demo = pivot.loc[common, "demographic_only"].values
    both = pivot.loc[common, "both"].values

    # Interaction contrast
    interaction = (both - demo) - (disc - ctrl)
    t_stat, p_val = stats.ttest_1samp(interaction, 0)
    cohens_d = np.mean(interaction) / np.std(interaction, ddof=1) if np.std(interaction) > 0 else 0

    return {
        "interaction_mean": np.mean(interaction),
        "interaction_std": np.std(interaction),
        "t_statistic": t_stat,
        "p_value": p_val,
        "cohens_d": cohens_d,
        "n": len(common),
        "cell_means": {
            "control": np.mean(ctrl),
            "disclosure_only": np.mean(disc),
            "demographic_only": np.mean(demo),
            "both": np.mean(both),
        },
    }


def compute_score_reliability(df, calibration="none"):
    """Compute ICC (intra-class correlation) for inter-run reliability."""
    sub = df[df["calibration"] == calibration]
    valid_runs = []
    for _, row in sub.iterrows():
        scores = [s for s in row["run_scores"] if s is not None]
        if len(scores) >= 2:
            valid_runs.append(scores)

    if len(valid_runs) < 10:
        return {"icc": None, "mean_range": None, "note": "Too few multi-run items"}

    # Mean range as simple reliability metric
    ranges = [max(s) - min(s) for s in valid_runs]
    mean_range = np.mean(ranges)

    # ICC(1,1) computation
    k = min(len(s) for s in valid_runs)
    data = np.array([s[:k] for s in valid_runs])
    n = data.shape[0]
    grand_mean = data.mean()
    ss_between = k * np.sum((data.mean(axis=1) - grand_mean) ** 2)
    ss_within = np.sum((data - data.mean(axis=1, keepdims=True)) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within) if (ms_between + (k - 1) * ms_within) > 0 else 0

    return {
        "icc": icc,
        "mean_range": mean_range,
        "n_items": len(valid_runs),
        "n_runs": k,
    }


# ─── Visualization ────────────────────────────────────────────────────

def plot_disclosure_penalty_comparison(df):
    """Bar chart comparing disclosure penalty across calibration strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = ["none", "fairness", "evidence_first", "blind"]
    labels = ["Vanilla\n(No Calibration)", "Fairness-Aware\nPrompting",
              "Evidence-First\nPrompting", "Blind\n(Oracle)"]
    penalties = []
    ci_lowers = []
    ci_uppers = []

    for strat in strategies:
        result = compute_disclosure_penalty(df, strat)
        penalties.append(result["penalty_mean"])
        ci_lowers.append(result["ci_95"][0])
        ci_uppers.append(result["ci_95"][1])

    errors_lower = [p - cl for p, cl in zip(penalties, ci_lowers)]
    errors_upper = [cu - p for p, cu in zip(penalties, ci_uppers)]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]
    bars = ax.bar(labels, penalties, yerr=[errors_lower, errors_upper],
                  capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Score Difference\n(Disclosure - Control)", fontsize=12)
    ax.set_title("AI Disclosure Penalty Across Calibration Strategies", fontsize=14, fontweight="bold")
    ax.set_ylim(min(min(ci_lowers), -0.5) - 0.1, max(max(ci_uppers), 0.2) + 0.1)

    # Add value labels on bars
    for bar, pen in zip(bars, penalties):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{pen:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "disclosure_penalty_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: disclosure_penalty_comparison.png")


def plot_condition_means_heatmap(df):
    """Heatmap of mean scores by condition and calibration strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = ["none", "fairness", "evidence_first", "blind"]
    conditions = ["control", "disclosure_only", "demographic_only", "both"]
    cond_labels = ["Control", "AI Disclosure", "Non-Native\nSpeaker", "Both"]
    strat_labels = ["Vanilla", "Fairness-Aware", "Evidence-First", "Blind (Oracle)"]

    data = np.zeros((len(strategies), len(conditions)))
    for i, strat in enumerate(strategies):
        sub = df[(df["calibration"] == strat)].dropna(subset=["mean_score"])
        for j, cond in enumerate(conditions):
            scores = sub[sub["condition"] == cond]["mean_score"]
            data[i, j] = scores.mean() if len(scores) > 0 else np.nan

    sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                xticklabels=cond_labels, yticklabels=strat_labels,
                vmin=data.min() - 0.2, vmax=data.max() + 0.2,
                linewidths=0.5, linecolor="white")
    ax.set_title("Mean Evaluation Scores by Condition and Calibration Strategy",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Evaluation Condition", fontsize=11)
    ax.set_ylabel("Calibration Strategy", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "condition_means_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: condition_means_heatmap.png")


def plot_interaction_effects(df):
    """Interaction plot: Disclosure × Demographic for each calibration strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    strategies = ["none", "fairness", "evidence_first", "blind"]
    titles = ["Vanilla (No Calibration)", "Fairness-Aware Prompting",
              "Evidence-First Prompting", "Blind (Oracle)"]

    for idx, (strat, title) in enumerate(zip(strategies, titles)):
        ax = axes[idx // 2][idx % 2]
        result = compute_interaction_effect(df, strat)

        if result.get("cell_means"):
            means = result["cell_means"]
            # Plot lines for no-demographic and demographic conditions
            x = [0, 1]
            no_demo = [means["control"], means["disclosure_only"]]
            yes_demo = [means["demographic_only"], means["both"]]

            ax.plot(x, no_demo, "o-", color="#3498db", linewidth=2, markersize=8,
                    label="No Demographic Signal")
            ax.plot(x, yes_demo, "s--", color="#e74c3c", linewidth=2, markersize=8,
                    label="Non-Native Speaker")

            ax.set_xticks(x)
            ax.set_xticklabels(["No Disclosure", "AI Disclosure"])
            ax.set_ylabel("Mean Score")
            ax.set_title(f"{title}\n(interaction d={result.get('cohens_d', 0):.2f}, "
                        f"p={result.get('p_value', 1):.3f})", fontsize=10)
            ax.legend(fontsize=8)
            ax.set_ylim(1, 5)

    plt.suptitle("Disclosure × Demographic Interaction Effects", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "interaction_effects.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: interaction_effects.png")


def plot_score_distributions(df):
    """Violin plot of score distributions by condition for baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = df[(df["calibration"] == "none")].dropna(subset=["mean_score"])

    order = ["control", "disclosure_only", "demographic_only", "both"]
    labels = {"control": "Control", "disclosure_only": "AI Disclosure",
              "demographic_only": "Non-Native Speaker", "both": "Both"}
    sub = sub.copy()
    sub["condition_label"] = sub["condition"].map(labels)

    sns.violinplot(data=sub, x="condition_label", y="mean_score",
                   order=[labels[c] for c in order], ax=ax,
                   palette=["#3498db", "#e74c3c", "#f39c12", "#9b59b6"],
                   inner="box", cut=0)
    ax.set_ylabel("Evaluation Score", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("Score Distributions by Condition (Vanilla Evaluator)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 5.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "score_distributions_baseline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: score_distributions_baseline.png")


def plot_penalty_by_quality(df):
    """Show how disclosure penalty varies by ground truth score level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = df[df["calibration"] == "none"].dropna(subset=["mean_score"])

    gt_scores = sorted(sub["ground_truth_score"].unique())
    penalties = []
    ci_lows = []
    ci_highs = []

    for gt in gt_scores:
        gt_sub = sub[sub["ground_truth_score"] == gt]
        ctrl = gt_sub[gt_sub["condition"] == "control"].set_index("sample_id")["mean_score"]
        disc = gt_sub[gt_sub["condition"] == "disclosure_only"].set_index("sample_id")["mean_score"]
        common = ctrl.index.intersection(disc.index)
        if len(common) >= 3:
            diff = disc[common].values - ctrl[common].values
            penalties.append(np.mean(diff))
            # Bootstrap CI
            rng = np.random.RandomState(SEED)
            boot = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(500)]
            ci_lows.append(np.percentile(boot, 2.5))
            ci_highs.append(np.percentile(boot, 97.5))
        else:
            penalties.append(np.nan)
            ci_lows.append(np.nan)
            ci_highs.append(np.nan)

    ax.bar(gt_scores, penalties, color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.errorbar(gt_scores, penalties,
                yerr=[np.array(penalties) - np.array(ci_lows),
                      np.array(ci_highs) - np.array(penalties)],
                fmt="none", capsize=5, color="black")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Ground Truth Score", fontsize=12)
    ax.set_ylabel("Disclosure Penalty\n(Disclosure - Control)", fontsize=12)
    ax.set_title("AI Disclosure Penalty by Response Quality Level", fontsize=13, fontweight="bold")
    ax.set_xticks(gt_scores)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "penalty_by_quality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: penalty_by_quality.png")


# ─── Full Analysis Report ─────────────────────────────────────────────

def run_full_analysis():
    """Run complete analysis and print summary."""
    print("=" * 70)
    print("FULL STATISTICAL ANALYSIS")
    print("=" * 70)
    print()

    df = combine_all_results()
    print(f"Total result rows: {len(df)}")
    print(f"Calibration strategies: {df['calibration'].unique().tolist()}")
    print(f"Conditions: {df['condition'].unique().tolist()}")
    print(f"Unique samples: {df['sample_id'].nunique()}")
    print()

    # ── 1. Disclosure Penalty by Strategy ─────────────────────────
    print("1. DISCLOSURE PENALTY BY CALIBRATION STRATEGY")
    print("-" * 50)
    all_penalties = {}
    for strat in ["none", "fairness", "evidence_first", "blind"]:
        strat_df = df[df["calibration"] == strat]
        if len(strat_df) == 0:
            continue
        result = compute_disclosure_penalty(df, strat)
        all_penalties[strat] = result
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
        print(f"  {strat:20s}: penalty={result['penalty_mean']:+.3f} "
              f"(d={result['cohens_d']:.2f}), t={result['t_statistic']:.2f}, "
              f"p={result['p_value']:.4f} {sig}, "
              f"95% CI=[{result['ci_95'][0]:.3f}, {result['ci_95'][1]:.3f}]")
    print()

    # ── 2. Demographic Penalty by Strategy ────────────────────────
    print("2. DEMOGRAPHIC PENALTY BY CALIBRATION STRATEGY")
    print("-" * 50)
    all_demo_penalties = {}
    for strat in ["none", "fairness", "evidence_first", "blind"]:
        strat_df = df[df["calibration"] == strat]
        if len(strat_df) == 0:
            continue
        result = compute_demographic_penalty(df, strat)
        all_demo_penalties[strat] = result
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
        print(f"  {strat:20s}: penalty={result['penalty_mean']:+.3f} "
              f"(d={result['cohens_d']:.2f}), t={result['t_statistic']:.2f}, "
              f"p={result['p_value']:.4f} {sig}, "
              f"95% CI=[{result['ci_95'][0]:.3f}, {result['ci_95'][1]:.3f}]")
    print()

    # ── 3. Interaction Effects ────────────────────────────────────
    print("3. DISCLOSURE × DEMOGRAPHIC INTERACTION EFFECTS")
    print("-" * 50)
    all_interactions = {}
    for strat in ["none", "fairness", "evidence_first", "blind"]:
        strat_df = df[df["calibration"] == strat]
        if len(strat_df) == 0:
            continue
        result = compute_interaction_effect(df, strat)
        all_interactions[strat] = result
        if result.get("p_value") is not None:
            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
            print(f"  {strat:20s}: interaction={result['interaction_mean']:+.3f} "
                  f"(d={result['cohens_d']:.2f}), t={result['t_statistic']:.2f}, "
                  f"p={result['p_value']:.4f} {sig}")
            if result.get("cell_means"):
                cm = result["cell_means"]
                print(f"    Cell means: ctrl={cm['control']:.2f}, disc={cm['disclosure_only']:.2f}, "
                      f"demo={cm['demographic_only']:.2f}, both={cm['both']:.2f}")
        else:
            print(f"  {strat:20s}: {result.get('note', 'N/A')}")
    print()

    # ── 4. Reliability ────────────────────────────────────────────
    print("4. INTER-RUN RELIABILITY")
    print("-" * 50)
    for strat in ["none", "fairness", "evidence_first", "blind"]:
        rel = compute_score_reliability(df, strat)
        if rel.get("icc") is not None:
            print(f"  {strat:20s}: ICC={rel['icc']:.3f}, mean_range={rel['mean_range']:.2f}, "
                  f"n={rel['n_items']}, runs={rel['n_runs']}")
    print()

    # ── 5. Visualizations ─────────────────────────────────────────
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 50)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_disclosure_penalty_comparison(df)
    plot_condition_means_heatmap(df)
    plot_interaction_effects(df)
    plot_score_distributions(df)
    plot_penalty_by_quality(df)
    print()

    # ── 6. Save analysis summary ──────────────────────────────────
    summary = {
        "disclosure_penalties": {k: {kk: (vv if not isinstance(vv, tuple) else list(vv))
                                     for kk, vv in v.items()}
                                 for k, v in all_penalties.items()},
        "demographic_penalties": {k: {kk: (vv if not isinstance(vv, tuple) else list(vv))
                                      for kk, vv in v.items()}
                                  for k, v in all_demo_penalties.items()},
        "interaction_effects": {k: {kk: vv for kk, vv in v.items()}
                                for k, v in all_interactions.items()},
    }
    with open(os.path.join(RESULTS_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Saved: analysis_summary.json")

    return df, all_penalties, all_demo_penalties, all_interactions


if __name__ == "__main__":
    run_full_analysis()
