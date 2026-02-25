#!/usr/bin/env python3
"""
Phase 3: Statistical Analysis & Visualization

Reads Phase 2 results and produces:
  - Learning curves (Force MAE vs DFT budget)
  - Paired t-tests for each budget
  - Cohen's d effect sizes
  - Energy parity plots
  - Summary tables
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

DFT_BUDGETS = [50, 100, 200, 500]
DATA_SEEDS = [42, 123, 7]


def load_results():
    path = RESULTS_DIR / "phase2_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase 2 results not found at {path}. Run phase2_experiment.py first.")
    return json.loads(path.read_text())


def extract_metric(results, method, budgets, seeds, metric="force_mae_meV_A"):
    """Extract metric values grouped by budget."""
    by_budget = {}
    for N in budgets:
        vals = []
        for s in seeds:
            key = f"{method}_N{N}_s{s}"
            if key in results:
                vals.append(results[key][metric])
        by_budget[N] = vals
    return by_budget


def statistical_tests(direct_by_budget, delta_by_budget, budgets):
    """Paired t-tests and effect sizes at each budget."""
    stats = {}
    for N in budgets:
        d_vals = direct_by_budget.get(N, [])
        delta_vals = delta_by_budget.get(N, [])

        if len(d_vals) < 2 or len(delta_vals) < 2:
            stats[N] = {"note": "insufficient data"}
            continue

        d_arr = np.array(d_vals)
        delta_arr = np.array(delta_vals)

        if len(d_arr) == len(delta_arr) and len(d_arr) >= 2:
            t_stat, p_value = ttest_rel(d_arr, delta_arr)
        else:
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(d_arr, delta_arr)

        pooled_std = np.sqrt((np.std(d_arr)**2 + np.std(delta_arr)**2) / 2)
        cohens_d = (np.mean(d_arr) - np.mean(delta_arr)) / pooled_std if pooled_std > 0 else 0

        improvement = np.mean(d_arr) / np.mean(delta_arr) if np.mean(delta_arr) > 0 else float("inf")

        if abs(cohens_d) < 0.2:
            effect_label = "NEGLIGIBLE"
        elif abs(cohens_d) < 0.5:
            effect_label = "SMALL"
        elif abs(cohens_d) < 0.8:
            effect_label = "MEDIUM"
        else:
            effect_label = "LARGE"

        stats[N] = {
            "direct_mean": float(np.mean(d_arr)),
            "direct_std": float(np.std(d_arr)),
            "delta_mean": float(np.mean(delta_arr)),
            "delta_std": float(np.std(delta_arr)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "effect_label": effect_label,
            "significant": bool(p_value < 0.05),
            "improvement_ratio": float(improvement),
        }

    return stats


def plot_learning_curves(results, budgets, seeds):
    """Learning curves: Force MAE and Energy MAE vs DFT budget."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for metric, ax, ylabel in [
        ("force_mae_meV_A", axes[0], "Force MAE (meV/Å)"),
        ("energy_mae_meV_atom", axes[1], "Energy MAE (meV/atom)"),
    ]:
        for method, label, color, marker in [
            ("direct", "Direct MACE", "#2196F3", "o"),
            ("delta", "Δ-MACE", "#F44336", "s"),
        ]:
            by_budget = extract_metric(results, method, budgets, seeds, metric)
            means = [np.mean(by_budget.get(N, [0])) for N in budgets]
            stds = [np.std(by_budget.get(N, [0])) for N in budgets]
            valid_budgets = [N for N in budgets if len(by_budget.get(N, [])) > 0]
            valid_means = [np.mean(by_budget[N]) for N in valid_budgets]
            valid_stds = [np.std(by_budget[N]) for N in valid_budgets]

            if valid_budgets:
                ax.errorbar(valid_budgets, valid_means, yerr=valid_stds,
                           label=label, color=color, marker=marker,
                           capsize=5, linewidth=2, markersize=8)

        if "direct_full" in results:
            full_val = results["direct_full"][metric]
            ax.axhline(full_val, color="green", linestyle="--", linewidth=1.5,
                      label=f"Direct (full, N={1484})", alpha=0.7)

        ax.set_xlabel("Number of DFT Training Structures", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])

    axes[0].set_title("Force Accuracy vs DFT Budget", fontsize=13)
    axes[1].set_title("Energy Accuracy vs DFT Budget", fontsize=13)

    fig.suptitle("Delta-Learning MACE Hypothesis Test: Learning Curves",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves to {FIG_DIR / 'learning_curves.png'}")


def plot_improvement_bars(stats, budgets):
    """Bar chart of improvement ratios at each budget."""
    valid_budgets = [N for N in budgets if N in stats and "improvement_ratio" in stats[N]]
    if not valid_budgets:
        return

    ratios = [stats[N]["improvement_ratio"] for N in valid_budgets]
    p_vals = [stats[N]["p_value"] for N in valid_budgets]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(N) for N in valid_budgets], ratios, color="#4CAF50", edgecolor="black", alpha=0.8)

    for i, (bar, p) in enumerate(zip(bars, p_vals)):
        label = f"p={p:.3f}" if p >= 0.001 else f"p<0.001"
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f"{stars}\n{label}", ha="center", va="bottom", fontsize=9)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="No improvement")
    ax.set_xlabel("DFT Budget (N structures)", fontsize=12)
    ax.set_ylabel("Improvement Ratio (Direct MAE / Δ-MACE MAE)", fontsize=12)
    ax.set_title("Δ-MACE Improvement Over Direct Training", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "improvement_bars.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved improvement bars to {FIG_DIR / 'improvement_bars.png'}")


def plot_force_distributions(results, budgets, seeds):
    """Box plots of per-structure force MAEs for direct vs delta at each budget."""
    fig, axes = plt.subplots(1, len(budgets), figsize=(4 * len(budgets), 5), sharey=True)
    if len(budgets) == 1:
        axes = [axes]

    for ax, N in zip(axes, budgets):
        direct_all = []
        delta_all = []
        for s in seeds:
            dk = f"direct_N{N}_s{s}"
            delk = f"delta_N{N}_s{s}"
            if dk in results and "force_mae_per_struct_meV_A" in results[dk]:
                direct_all.extend(results[dk]["force_mae_per_struct_meV_A"])
            if delk in results and "force_mae_per_struct_meV_A" in results[delk]:
                delta_all.extend(results[delk]["force_mae_per_struct_meV_A"])

        data = []
        labels = []
        if direct_all:
            data.append(direct_all)
            labels.append("Direct")
        if delta_all:
            data.append(delta_all)
            labels.append("Δ-MACE")

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = ["#2196F3", "#F44336"]
            for patch, color in zip(bp["boxes"], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        ax.set_title(f"N = {N}", fontsize=12)
        ax.set_ylabel("Force MAE (meV/Å)" if ax == axes[0] else "", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Per-Structure Force MAE Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "force_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved force distributions to {FIG_DIR / 'force_distributions.png'}")


def generate_summary_table(stats, budgets):
    """Print a formatted summary table."""
    w = 78
    sep = "=" * w
    thin = "-" * w

    lines = []
    lines.append(sep)
    lines.append("  PHASE 3: STATISTICAL ANALYSIS SUMMARY")
    lines.append(sep)
    lines.append("")

    lines.append(f"  {'N_DFT':>6}  {'Direct MAE':>16}  {'Δ-MACE MAE':>16}  {'Ratio':>6}  {'p-value':>9}  {'Effect':>10}")
    lines.append(f"  {thin}")

    for N in budgets:
        if N not in stats or "direct_mean" not in stats[N]:
            continue
        s = stats[N]
        d_str = f"{s['direct_mean']:.2f}±{s['direct_std']:.2f}"
        del_str = f"{s['delta_mean']:.2f}±{s['delta_std']:.2f}"
        sig = "*" if s["significant"] else " "
        lines.append(
            f"  {N:>6}  {d_str:>16}  {del_str:>16}  "
            f"{s['improvement_ratio']:>5.2f}×  {s['p_value']:>8.4f}{sig}  {s['effect_label']:>10}"
        )

    lines.append("")
    lines.append("  * = statistically significant (p < 0.05)")
    lines.append("")

    table = "\n".join(lines)
    print(table)
    return table


def main():
    print("=" * 70)
    print("  PHASE 3: STATISTICAL ANALYSIS")
    print("=" * 70)

    results = load_results()

    direct_forces = extract_metric(results, "direct", DFT_BUDGETS, DATA_SEEDS, "force_mae_meV_A")
    delta_forces = extract_metric(results, "delta", DFT_BUDGETS, DATA_SEEDS, "force_mae_meV_A")

    stats = statistical_tests(direct_forces, delta_forces, DFT_BUDGETS)

    table = generate_summary_table(stats, DFT_BUDGETS)

    plot_learning_curves(results, DFT_BUDGETS, DATA_SEEDS)
    plot_improvement_bars(stats, DFT_BUDGETS)
    plot_force_distributions(results, DFT_BUDGETS, DATA_SEEDS)

    energy_stats = statistical_tests(
        extract_metric(results, "direct", DFT_BUDGETS, DATA_SEEDS, "energy_mae_meV_atom"),
        extract_metric(results, "delta", DFT_BUDGETS, DATA_SEEDS, "energy_mae_meV_atom"),
        DFT_BUDGETS,
    )

    analysis = {
        "force_stats": {str(k): v for k, v in stats.items()},
        "energy_stats": {str(k): v for k, v in energy_stats.items()},
    }
    (RESULTS_DIR / "phase3_analysis.json").write_text(json.dumps(analysis, indent=2))
    (RESULTS_DIR / "phase3_summary.txt").write_text(table)

    print(f"\nSaved analysis to {RESULTS_DIR / 'phase3_analysis.json'}")


if __name__ == "__main__":
    main()
