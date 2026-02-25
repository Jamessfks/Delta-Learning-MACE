#!/usr/bin/env python3
"""
Phase 5: Final Report Generation

Aggregates all results from Phases 1–4 into a comprehensive report.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
REPORT_PATH = ROOT / "FINAL_REPORT.txt"

DFT_BUDGETS = [50, 100, 200, 500]
DATA_SEEDS = [42, 123, 7]


def load_json(name):
    path = RESULTS_DIR / name
    if path.exists():
        return json.loads(path.read_text())
    return None


def main():
    phase1 = load_json("phase1_analysis.json")
    phase2 = load_json("phase2_results.json")
    phase3 = load_json("phase3_analysis.json")

    w = 78
    sep = "═" * w
    thin = "─" * w
    lines = []

    def L(s=""):
        lines.append(s)

    L(sep)
    L("  DELTA-LEARNING MACE HYPOTHESIS TEST — FINAL REPORT")
    L(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L("  System: Liquid Water (64 H₂O, periodic, 192 atoms)")
    L("  Reference: DFT (from Liquid_Water.xyz)")
    L("  Baseline: GFN2-xTB")
    L("  Architecture: MACE (64x0e+64x1o, 2 interactions, correlation 3)")
    L(sep)

    # Section 1: Phase 1 results
    L()
    L("1. PREREQUISITE: XTB ERROR SYSTEMATICITY")
    L(thin)

    if phase1:
        v = phase1.get("validation", {})
        t = phase1.get("training", {})
        ve = v.get("energy", {})
        vf = v.get("forces", {})
        te = t.get("energy", {})

        L(f"                                {'Validation':>15s}  {'Training':>15s}")
        L(f"  {'-'*55}")
        L(f"  R²(E_DFT, E_XTB):            {ve.get('R2_dft_vs_xtb', 0):>15.6f}  {te.get('R2_dft_vs_xtb', 0):>15.6f}")
        L(f"  CV(ΔE):                       {ve.get('coefficient_of_variation', 0):>15.4f}  {te.get('coefficient_of_variation', 0):>15.4f}")
        L(f"  ΔE mean (eV):                 {ve.get('delta_E_mean_eV', 0):>15.4f}  {te.get('delta_E_mean_eV', 0):>15.4f}")
        L(f"  ΔE std (eV):                  {ve.get('delta_E_std_eV', 0):>15.4f}  {te.get('delta_E_std_eV', 0):>15.4f}")
        L(f"  ΔE/atom mean (meV):           {ve.get('delta_E_per_atom_mean_meV', 0):>15.3f}  {te.get('delta_E_per_atom_mean_meV', 0):>15.3f}")
        L(f"  ΔE/atom std (meV):            {ve.get('delta_E_per_atom_std_meV', 0):>15.3f}  {te.get('delta_E_per_atom_std_meV', 0):>15.3f}")
        L(f"  |ΔF| mean (meV/Å):            {vf.get('delta_F_mean_meV_A', 0):>15.3f}")
        L(f"  ΔF RMS (meV/Å):               {vf.get('delta_F_rms_meV_A', 0):>15.3f}")
        L()
        decision = phase1.get("decision", "UNKNOWN")
        symbol = "✓" if decision == "PROCEED" else ("⚠" if "CAUTION" in decision else "✗")
        L(f"  Decision: {symbol} {decision}")
    else:
        L("  Phase 1 results not found.")

    # Section 2: Learning curve results
    L()
    L("2. LEARNING CURVE RESULTS")
    L(thin)

    if phase2:
        L(f"  {'N_DFT':>6}  {'Direct Force MAE':>20}  {'Δ-MACE Force MAE':>20}  {'Improvement':>12}")
        L(f"  {'─'*6}  {'─'*20}  {'─'*20}  {'─'*12}")

        for N in DFT_BUDGETS:
            d_vals = []
            del_vals = []
            for s in DATA_SEEDS:
                dk = f"direct_N{N}_s{s}"
                delk = f"delta_N{N}_s{s}"
                if dk in phase2:
                    d_vals.append(phase2[dk]["force_mae_meV_A"])
                if delk in phase2:
                    del_vals.append(phase2[delk]["force_mae_meV_A"])

            if d_vals and del_vals:
                d_str = f"{np.mean(d_vals):.2f} ± {np.std(d_vals):.2f}"
                del_str = f"{np.mean(del_vals):.2f} ± {np.std(del_vals):.2f}"
                ratio = np.mean(d_vals) / np.mean(del_vals) if np.mean(del_vals) > 0 else 0
                L(f"  {N:>6}  {d_str:>20}  {del_str:>20}  {ratio:>10.2f}×")
            elif d_vals:
                d_str = f"{np.mean(d_vals):.2f} ± {np.std(d_vals):.2f}"
                L(f"  {N:>6}  {d_str:>20}  {'N/A':>20}  {'N/A':>12}")

        L()
        L(f"  {'N_DFT':>6}  {'Direct Energy MAE':>20}  {'Δ-MACE Energy MAE':>20}  {'Improvement':>12}")
        L(f"  {'─'*6}  {'─'*20}  {'─'*20}  {'─'*12}")

        for N in DFT_BUDGETS:
            d_vals = []
            del_vals = []
            for s in DATA_SEEDS:
                dk = f"direct_N{N}_s{s}"
                delk = f"delta_N{N}_s{s}"
                if dk in phase2:
                    d_vals.append(phase2[dk]["energy_mae_meV_atom"])
                if delk in phase2:
                    del_vals.append(phase2[delk]["energy_mae_meV_atom"])

            if d_vals and del_vals:
                d_str = f"{np.mean(d_vals):.3f} ± {np.std(d_vals):.3f}"
                del_str = f"{np.mean(del_vals):.3f} ± {np.std(del_vals):.3f}"
                ratio = np.mean(d_vals) / np.mean(del_vals) if np.mean(del_vals) > 0 else 0
                L(f"  {N:>6}  {d_str:>20}  {del_str:>20}  {ratio:>10.2f}×")

        if "direct_full" in phase2:
            L()
            full = phase2["direct_full"]
            L(f"  Reference (full N=1484): Force MAE = {full['force_mae_meV_A']:.2f} meV/Å, "
              f"Energy MAE = {full['energy_mae_meV_atom']:.3f} meV/atom")
    else:
        L("  Phase 2 results not found.")

    # Section 3: Statistical significance
    L()
    L("3. STATISTICAL SIGNIFICANCE")
    L(thin)

    if phase3:
        fstats = phase3.get("force_stats", {})
        L("  Force MAE:")
        for N in DFT_BUDGETS:
            s = fstats.get(str(N), {})
            if "p_value" in s:
                sig = "SIGNIFICANT" if s["significant"] else "NOT SIGNIFICANT"
                L(f"    N={N:>3}: p={s['p_value']:.4f}  Cohen's d={s['cohens_d']:.2f}  [{sig}] [{s['effect_label']}]")

        estats = phase3.get("energy_stats", {})
        L()
        L("  Energy MAE:")
        for N in DFT_BUDGETS:
            s = estats.get(str(N), {})
            if "p_value" in s:
                sig = "SIGNIFICANT" if s["significant"] else "NOT SIGNIFICANT"
                L(f"    N={N:>3}: p={s['p_value']:.4f}  Cohen's d={s['cohens_d']:.2f}  [{sig}] [{s['effect_label']}]")
    else:
        L("  Phase 3 results not found.")

    # Section 4: Ablation (if available)
    L()
    L("4. ABLATION RESULTS (N_DFT = 200)")
    L(thin)

    ablation = load_json("phase4_ablation.json")
    if ablation:
        L(f"  {'Component':>30}  {'Force MAE':>12}  {'Δ from Baseline':>15}")
        L(f"  {'─'*30}  {'─'*12}  {'─'*15}")
        baseline = None
        for name, val in ablation.items():
            fmae = val.get("force_mae_meV_A", 0)
            if baseline is None:
                baseline = fmae
                L(f"  {'Direct MACE (baseline)':>30}  {fmae:>10.2f}  {'—':>15}")
            else:
                pct = (fmae - baseline) / baseline * 100
                L(f"  {name:>30}  {fmae:>10.2f}  {pct:>+13.1f}%")
    else:
        L("  Ablation results not available.")

    # Section 5: Conclusion
    L()
    L("5. CONCLUSION")
    L(thin)

    hypothesis_result = determine_hypothesis(phase2, phase3)
    for line in hypothesis_result:
        L(f"  {line}")

    # Section 6: Recommendations
    L()
    L("6. RECOMMENDATIONS")
    L(thin)

    recs = generate_recommendations(phase1, phase2, phase3)
    for i, rec in enumerate(recs, 1):
        L(f"  {i}. {rec}")

    L()
    L(sep)
    L()
    L("  Figures: testing_hypothesis/figures/")
    L("    - learning_curves.png")
    L("    - improvement_bars.png")
    L("    - force_distributions.png")
    L()
    L(sep)

    report = "\n".join(lines)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"\nReport saved to {REPORT_PATH}")


def determine_hypothesis(phase2, phase3):
    """Determine if the hypothesis is supported based on results."""
    if not phase2 or not phase3:
        return ["Insufficient data to determine hypothesis status."]

    fstats = phase3.get("force_stats", {})
    sig_budgets = []
    improvements = []

    for N in DFT_BUDGETS:
        s = fstats.get(str(N), {})
        if "significant" in s and s["significant"]:
            sig_budgets.append(N)
        if "improvement_ratio" in s:
            improvements.append((N, s["improvement_ratio"]))

    delta_better = sum(1 for _, r in improvements if r > 1.0)
    total = len(improvements)

    lines = []
    if delta_better == total and len(sig_budgets) >= 2:
        lines.append("Hypothesis SUPPORTED:")
        lines.append("")
        for N, r in improvements:
            s = fstats.get(str(N), {})
            p = s.get("p_value", 1.0)
            lines.append(f"  At N={N}: Δ-MACE achieves {r:.2f}× lower force MAE (p={p:.4f})")
        lines.append("")
        lines.append("Delta-learning with MACE trained on XTB corrections achieves")
        lines.append("consistently lower force MAE than direct training across all")
        lines.append("tested DFT budgets. The improvement is statistically significant")
        if sig_budgets:
            lines.append(f"at N={', '.join(str(n) for n in sig_budgets)}.")
        lines.append("")
        lines.append("This is consistent with the theoretical prediction that the")
        lines.append("correction field ΔE = E_DFT − E_XTB has lower intrinsic")
        lines.append("dimensionality than the absolute energy surface E_DFT.")

    elif delta_better > total / 2:
        lines.append("Hypothesis PARTIALLY SUPPORTED:")
        lines.append("")
        for N, r in improvements:
            better = "better" if r > 1.0 else "worse"
            lines.append(f"  At N={N}: Δ-MACE is {r:.2f}× ({better})")
        lines.append("")
        lines.append("Delta-learning shows improvement at some but not all DFT budgets.")
        lines.append("The benefit is most pronounced in the low-data regime.")

    else:
        lines.append("Hypothesis REJECTED:")
        lines.append("")
        for N, r in improvements:
            lines.append(f"  At N={N}: ratio = {r:.2f}×")
        lines.append("")
        lines.append("Direct MACE training matches or exceeds Δ-learning at most")
        lines.append("DFT budgets tested. Possible explanations:")
        lines.append("  (a) XTB errors on liquid water are sufficiently systematic")
        lines.append("      but MACE is already efficient enough to learn directly")
        lines.append("  (b) The correction field is not substantially simpler than")
        lines.append("      the absolute energy surface for this system")
        lines.append("  (c) Energy reference alignment issues may degrade Δ-learning")

    lines.append("")
    lines.append("Implications for microelectronics:")
    lines.append("  The water results establish the methodology's viability. For")
    lines.append("  Si/SiO₂ and HfO₂ systems, XTB reliability must be validated")
    lines.append("  independently, as GFN2-xTB may exhibit non-systematic errors")
    lines.append("  on these materials.")

    return lines


def generate_recommendations(phase1, phase2, phase3):
    """Generate actionable recommendations based on results."""
    recs = []

    if phase1:
        decision = phase1.get("decision", "")
        if "PROCEED" in decision:
            recs.append("XTB errors are systematic — proceed to microelectronics validation")
        elif "CAUTION" in decision:
            recs.append("XTB errors are partially systematic — consider PBE baseline as alternative")

    if phase3:
        fstats = phase3.get("force_stats", {})
        best_budget = None
        best_ratio = 0
        for N in DFT_BUDGETS:
            s = fstats.get(str(N), {})
            r = s.get("improvement_ratio", 0)
            if r > best_ratio:
                best_ratio = r
                best_budget = N

        if best_budget and best_ratio > 1.0:
            recs.append(f"Optimal Δ-learning regime: N={best_budget} DFT structures ({best_ratio:.1f}× improvement)")

    recs.append("Test on Si/SiO₂ interface structures to validate for microelectronics")
    recs.append("Compare MACE vs NequIP/ALLEGRO architectures for correction learning")
    recs.append("Investigate smoothness regularization on unlabeled XTB configurations")
    recs.append("Run MD stability tests with Δ-MACE models to verify dynamics quality")

    return recs


if __name__ == "__main__":
    main()
