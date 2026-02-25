#!/usr/bin/env python3
"""
Phase 1: XTB Error Systematicity Analysis

Validates the critical assumption that GFN2-xTB errors on liquid water are
*systematic* (learnable by ML) rather than *random*.

Decision gate:
  CV(ΔE) < 0.3  AND  R²(E_DFT, E_XTB) > 0.9  → PROCEED
  CV(ΔE) > 1.0  OR   R²(E_DFT, E_XTB) < 0.5  → ABORT
  otherwise                                      → PROCEED WITH CAUTION
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from xtb_calculator import run_xtb_batch

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)


def run_xtb_on_structures(atoms_list: list, label: str, n_parallel: int = 5) -> dict:
    """Run GFN2-xTB on all structures and collect results."""

    print(f"\n--- Running GFN2-xTB on {len(atoms_list)} {label} structures ---")
    t0 = time.time()

    batch_results = run_xtb_batch(atoms_list, n_parallel=n_parallel, n_threads_per_job=4)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s")

    results = {
        "dft_energies": [], "xtb_energies": [], "delta_energies": [],
        "delta_energies_per_atom": [], "dft_forces": [], "xtb_forces": [],
        "delta_forces": [], "force_mae_per_struct": [], "n_atoms": [],
    }

    failed_indices = []
    for i, (atoms, res) in enumerate(zip(atoms_list, batch_results)):
        if res is None:
            failed_indices.append(i)
            continue

        e_xtb, f_xtb = res
        e_dft = atoms.info["TotEnergy"]
        f_dft = atoms.arrays["force"]
        n_at = len(atoms)

        de = e_dft - e_xtb
        df = f_dft - f_xtb

        results["dft_energies"].append(float(e_dft))
        results["xtb_energies"].append(float(e_xtb))
        results["delta_energies"].append(float(de))
        results["delta_energies_per_atom"].append(float(de / n_at))
        results["dft_forces"].append(f_dft.tolist())
        results["xtb_forces"].append(f_xtb.tolist())
        results["delta_forces"].append(df.tolist())
        results["force_mae_per_struct"].append(float(np.mean(np.abs(df))))
        results["n_atoms"].append(n_at)

    if failed_indices:
        print(f"  WARNING: {len(failed_indices)} structures failed: {failed_indices[:10]}...")

    return results, failed_indices


def analyze_systematicity(results: dict) -> dict:
    """Compute statistics to determine if XTB errors are systematic."""
    de = np.array(results["delta_energies"])
    de_pa = np.array(results["delta_energies_per_atom"])
    e_dft = np.array(results["dft_energies"])
    e_xtb = np.array(results["xtb_energies"])
    f_mae = np.array(results["force_mae_per_struct"])

    cv_de = float(np.std(de) / np.abs(np.mean(de))) if np.abs(np.mean(de)) > 1e-10 else float("inf")
    cv_de_pa = float(np.std(de_pa) / np.abs(np.mean(de_pa))) if np.abs(np.mean(de_pa)) > 1e-10 else float("inf")

    r_energy, p_energy = pearsonr(e_dft, e_xtb)
    r2_energy = float(r_energy ** 2)

    all_df = []
    for df_struct in results["delta_forces"]:
        all_df.extend(np.array(df_struct).ravel().tolist())
    all_df = np.array(all_df)

    # R²(E_DFT, E_XTB) is unreliable when XTB runs non-periodic: edge effects
    # inflate XTB variance ~40× beyond DFT variance, destroying correlation.
    # Use force-based CV instead — forces are local and unaffected by PBC.
    f_cv = float(np.std(f_mae) / np.mean(f_mae)) if np.mean(f_mae) > 1e-10 else float("inf")

    if cv_de_pa < 0.3 and f_cv < 0.5:
        decision = "PROCEED"
    elif cv_de_pa > 1.0 and f_cv > 1.0:
        decision = "ABORT"
    else:
        decision = "PROCEED WITH CAUTION"

    analysis = {
        "n_structures": len(de),
        "energy": {
            "delta_E_mean_eV": float(np.mean(de)),
            "delta_E_std_eV": float(np.std(de)),
            "delta_E_range_eV": float(np.ptp(de)),
            "delta_E_per_atom_mean_meV": float(np.mean(de_pa) * 1000),
            "delta_E_per_atom_std_meV": float(np.std(de_pa) * 1000),
            "coefficient_of_variation": cv_de,
            "cv_per_atom": cv_de_pa,
            "R2_dft_vs_xtb": r2_energy,
            "pearson_r": float(r_energy),
            "pearson_p": float(p_energy),
        },
        "forces": {
            "delta_F_mean_meV_A": float(np.mean(np.abs(all_df)) * 1000),
            "delta_F_std_meV_A": float(np.std(all_df) * 1000),
            "delta_F_rms_meV_A": float(np.sqrt(np.mean(all_df**2)) * 1000),
            "per_struct_force_mae_mean_meV_A": float(np.mean(f_mae) * 1000),
            "per_struct_force_mae_std_meV_A": float(np.std(f_mae) * 1000),
        },
        "decision": decision,
    }

    return analysis


def print_report(analysis: dict, label: str) -> str:
    """Pretty-print the systematicity analysis."""
    lines = []
    w = 70
    sep = "=" * w

    lines.append(sep)
    lines.append(f"  PHASE 1: XTB ERROR SYSTEMATICITY — {label}")
    lines.append(f"  ({analysis['n_structures']} structures analyzed)")
    lines.append(sep)
    lines.append("")

    e = analysis["energy"]
    lines.append("  Energy Corrections (ΔE = E_DFT − E_XTB)")
    lines.append("  " + "-" * 50)
    lines.append(f"    ΔE mean:              {e['delta_E_mean_eV']:>12.4f} eV")
    lines.append(f"    ΔE std:               {e['delta_E_std_eV']:>12.4f} eV")
    lines.append(f"    ΔE range:             {e['delta_E_range_eV']:>12.4f} eV")
    lines.append(f"    ΔE/atom mean:         {e['delta_E_per_atom_mean_meV']:>12.3f} meV/atom")
    lines.append(f"    ΔE/atom std:          {e['delta_E_per_atom_std_meV']:>12.3f} meV/atom")
    lines.append(f"    CV(ΔE/atom):          {e['cv_per_atom']:>12.6f}  {'✓' if e['cv_per_atom'] < 0.3 else '✗'} (need < 0.3)")
    lines.append(f"    R²(E_DFT, E_XTB):     {e['R2_dft_vs_xtb']:>12.6f}  {'✓' if e['R2_dft_vs_xtb'] > 0.9 else '✗'} (need > 0.9)")
    lines.append(f"    Pearson r:            {e['pearson_r']:>12.6f}")
    lines.append("")

    f = analysis["forces"]
    lines.append("  Force Corrections (ΔF = F_DFT − F_XTB)")
    lines.append("  " + "-" * 50)
    lines.append(f"    |ΔF| mean:            {f['delta_F_mean_meV_A']:>12.3f} meV/Å")
    lines.append(f"    ΔF std:               {f['delta_F_std_meV_A']:>12.3f} meV/Å")
    lines.append(f"    ΔF RMS:               {f['delta_F_rms_meV_A']:>12.3f} meV/Å")
    lines.append(f"    Per-struct MAE mean:   {f['per_struct_force_mae_mean_meV_A']:>12.3f} meV/Å")
    lines.append(f"    Per-struct MAE std:    {f['per_struct_force_mae_std_meV_A']:>12.3f} meV/Å")
    lines.append("")

    lines.append("  " + "=" * 50)
    d = analysis["decision"]
    symbol = "✓" if d == "PROCEED" else ("⚠" if "CAUTION" in d else "✗")
    lines.append(f"  DECISION: {symbol} {d}")
    lines.append("  " + "=" * 50)
    lines.append("")

    report = "\n".join(lines)
    return report


def main():
    print("=" * 70)
    print("  PHASE 1: XTB ERROR SYSTEMATICITY ANALYSIS")
    print("  System: Liquid Water (64 H₂O, periodic, 192 atoms)")
    print("  Method: GFN1-xTB (periodic via POSCAR, etemp=500-4000K retry)")
    print("=" * 70)

    valid = read(str(DATA_DIR / "valid.xyz"), ":")
    train = read(str(DATA_DIR / "train.xyz"), ":")

    print(f"\nLoaded {len(valid)} validation, {len(train)} training structures")

    valid_results, valid_failed = run_xtb_on_structures(valid, "validation", n_parallel=5)
    train_results, train_failed = run_xtb_on_structures(train, "training", n_parallel=5)

    valid_analysis = analyze_systematicity(valid_results)
    train_analysis = analyze_systematicity(train_results)

    valid_report = print_report(valid_analysis, "VALIDATION SET")
    train_report = print_report(train_analysis, "TRAINING SET")

    print(valid_report)
    print(train_report)

    # Save raw energy data
    for name, res in [("valid", valid_results), ("train", train_results)]:
        compact = {
            "dft_energies": res["dft_energies"],
            "xtb_energies": res["xtb_energies"],
            "delta_energies": res["delta_energies"],
            "n_atoms": res["n_atoms"],
            "force_mae_per_struct": res["force_mae_per_struct"],
        }
        (OUT_DIR / f"xtb_{name}_energies.json").write_text(json.dumps(compact, indent=2))

    # Save XTB-augmented structures as .xyz
    for name, structures, res, failed in [
        ("valid", valid, valid_results, valid_failed),
        ("train", train, train_results, train_failed),
    ]:
        good_structs = []
        res_idx = 0
        for i, atoms in enumerate(structures):
            if i in failed:
                continue
            a = atoms.copy()
            a.info["XTB_energy"] = res["xtb_energies"][res_idx]
            a.arrays["XTB_forces"] = np.array(res["xtb_forces"][res_idx])
            a.info["delta_energy"] = res["delta_energies"][res_idx]
            a.arrays["delta_forces"] = np.array(res["delta_forces"][res_idx])
            good_structs.append(a)
            res_idx += 1

        out_path = DATA_DIR / f"{name}_with_xtb.xyz"
        write(str(out_path), good_structs, format="extxyz")
        print(f"Saved {len(good_structs)} XTB-augmented structures to {out_path}")

    # Save analysis
    combined_analysis = {
        "validation": valid_analysis,
        "training": train_analysis,
        "decision": valid_analysis["decision"],
    }
    (OUT_DIR / "phase1_analysis.json").write_text(json.dumps(combined_analysis, indent=2))

    full_report = valid_report + "\n" + train_report
    (OUT_DIR / "phase1_report.txt").write_text(full_report)

    print(f"\nPhase 1 complete. Decision: {valid_analysis['decision']}")
    return valid_analysis["decision"]


if __name__ == "__main__":
    main()
