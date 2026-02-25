#!/usr/bin/env python3
"""
Phase 2: Controlled Experiment — Direct MACE vs Δ-MACE

Trains and evaluates:
  Arm A: Direct-MACE (full dataset, reference baseline)
  Arm B: Direct-MACE (limited DFT budgets: 50, 100, 200, 500)
  Arm C: Δ-MACE (same limited budgets, trained on corrections)

Each budget × 3 data seeds for statistical power.
Uses smaller MACE architecture for fast iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

os.environ["OMP_NUM_THREADS"] = "4"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DFT_BUDGETS = [50, 100, 200, 500]
DATA_SEEDS = [42, 123, 7]

COMMON_MACE_ARGS = [
    "--max_num_epochs", "150",
    "--patience", "30",
    "--model", "MACE",
    "--hidden_irreps", "64x0e+64x1o",
    "--r_max", "5.0",
    "--num_interactions", "2",
    "--correlation", "3",
    "--batch_size", "4",
    "--lr", "0.01",
    "--swa",
    "--swa_lr", "0.001",
    "--swa_energy_weight", "100.0",
    "--swa_forces_weight", "10.0",
    "--ema",
    "--ema_decay", "0.99",
    "--energy_key", "REF_energy",
    "--forces_key", "REF_forces",
    "--error_table", "PerAtomMAE",
    "--E0s", "average",
    "--enable_cueq=True",
]


def compute_isolated_atom_energies_xtb():
    """Compute GFN2-xTB isolated atom energies for H and O."""
    from ase import Atoms
    from xtb.ase.calculator import XTB

    e0 = {}
    for sym, z in [("H", 1), ("O", 8)]:
        atom = Atoms(sym, positions=[[0, 0, 0]])
        atom.calc = XTB(method="GFN2-xTB")
        try:
            e0[sym] = atom.get_potential_energy()
        except Exception:
            e0[sym] = 0.0
            print(f"  Warning: Could not compute XTB E0 for {sym}, using 0.0")
    return e0


def prepare_delta_dataset(structures: list, xtb_e0: dict) -> list:
    """
    Convert structures with DFT+XTB data into delta-learning targets.

    For delta-learning:
      REF_energy = E_DFT - E_XTB  (the correction MACE needs to learn)
      REF_forces = F_DFT - F_XTB
    """
    delta_structs = []

    for atoms in structures:
        a = atoms.copy()

        e_dft = atoms.info["TotEnergy"]
        e_xtb = atoms.info["XTB_energy"]
        f_dft = atoms.arrays["force"]
        f_xtb = atoms.arrays["XTB_forces"]

        delta_e = e_dft - e_xtb
        delta_f = f_dft - f_xtb

        a.info = {"REF_energy": float(delta_e)}
        new_arrays = {"REF_forces": delta_f.copy()}
        for key in list(a.arrays.keys()):
            if key not in ("numbers", "positions"):
                del a.arrays[key]
        a.arrays["REF_forces"] = new_arrays["REF_forces"]

        delta_structs.append(a)

    return delta_structs


def prepare_direct_dataset(structures: list) -> list:
    """Convert structures to direct-learning format (DFT targets only)."""
    direct_structs = []

    for atoms in structures:
        a = atoms.copy()

        a.info = {"REF_energy": float(atoms.info["TotEnergy"])}
        f_dft = atoms.arrays["force"].copy()
        for key in list(a.arrays.keys()):
            if key not in ("numbers", "positions"):
                del a.arrays[key]
        a.arrays["REF_forces"] = f_dft

        direct_structs.append(a)

    return direct_structs


def train_mace(name: str, train_file: str, valid_file: str, seed: int, device: str, extra_args: list = None):
    """Launch MACE training directly via mace_run_train (bypass wrapper to avoid arg conflicts)."""
    cmd = [
        "mace_run_train",
        "--train_file", train_file,
        "--valid_file", valid_file,
        "--work_dir", str(RUNS_DIR / name),
        "--name", name,
        "--seed", str(seed),
        "--device", device,
    ] + COMMON_MACE_ARGS

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Train: {train_file}")
    print(f"  Valid: {valid_file}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s (exit code: {result.returncode})")

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        log_dir = RUNS_DIR / name / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                log_text = log_file.read_text()
                print(f"  Log tail ({log_file.name}):")
                for line in log_text.strip().split("\n")[-10:]:
                    print(f"    {line}")

    return result.returncode, elapsed


def find_model(name: str) -> Path | None:
    """Find the best trained model checkpoint. Prefers SWA (stagetwo) over stage 1."""
    run_dir = RUNS_DIR / name

    # SWA model is better when available
    swa = run_dir / f"{name}_stagetwo.model"
    if swa.exists():
        return swa

    stage1 = run_dir / f"{name}.model"
    if stage1.exists():
        return stage1

    for f in sorted(run_dir.rglob("*_stagetwo.model")):
        if "compiled" not in f.name:
            return f

    for f in sorted(run_dir.rglob("*.model")):
        if "compiled" not in f.name and "stagetwo" not in f.name:
            return f

    return None


def evaluate_model(model_path: Path, valid_structs: list, device: str) -> dict:
    """Evaluate a MACE model on validation structures."""
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=str(model_path), device=device, default_dtype="float32")

    pred_e, pred_f = [], []
    for atoms in valid_structs:
        a = atoms.copy()
        a.calc = calc
        pred_e.append(a.get_potential_energy())
        pred_f.append(a.get_forces().copy())

    return pred_e, pred_f


def evaluate_delta_model(model_path: Path, valid_structs: list, device: str) -> tuple:
    """
    Evaluate a Δ-MACE model: prediction = XTB + MACE_correction.

    valid_structs must have XTB_energy and XTB_forces in info/arrays.
    """
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=str(model_path), device=device, default_dtype="float32")

    pred_e, pred_f = [], []
    for atoms in valid_structs:
        e_xtb = atoms.info["XTB_energy"]
        f_xtb = atoms.arrays["XTB_forces"]

        a = atoms.copy()
        for key in ["TotEnergy", "XTB_energy", "delta_energy", "cutoff", "nneightol"]:
            a.info.pop(key, None)
        for key in ["force", "XTB_forces", "delta_forces"]:
            if key in a.arrays:
                del a.arrays[key]

        a.calc = calc
        delta_e = a.get_potential_energy()
        delta_f = a.get_forces()

        pred_e.append(e_xtb + delta_e)
        pred_f.append(f_xtb + delta_f)

    return pred_e, pred_f


def compute_metrics(pred_e, pred_f, ref_e, ref_f, n_atoms: int) -> dict:
    """Compute energy and force error metrics."""
    pred_e = np.array(pred_e)
    ref_e = np.array(ref_e)

    e_errors = np.abs(pred_e - ref_e) / n_atoms
    e_mae = float(np.mean(e_errors)) * 1000
    e_rmse = float(np.sqrt(np.mean(e_errors**2))) * 1000

    f_errors_all = np.concatenate([np.abs(pf - rf).ravel() for pf, rf in zip(pred_f, ref_f)])
    f_mae = float(np.mean(f_errors_all)) * 1000
    f_rmse = float(np.sqrt(np.mean(f_errors_all**2))) * 1000

    f_mae_per_struct = [float(np.mean(np.abs(pf - rf))) * 1000 for pf, rf in zip(pred_f, ref_f)]

    cos_vals = []
    for pf, rf in zip(pred_f, ref_f):
        pn = np.linalg.norm(pf, axis=1, keepdims=True).clip(1e-12)
        rn = np.linalg.norm(rf, axis=1, keepdims=True).clip(1e-12)
        cos_vals.append(float(np.mean(np.sum(pf / pn * rf / rn, axis=1))))

    r2 = float(np.corrcoef(pred_e, ref_e)[0, 1] ** 2) if len(pred_e) > 1 else 0.0

    return {
        "energy_mae_meV_atom": e_mae,
        "energy_rmse_meV_atom": e_rmse,
        "force_mae_meV_A": f_mae,
        "force_rmse_meV_A": f_rmse,
        "force_mae_per_struct_meV_A": f_mae_per_struct,
        "force_cosine_sim": float(np.mean(cos_vals)),
        "energy_R2": r2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budgets", nargs="+", type=int, default=DFT_BUDGETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DATA_SEEDS)
    ap.add_argument("--skip-training", action="store_true", help="Skip training, just evaluate")
    ap.add_argument("--train-only", action="store_true", help="Skip evaluation")
    args = ap.parse_args()

    print("=" * 70)
    print("  PHASE 2: CONTROLLED EXPERIMENT")
    print("  Direct MACE vs Δ-MACE at varying DFT budgets")
    print("=" * 70)

    xtb_train_path = DATA_DIR / "train_with_xtb.xyz"
    xtb_valid_path = DATA_DIR / "valid_with_xtb.xyz"

    if not xtb_train_path.exists() or not xtb_valid_path.exists():
        print("\nERROR: XTB-augmented data not found. Run phase1_xtb_systematicity.py first.")
        sys.exit(1)

    train_all = read(str(xtb_train_path), ":")
    valid_all = read(str(xtb_valid_path), ":")
    n_atoms = len(valid_all[0])

    ref_e = [a.info["TotEnergy"] for a in valid_all]
    ref_f = [a.arrays["force"].copy() for a in valid_all]

    valid_direct = prepare_direct_dataset(valid_all)
    valid_direct_path = DATA_DIR / "valid_direct.xyz"
    write(str(valid_direct_path), valid_direct, format="extxyz")

    valid_delta = prepare_delta_dataset(valid_all, {})
    valid_delta_path = DATA_DIR / "valid_delta.xyz"
    write(str(valid_delta_path), valid_delta, format="extxyz")

    print(f"\nDataset: {len(train_all)} train, {len(valid_all)} valid, {n_atoms} atoms/struct")

    all_results = {}

    if not args.skip_training:
        for N in args.budgets:
            for seed in args.seeds:
                np.random.seed(seed)
                indices = np.random.choice(len(train_all), N, replace=False)
                subset = [train_all[i] for i in indices]

                # --- ARM B: Direct-MACE ---
                direct_name = f"direct_N{N}_s{seed}"
                direct_subset = prepare_direct_dataset(subset)
                direct_train_path = DATA_DIR / f"direct_N{N}_s{seed}.xyz"
                write(str(direct_train_path), direct_subset, format="extxyz")

                rc, elapsed = train_mace(
                    name=direct_name,
                    train_file=str(direct_train_path),
                    valid_file=str(valid_direct_path),
                    seed=seed,
                    device=args.device,
                )

                # --- ARM C: Δ-MACE ---
                delta_name = f"delta_N{N}_s{seed}"
                delta_subset = prepare_delta_dataset(subset, {})
                delta_train_path = DATA_DIR / f"delta_N{N}_s{seed}.xyz"
                write(str(delta_train_path), delta_subset, format="extxyz")

                rc, elapsed = train_mace(
                    name=delta_name,
                    train_file=str(delta_train_path),
                    valid_file=str(valid_delta_path),
                    seed=seed,
                    device=args.device,
                )

        # --- ARM A: Direct-MACE on full dataset (reference) ---
        full_direct = prepare_direct_dataset(train_all)
        full_direct_path = DATA_DIR / "direct_full.xyz"
        write(str(full_direct_path), full_direct, format="extxyz")

        train_mace(
            name="direct_full",
            train_file=str(full_direct_path),
            valid_file=str(valid_direct_path),
            seed=42,
            device=args.device,
        )

    if args.train_only:
        print("\n--train-only specified, skipping evaluation.")
        return

    # Evaluation
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)

    for N in args.budgets:
        for seed in args.seeds:
            # Direct
            direct_name = f"direct_N{N}_s{seed}"
            model = find_model(direct_name)
            if model:
                print(f"\nEvaluating {direct_name}...")
                pe, pf = evaluate_model(model, valid_direct, args.device)
                metrics = compute_metrics(pe, pf, ref_e, ref_f, n_atoms)
                key = f"direct_N{N}_s{seed}"
                all_results[key] = metrics
                print(f"  Energy MAE: {metrics['energy_mae_meV_atom']:.3f} meV/atom")
                print(f"  Force MAE:  {metrics['force_mae_meV_A']:.3f} meV/Å")
            else:
                print(f"  Model not found for {direct_name}")

            # Delta
            delta_name = f"delta_N{N}_s{seed}"
            model = find_model(delta_name)
            if model:
                print(f"\nEvaluating {delta_name}...")
                pe, pf = evaluate_delta_model(model, valid_all, args.device)
                metrics = compute_metrics(pe, pf, ref_e, ref_f, n_atoms)
                key = f"delta_N{N}_s{seed}"
                all_results[key] = metrics
                print(f"  Energy MAE: {metrics['energy_mae_meV_atom']:.3f} meV/atom")
                print(f"  Force MAE:  {metrics['force_mae_meV_A']:.3f} meV/Å")
            else:
                print(f"  Model not found for {delta_name}")

    # Full dataset reference
    model = find_model("direct_full")
    if model:
        print(f"\nEvaluating direct_full (reference)...")
        pe, pf = evaluate_model(model, valid_direct, args.device)
        metrics = compute_metrics(pe, pf, ref_e, ref_f, n_atoms)
        all_results["direct_full"] = metrics
        print(f"  Energy MAE: {metrics['energy_mae_meV_atom']:.3f} meV/atom")
        print(f"  Force MAE:  {metrics['force_mae_meV_A']:.3f} meV/Å")

    # Save all results
    serializable = {}
    for k, v in all_results.items():
        sv = {}
        for mk, mv in v.items():
            if isinstance(mv, list):
                sv[mk] = [round(x, 4) for x in mv]
            else:
                sv[mk] = round(mv, 6)
        serializable[k] = sv

    (RESULTS_DIR / "phase2_results.json").write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved to {RESULTS_DIR / 'phase2_results.json'}")


if __name__ == "__main__":
    main()
