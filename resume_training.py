#!/usr/bin/env python3
"""
Resume Phase 2 training — skips already-completed models.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "models"

DFT_BUDGETS = [100, 200, 500]
DATA_SEEDS = [42, 123, 7]

MACE_ARGS = [
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


def is_done(name: str) -> bool:
    d = RUNS_DIR / name
    models = list(d.glob("*.model")) if d.exists() else []
    return any("compiled" not in m.name for m in models)


def prepare_direct(structures):
    out = []
    for atoms in structures:
        a = atoms.copy()
        a.info = {"REF_energy": float(atoms.info["TotEnergy"])}
        f = atoms.arrays["force"].copy()
        for key in list(a.arrays.keys()):
            if key not in ("numbers", "positions"):
                del a.arrays[key]
        a.arrays["REF_forces"] = f
        out.append(a)
    return out


def prepare_delta(structures):
    out = []
    for atoms in structures:
        a = atoms.copy()
        e_dft = atoms.info["TotEnergy"]
        e_xtb = atoms.info["XTB_energy"]
        f_dft = atoms.arrays["force"]
        f_xtb = atoms.arrays["XTB_forces"]
        a.info = {"REF_energy": float(e_dft - e_xtb)}
        df = (f_dft - f_xtb).copy()
        for key in list(a.arrays.keys()):
            if key not in ("numbers", "positions"):
                del a.arrays[key]
        a.arrays["REF_forces"] = df
        out.append(a)
    return out


def train(name, train_file, valid_file, seed, device="cuda"):
    cmd = [
        "mace_run_train",
        "--train_file", train_file,
        "--valid_file", valid_file,
        "--work_dir", str(RUNS_DIR / name),
        "--name", name,
        "--seed", str(seed),
        "--device", device,
    ] + MACE_ARGS

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"  {'OK' if ok else 'FAILED'} in {elapsed:.0f}s", flush=True)
    if not ok:
        print(f"  STDERR tail: {result.stderr[-300:]}", flush=True)
    return ok


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"

    train_all = read(str(DATA_DIR / "train_with_xtb.xyz"), ":")
    valid_all = read(str(DATA_DIR / "valid_with_xtb.xyz"), ":")

    # Prepare validation sets (if not already done)
    vd_path = DATA_DIR / "valid_direct.xyz"
    if not vd_path.exists():
        write(str(vd_path), prepare_direct(valid_all), format="extxyz")
    vdel_path = DATA_DIR / "valid_delta.xyz"
    if not vdel_path.exists():
        write(str(vdel_path), prepare_delta(valid_all), format="extxyz")

    total = 0
    done = 0
    skipped = 0

    for N in DFT_BUDGETS:
        for seed in DATA_SEEDS:
            np.random.seed(seed)
            indices = np.random.choice(len(train_all), N, replace=False)
            subset = [train_all[i] for i in indices]

            for arm, prep_fn, valid_path in [
                ("direct", prepare_direct, str(vd_path)),
                ("delta", prepare_delta, str(vdel_path)),
            ]:
                name = f"{arm}_N{N}_s{seed}"
                total += 1
                if is_done(name):
                    print(f"  SKIP {name} (already done)", flush=True)
                    skipped += 1
                    continue

                train_path = DATA_DIR / f"{name}.xyz"
                if not train_path.exists():
                    write(str(train_path), prep_fn(subset), format="extxyz")

                ok = train(name, str(train_path), valid_path, seed, device)
                if ok:
                    done += 1

    # Full dataset reference
    name = "direct_full"
    total += 1
    if is_done(name):
        print(f"  SKIP {name} (already done)", flush=True)
        skipped += 1
    else:
        full_path = DATA_DIR / "direct_full.xyz"
        if not full_path.exists():
            write(str(full_path), prepare_direct(train_all), format="extxyz")
        ok = train(name, str(full_path), str(vd_path), 42, device)
        if ok:
            done += 1

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {done} trained, {skipped} skipped, {total} total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
