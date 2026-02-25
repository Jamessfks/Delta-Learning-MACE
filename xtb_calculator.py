#!/usr/bin/env python3
"""
XTB Calculator Wrapper — Periodic GFN1-xTB via CLI

Uses VASP POSCAR format to pass lattice vectors to the xtb binary,
which auto-detects periodicity. GFN2-xTB does NOT support PBC
("Multipoles not available with PBC"), so GFN1-xTB is used.

References:
  - Vicent-Luna et al. (arXiv:2104.01738): GFN1-xTB validated for periodic systems
  - Schönbauer et al. (arXiv:2507.06929): Delta-learning for periodic solids requires PBC consistency
  - Gawkowski et al. (arXiv:2509.16601): Non-periodic→periodic transfer creates artefacts
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177249


def run_xtb_single(atoms: Atoms, method: str = "GFN1-xTB", n_threads: int = 4,
                    timeout: int = 600) -> tuple[float, np.ndarray]:
    """
    Run GFN1-xTB on a single structure with periodic boundary conditions.

    Writes VASP POSCAR format so that xtb auto-detects the lattice.
    Uses progressive electronic temperature (Fermi smearing) for SCF
    convergence on dense liquid systems — starts at 500K, retries at
    1000K and 2000K if SCF fails. Justified by Niklasson (arXiv:2003.09050):
    elevated electronic temperature aids convergence when the HOMO-LUMO
    gap is small, as in condensed-phase water configurations.
    """
    a = atoms.copy()
    cell = a.cell.array.copy()
    cell[np.abs(cell) < 1e-3] = 0.0
    a.set_cell(cell)

    sort_idx = np.argsort(a.get_atomic_numbers())
    unsort_idx = np.argsort(sort_idx)
    a = a[sort_idx]

    gfn_level = "1"
    if "GFN2" in method.upper():
        gfn_level = "2"
    elif "GFN0" in method.upper():
        gfn_level = "0"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)
    env["OMP_STACKSIZE"] = "4G"

    for etemp in [500, 1000, 2000, 4000]:
        with tempfile.TemporaryDirectory() as td:
            coord_file = os.path.join(td, "POSCAR")
            write(coord_file, a, format="vasp", sort=True)

            cmd = [
                "xtb", coord_file, "--gfn", gfn_level, "--grad",
                "--iterations", "500", "--etemp", str(etemp),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=td,
                timeout=timeout, env=env
            )

            if result.returncode == 0:
                energy_ev = _parse_energy(result.stdout)
                forces_sorted = _parse_gradient(os.path.join(td, "gradient"), len(a))
                forces_ev_ang = forces_sorted[unsort_idx]
                return energy_ev, forces_ev_ang

    error_msg = ""
    for line in result.stdout.split("\n"):
        if "error" in line.lower() or "converge" in line.lower():
            error_msg += line.strip() + "; "
    raise RuntimeError(f"XTB failed after all etemp retries: {error_msg or result.stderr[-300:]}")


def _parse_energy(stdout: str) -> float:
    """Extract total energy in eV from XTB stdout."""
    for line in stdout.split("\n"):
        if "TOTAL ENERGY" in line:
            parts = line.split()
            return float(parts[3]) * HA_TO_EV
    raise ValueError("Could not parse energy from XTB output")


def _parse_gradient(grad_file: str, n_atoms: int) -> np.ndarray:
    """
    Parse the gradient file and convert to forces in eV/Å.
    Gradient = -force, in Hartree/Bohr.
    """
    with open(grad_file) as f:
        lines = f.readlines()

    grad_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("cycle"):
            grad_start = i + 1 + n_atoms
            break

    if grad_start is None:
        raise ValueError("Could not find gradient data in file")

    gradients = []
    for i in range(grad_start, grad_start + n_atoms):
        parts = lines[i].split()
        gx, gy, gz = float(parts[0]), float(parts[1]), float(parts[2])
        gradients.append([gx, gy, gz])

    gradients = np.array(gradients)
    forces = -gradients * HA_TO_EV / BOHR_TO_ANG

    return forces


def compute_isolated_atom_energy_xtb(symbol: str, method: str = "GFN1-xTB") -> float:
    """Compute isolated atom energy using XTB (non-periodic, single atom)."""
    atom = Atoms(symbol, positions=[[0, 0, 0]])

    with tempfile.TemporaryDirectory() as td:
        coord_file = os.path.join(td, "input.xyz")
        write(coord_file, atom, format="xyz")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OMP_STACKSIZE"] = "1G"

        gfn_level = "1"
        if "GFN2" in method.upper():
            gfn_level = "2"

        result = subprocess.run(
            ["xtb", coord_file, "--gfn", gfn_level, "--grad", "--uhf", "0"],
            capture_output=True, text=True, cwd=td, timeout=60, env=env
        )

        if result.returncode != 0:
            return 0.0

        return _parse_energy(result.stdout)


def run_xtb_batch(atoms_list: list, method: str = "GFN1-xTB",
                  n_parallel: int = 5, n_threads_per_job: int = 4,
                  timeout: int = 600) -> list[tuple[float, np.ndarray]]:
    """
    Run XTB on a batch of structures with thread-level parallelism.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    results = [None] * len(atoms_list)
    n = len(atoms_list)
    t0 = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {}
        for i, atoms in enumerate(atoms_list):
            future = executor.submit(run_xtb_single, atoms, method, n_threads_per_job, timeout)
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                e, f = future.result()
                results[idx] = (e, f)
                completed += 1
            except Exception as ex:
                print(f"  WARNING: Structure {idx} failed: {ex}")
                results[idx] = None
                failed += 1

            done = completed + failed
            if done % 20 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(f"  XTB progress: {done}/{n} ({completed} ok, {failed} failed) "
                      f"[{rate:.1f}/s, ETA {eta:.0f}s]", flush=True)

    return results


if __name__ == "__main__":
    from ase.io import read
    import numpy as np

    data_dir = Path(__file__).resolve().parent / "data"
    structures = read(str(data_dir / "valid.xyz"), ":")[:3]
    print(f"Testing periodic GFN1-xTB on {len(structures)} structures...")

    for i, atoms in enumerate(structures):
        e, f = run_xtb_single(atoms)
        e_dft = atoms.info["TotEnergy"]
        f_dft = atoms.arrays["force"]
        de = e_dft - e
        df_mae = np.mean(np.abs(f_dft - f)) * 1000
        print(f"  Struct {i}: E_XTB={e:.2f} eV, E_DFT={e_dft:.4f} eV, "
              f"ΔE={de:.2f} eV ({de/len(atoms)*1000:.1f} meV/at), "
              f"ΔF_MAE={df_mae:.1f} meV/Å")

    print("\nIsolated atom energies:")
    for sym in ["H", "O"]:
        e0 = compute_isolated_atom_energy_xtb(sym)
        print(f"  {sym}: {e0:.6f} eV")
