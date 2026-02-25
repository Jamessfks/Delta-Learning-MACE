# Fix Prompt: Periodic Boundary Condition Issue in Delta-Learning Experiment

## Context for the AI Agent

You are working on the MACE-Helper project at `/home/james/MACE-Helper/`. We are testing whether delta-learning (training MACE to predict corrections from XTB to DFT, rather than training directly on DFT) improves data efficiency for ML force fields.

**The experiment ran but produced a scientifically invalid result because of a critical bug: XTB was run non-periodically on structures from periodic DFT simulations.** This inflated XTB energy variance by 38.5x, destroyed the energy correlation (R²=0.068), and made delta-learning appear 300x worse than direct training — but this comparison is meaningless because the XTB baseline was wrong.

Your job is to fix the XTB calculator, re-run Phase 1 analysis, regenerate the XTB-augmented datasets, and re-run the Phase 2 training experiment. You must also implement per-element atomic energy subtraction, which the original code skipped.

---

## Scientific Background: Why This Fix Matters (from arXiv literature)

The following arXiv papers establish that (a) periodicity consistency is essential, (b) delta-learning works when the baseline is correct, and (c) energy referencing is critical in multi-fidelity setups.

### 1. Delta-learning requires a physically consistent baseline

**Böselt et al., arXiv:2010.11610** — "Machine Learning in QM/MM Molecular Dynamics Simulations of Condensed-Phase Systems" demonstrated delta-learning (DFTB→DFT corrections) for condensed-phase systems. Key finding: the delta-learning scheme reached DFT accuracy while requiring significantly fewer parameters than direct learning, **but only when the baseline (DFTB) used consistent boundary conditions with the target (DFT)**. They explicitly used electrostatic embedding with the MM environment to maintain physical consistency. Running a non-periodic semi-empirical calculation on a periodic system would violate this principle.

### 2. Implicit Delta Learning achieves 50x data reduction with semi-empirical baselines

**Thaler et al., arXiv:2412.06064** — "Implicit Delta Learning of High Fidelity Neural Network Potentials" introduced IDLe, which leverages cheaper semi-empirical QM computations. IDLe achieves the same accuracy as single high-fidelity baselines while using **up to 50x less high-fidelity data**. Crucially, they provide 11 million semi-empirical QM calculations as training data — demonstrating that when the semi-empirical method is applied consistently, the approach works at scale.

### 3. Multi-fidelity M3GNet achieves 8x data efficiency (water and silicon)

**Ko & Ong, arXiv:2409.00957** — "Data-Efficient Construction of High-Fidelity Graph Deep Learning Interatomic Potentials" showed that multi-fidelity M3GNet trained on GGA+10% SCAN achieves accuracy comparable to 8x more SCAN data alone. **They tested on water and silicon** — directly relevant to our system. Critically, they used consistent periodic DFT calculations at both fidelity levels.

### 4. Cluster-to-bulk transfer requires careful energy referencing

**Gawkowski et al., arXiv:2509.16601** — "The Good, the Bad, and the Ugly of Atomistic Learning for Clusters-to-Bulk Generalization" directly addresses our failure mode. They found that transferring accuracy from clusters (non-periodic) to bulk (periodic) **requires regularization** and that training only on energies without forces "introduces artefacts: stable trajectories and low energy errors conceal large force errors." This is exactly what we observed: our non-periodic XTB cluster energies cannot transfer to periodic DFT targets.

### 5. Energy referencing is critical for cross-fidelity transfer

**Huang et al., arXiv:2504.05565** — "Cross-functional transferability in universal machine learning interatomic potentials" showed that "significant energy scale shifts and poor correlations between GGA and r²SCAN pose challenges to cross-functional data transferability." They demonstrated that **elemental energy referencing is essential** for transfer learning in universal MLIPs. Our R²=0.068 is a direct consequence of missing this step.

### 6. Multi-fidelity training outperforms delta-learning when done correctly

**Kim et al., arXiv:2409.07947** — "Data-efficient multi-fidelity training for high-fidelity machine learning interatomic potentials" demonstrated that multi-fidelity learning is **more effective than transfer learning or Δ-learning** for certain systems, and that geometric/compositional spaces not covered by high-fidelity data can be inferred from low-fidelity data. This suggests our experiment should compare delta-learning against multi-fidelity training as an additional arm.

### 7. GFN1-xTB validated for periodic systems

**Vicent-Luna et al., arXiv:2104.01738** — "Efficient Computation of Metal Halide Perovskites Properties using GFN1-xTB" benchmarked GFN1-xTB for periodic systems (perovskites) against DFT, showing it produces accurate structural and vibrational properties comparable to DFT. This confirms GFN1-xTB is suitable as a periodic baseline method.

**Komissarov & Verstraelen, arXiv:2109.10416** — "Improving the Silicon Interactions of GFN-xTB" showed that GFN1-xTB's silicon parameters needed re-fitting for organosilicon compounds, but the method itself works for periodic silicon-containing systems. Relevant for the microelectronics target domain.

### 8. Delta-learning for periodic force fields at coupled-cluster level

**Schönbauer et al., arXiv:2507.06929** — "Machine-Learned Force Fields for Lattice Dynamics at Coupled-Cluster Level Accuracy" used delta-learning (CC-DFT corrections) for periodic solids (diamond, LiH). They explicitly note that delta-learning overcomes limitations from long-range effects in periodic systems. This validates our approach for periodic systems, but only when both levels of theory use PBC.

---

## The Bug

File: `testing_hypothesis/xtb_calculator.py`, line 37:

```python
a.pbc = False  # <-- THIS IS THE BUG
```

The training data (`data/train.xyz`, `data/valid.xyz`) contains 192-atom periodic liquid water structures (64 H₂O in a ~24 Å cubic box, `pbc=[True, True, True]`). Setting `pbc=False` strips the periodicity, creating a molecular cluster in vacuum. This causes:

1. **Energy**: XTB energy spread = 125.4 eV vs DFT spread = 2.74 eV (38.5x inflation from surface effects)
2. **Energy offset**: ΔE mean = 7203 eV (37,517 meV/atom) — an enormous, meaningless offset
3. **Energy correlation**: R²(E_DFT, E_XTB) = 0.068 — essentially random
4. **Forces**: Less affected (forces are local), but still contaminated by edge atoms

---

## The Fix: Two Options (Implement Both, Use Best)

### Option A: GFN1-xTB with Periodic Boundaries (CONFIRMED WORKING)

I have verified that `xtb` CLI with GFN1-xTB supports periodic systems via POSCAR input format. GFN2-xTB does NOT support PBC (fails with "Multipoles not available with PBC"). Test results:

- `xtb POSCAR --gfn 1 --grad` → **SUCCESS** (periodic, ~4.6s per structure)
- `xtb POSCAR --gfn 2 --grad` → **FAILS** ("Multipoles not available with PBC")
- `xtb POSCAR --gfn 0 --grad` → **FAILS**

GFN1-xTB periodic energy for struct 0: -9395.2323 eV
Non-periodic energy for same struct: -9439.4954 eV
Difference: 44.26 eV (230 meV/atom) — this is the PBC error magnitude.

### Option B: GFN2-xTB via tblite (NEEDS INSTALLATION FIX)

The `tblite` library supports GFN2-xTB with PBC, but the Python bindings have a shared library mismatch on this system. The conda package `tblite 0.4.0` is installed but `tblite-python` fails to link (`undefined symbol: tblite_new_gb_solvation_epsilon`). If you can fix this, GFN2-xTB periodic would be preferable since it is more accurate than GFN1.

**Decision**: Implement Option A (GFN1-xTB periodic) as the primary path. If you can get tblite working, add it as Option B and run both for comparison.

---

## Step-by-Step Instructions

### Step 1: Fix `testing_hypothesis/xtb_calculator.py`

Rewrite `run_xtb_single()` to:

1. Preserve PBC from the input atoms (`atoms.pbc` is already `[True, True, True]`)
2. Write as VASP POSCAR format (carries lattice vectors) instead of XYZ format
3. Use `--gfn 1` instead of `--gfn 2` (GFN2 does not support PBC)
4. Remove `--etemp 1000` (elevated electronic temperature was a workaround for non-periodic SCF convergence and should not be needed with proper PBC)
5. Parse energy and forces the same way (the gradient file format is identical)

Key implementation details:
- Use `from ase.io import write` with `format='vasp'` to write POSCAR. ASE handles the lattice vectors correctly.
- The XTB CLI automatically detects periodicity from the POSCAR lattice. You do NOT need a `--periodic` flag — it's inferred from the input format.
- Keep the gradient parsing (`_parse_gradient`) as-is; gradient file format is the same for periodic calculations.
- Keep `run_xtb_batch()` as-is; it just calls `run_xtb_single()` in parallel.

Test your fix by running on 3 structures from `data/valid.xyz` and verifying:
- No errors
- Energies are ~-9400 eV range (periodic), NOT ~-8300 eV (non-periodic)
- Forces have magnitude ~0.1-2 eV/Å (reasonable for liquid water)

### Step 2: Implement Per-Element Atomic Energy Subtraction

This is critical and was skipped in the original code. The hypothesis paper (Section 4.1, Equation 7) specifies:

```
ΔE_i = (E_DFT,i - Σ_α E^DFT_atom,α) - (E_XTB,i - Σ_α E^XTB_atom,α)
```

Without this, the correction ΔE contains a huge constant offset (~8300 eV) that is meaningless and dominates the learning signal.

Implementation:
1. Compute isolated atom energies for H and O from both DFT and XTB (GFN1-xTB)
2. For DFT: the `--E0s average` flag in MACE training computes these automatically, but for delta targets we need explicit values. Compute from the training data: fit `E_DFT = N_H * e0_H + N_O * e0_O + interaction_energy` using least squares, or use the values MACE reports in its training logs.
3. For XTB: compute `xtb` on isolated H and O atoms (non-periodic is fine for single atoms): `xtb H.xyz --gfn 1 --grad` and `xtb O.xyz --gfn 1 --grad`
4. Subtract before computing corrections:
   ```python
   e_dft_shifted = e_dft - (n_H * e0_dft_H + n_O * e0_dft_O)
   e_xtb_shifted = e_xtb - (n_H * e0_xtb_H + n_O * e0_xtb_O)
   delta_e = e_dft_shifted - e_xtb_shifted
   ```

After subtraction, ΔE should be O(1-10 eV) total, or O(5-50 meV/atom), NOT O(7000 eV).

### Step 3: Re-run Phase 1 Analysis

Run the updated `phase1_xtb_systematicity.py` with periodic XTB. This will:
1. Recompute XTB energies and forces for all 159 valid + 1484 train structures
2. Regenerate `data/train_with_xtb.xyz` and `data/valid_with_xtb.xyz`
3. Recompute R²(E_DFT, E_XTB) — this should now be >> 0.9 if XTB is working properly
4. Recompute CV(ΔE) — should remain < 0.3
5. Update `testing_hypothesis/results/phase1_analysis.json` and `phase1_report.txt`

**Expected runtime**: ~1484 structures × ~5s/structure ÷ 5 parallel = ~25 minutes for training set. Validation set ~3 minutes.

**Verification checkpoint**: Before proceeding to Phase 2, print and check:
- R²(E_DFT, E_XTB) > 0.5 (ideally > 0.8)
- CV(ΔE/atom) < 0.3
- ΔE mean should be O(1-10 eV), NOT O(7000 eV)
- If R² is still very low, something else is wrong — stop and investigate.

### Step 4: Clean Up Old Results and Re-run Phase 2

1. Delete all old model directories under `runs/hypothesis_test/` — they were trained on invalid data:
   ```bash
   rm -rf runs/hypothesis_test/direct_N50_* runs/hypothesis_test/delta_N50_*
   rm -rf runs/hypothesis_test/direct_N100_* runs/hypothesis_test/delta_N100_*
   rm -rf runs/hypothesis_test/direct_full
   ```

2. Delete old intermediate data files:
   ```bash
   rm -f data/direct_N*.xyz data/delta_N*.xyz data/direct_full.xyz
   rm -f data/valid_direct.xyz data/valid_delta.xyz
   ```

3. Update `phase2_experiment.py` `prepare_delta_dataset()` to use the per-element shifted energies for computing corrections (Step 2 above). The forces do not need shifting (force corrections ΔF = F_DFT - F_XTB are already correct).

4. Run Phase 2:
   ```bash
   cd /home/james/MACE-Helper
   python testing_hypothesis/phase2_experiment.py --device cuda --budgets 50 100 200 500 --seeds 42 123 7
   ```

   This trains 4 budgets × 3 seeds × 2 arms = 24 models + 1 full reference = 25 models total.
   Estimated time: ~25 × 20 min = ~8 hours.

   If time is limited, start with `--budgets 50 100 --seeds 42` (4 models, ~80 min) to verify the fix works before committing to the full run.

### Step 5: Evaluate and Generate Report

After training completes, run evaluation:
```bash
python testing_hypothesis/phase2_experiment.py --skip-training --device cuda
```

This populates `testing_hypothesis/results/phase2_results.json` with metrics for all models.

### Step 6: Compare Old vs New Results

Print a comparison table showing how the fix changed the outcome. The key comparison:

| Metric | Old (non-periodic XTB) | New (periodic XTB) |
|--------|----------------------|-------------------|
| R²(E_DFT, E_XTB) | 0.068 | Should be >> 0.5 |
| ΔE mean | 7203 eV | Should be O(1-10 eV) |
| Direct N=50 force MAE | 9.2 meV/Å | ~similar |
| Delta N=50 force MAE | 45 meV/Å | Should be < 20 meV/Å |

---

## Important Notes

1. **GFN1 vs GFN2 — why GFN1 is correct here**: GFN2-xTB uses multipole electrostatics not implemented for periodic systems in the `xtb` CLI (confirmed error: "Multipoles not available with PBC"). GFN1-xTB uses a simpler Coulomb treatment that supports PBC (confirmed working via POSCAR input). For delta-learning, *physics-consistent* errors matter more than baseline accuracy. GFN1 with correct PBC is categorically better than GFN2 without PBC. Vicent-Luna et al. (arXiv:2104.01738) validated GFN1-xTB for periodic systems. Schönbauer et al. (arXiv:2507.06929) showed delta-learning works for periodic solids when both methods use PBC.

2. **POSCAR format — verified behavior**: ASE `write(path, atoms, format='vasp')` writes lattice vectors from `atoms.cell`. The `xtb` binary auto-detects periodicity from POSCAR — no `--periodic` flag needed. Verified: the first structure produces a ~24 Å cubic cell in the POSCAR header, and XTB correctly processes all 192 atoms.

3. **Gradient parsing is format-invariant**: The `gradient` file format is identical for periodic and non-periodic XTB: `cycle` header, N coordinate lines, N gradient lines (Hartree/Bohr). Existing `_parse_gradient()` works without changes. Verified on periodic GFN1 output.

4. **Timing**: Periodic GFN1 on 192-atom water takes ~4.6 s/structure (verified). Increase timeout to 600s. Total Phase 1: ~1484 × 5s ÷ 5 parallel ≈ 25 min for training set.

5. **Memory**: Keep `OMP_STACKSIZE=4G` and `OMP_NUM_THREADS=4` per job. With `n_parallel=5`, this uses ~20 threads and ~20 GB stack.

6. **Energy reference subtraction is scientifically essential**: Huang et al. (arXiv:2504.05565) showed "significant energy scale shifts" between methods destroy cross-fidelity transferability unless elemental energy references are subtracted. Yoo et al. (arXiv:1903.04366) demonstrated that incorrect atomic energy mapping leads to "ad hoc" mappings where total energy appears trained but per-atom decomposition is wrong. Our 7203 eV correction offset is precisely this failure mode.

7. **Validation checkpoint**: After fixing xtb_calculator.py, run its `__main__` block on 3 structures from `data/valid.xyz`. Verify:
   - Energies are in the ~-9400 eV range (periodic GFN1), NOT ~-8300 eV (non-periodic GFN2)
   - Forces have magnitude ~0.1-2 eV/Å
   - No convergence errors

8. **Do not modify** `mace_train.py`, `mace_active_learning.py`, `model_disagreement.py`, or `inference_test.py`. Only modify files inside `testing_hypothesis/`.

9. **File outputs to preserve**: Save old Phase 1 results before overwriting:
   ```bash
   cp testing_hypothesis/results/phase1_analysis.json testing_hypothesis/results/phase1_analysis_OLD_nonperiodic.json
   cp testing_hypothesis/results/phase1_report.txt testing_hypothesis/results/phase1_report_OLD_nonperiodic.txt
   ```

10. **Interpreting Phase 2 results in context of literature**: Based on Thaler et al. (arXiv:2412.06064), delta-learning can achieve comparable accuracy with up to 50x less high-fidelity data. Based on Ko & Ong (arXiv:2409.00957), multi-fidelity approaches achieve ~8x data efficiency on water. If delta-MACE does not outperform direct-MACE at N=50, the hypothesis is weakened even with correct PBC. If it outperforms at N=50 but not at N=500, this confirms the literature expectation that delta-learning's advantage is strongest in the low-data regime.

---

## arXiv References Used in This Prompt

| Paper ID | Authors | Title | Key Relevance |
|----------|---------|-------|---------------|
| 2010.11610 | Böselt et al. | ML in QM/MM MD of Condensed-Phase Systems | Delta-learning for condensed phase requires consistent boundaries |
| 2412.06064 | Thaler et al. | Implicit Delta Learning of High Fidelity NNPs | 50x data reduction with semi-empirical baselines |
| 2409.00957 | Ko & Ong | Data-Efficient Multi-Fidelity Graph Deep Learning Potentials | 8x efficiency on water with GGA+SCAN |
| 2509.16601 | Gawkowski et al. | Clusters-to-Bulk Generalization | Non-periodic→periodic transfer creates artefacts |
| 2504.05565 | Huang et al. | Cross-functional Transferability in uMLIPs | Elemental energy referencing is essential |
| 2409.07947 | Kim et al. | Data-efficient Multi-fidelity Training for MLIPs | Multi-fidelity outperforms Δ-learning for some systems |
| 2104.01738 | Vicent-Luna et al. | GFN1-xTB for Metal Halide Perovskites | GFN1-xTB validated for periodic systems |
| 2109.10416 | Komissarov & Verstraelen | Improving Silicon Interactions of GFN-xTB | GFN1-xTB silicon parameters; periodic support confirmed |
| 2507.06929 | Schönbauer et al. | MLFFs for Lattice Dynamics at CC Accuracy | Delta-learning for periodic solids (diamond, LiH) |
| 1903.04366 | Yoo et al. | Atomic Energy Mapping of Neural Network Potential | Incorrect atomic energy mapping leads to ad hoc results |
