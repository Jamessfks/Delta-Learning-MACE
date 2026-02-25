# Prompt: Delta-Learning MACE Force Field Hypothesis Test

## For Cursor AI Agent — Systematic Experimental Validation

---

### SYSTEM CONTEXT

You are a computational chemistry research agent specializing in machine learning interatomic potentials. You have access to the MACE-Helper codebase which includes:
- `mace_train.py`: MACE training wrapper with reproducibility controls
- `mace_active_learning.py`: Committee disagreement-based active learning
- `model_disagreement.py`: Ensemble uncertainty quantification
- `inference_test.py`: Model evaluation and benchmarking
- `benchmark_report.py`: Automated report generation
- A validated fine-tuned MACE model for liquid water (0.208 meV/atom energy MAE, 2.032 meV/Å force MAE)
- Training data: `data/train.xyz`, `data/valid.xyz`, `data/pool.xyz`

---

### HYPOTHESIS UNDER TEST

**H₀ (Null)**: Direct MACE training on limited DFT data achieves equivalent or better accuracy than Δ-learning (MACE trained on XTB-to-DFT corrections) for the same DFT budget.

**H₁ (Alternative)**: Δ-learning MACE trained on XTB corrections achieves significantly lower error than direct MACE training, with the improvement growing as DFT data decreases.

**Specific Prediction**: At N_DFT ≤ 200 structures, Δ-MACE will achieve ≤ 50% of the force MAE of direct-MACE, because the correction field ΔE = E_DFT − E_XTB has lower intrinsic dimensionality than E_DFT.

---

### PHASE 1: PREREQUISITE VALIDATION — XTB Error Systematicity

**Objective**: Before testing the main hypothesis, verify the critical assumption that XTB errors on our target system are *systematic* (learnable) rather than *random* (not learnable).

**Protocol**:

```python
"""
Step 1: Run GFN2-xTB on all structures in data/valid.xyz
Step 2: Compute ΔE = E_DFT - E_XTB and ΔF = F_DFT - F_XTB for each structure
Step 3: Analyze error statistics and systematicity
"""

from ase.io import read
from xtb.ase.calculator import XTB
import numpy as np
from scipy.stats import pearsonr

structures = read("data/valid.xyz", ":")

delta_energies = []
delta_forces = []
xtb_energies = []
dft_energies = []

for atoms in structures:
    # DFT reference (already in atoms.info)
    e_dft = atoms.info["REF_energy"]
    f_dft = atoms.arrays["REF_forces"]

    # XTB calculation
    atoms.calc = XTB(method="GFN2-xTB")
    e_xtb = atoms.get_potential_energy()
    f_xtb = atoms.get_forces()

    delta_energies.append(e_dft - e_xtb)
    delta_forces.append(f_dft - f_xtb)
    xtb_energies.append(e_xtb)
    dft_energies.append(e_dft)

# Analysis
delta_E = np.array(delta_energies)
print(f"ΔE statistics:")
print(f"  Mean:  {np.mean(delta_E):.4f} eV")
print(f"  Std:   {np.std(delta_E):.4f} eV")
print(f"  Range: {np.ptp(delta_E):.4f} eV")
print(f"  CV (std/|mean|): {np.std(delta_E)/np.abs(np.mean(delta_E)):.4f}")

# CRITICAL CHECK: If CV < 0.3, errors are systematic (proceed)
# If CV > 1.0, errors are chaotic (abort delta-learning)

# Check correlation between ΔE and structural descriptors
# (e.g., number of H-bonds, coordination numbers)
```

**Decision Gate**:
- CV(ΔE) < 0.3 AND R²(E_DFT, E_XTB) > 0.9 → **PROCEED** to Phase 2
- CV(ΔE) > 1.0 OR R²(E_DFT, E_XTB) < 0.5 → **ABORT**: XTB errors are not systematic; use PBE→SCAN correction instead
- 0.3 < CV(ΔE) < 1.0 → **PROCEED WITH CAUTION**: errors are partially systematic

---

### PHASE 2: CONTROLLED EXPERIMENT — Direct vs. Δ-Learning

**Objective**: Compare three training strategies under identical DFT budgets using the existing liquid water dataset.

**Experimental Design**:

| Arm | Method | Training Targets | DFT Structures Used |
|-----|--------|-----------------|-------------------|
| A | Direct-MACE (full) | E_DFT, F_DFT | All available (~1400) |
| B | Direct-MACE (limited) | E_DFT, F_DFT | 50, 100, 200, 500 |
| C | Δ-MACE (limited) | ΔE, ΔF | Same 50, 100, 200, 500 |
| D | Δ-MACE + smoothness | ΔE, ΔF + L_smooth | Same + all XTB configs |

**Protocol**:

```python
"""
For each DFT budget N in [50, 100, 200, 500]:
    1. Randomly sample N structures from data/train.xyz (3 random seeds)
    2. For Arms A/B: Train MACE directly on (E_DFT, F_DFT) targets
    3. For Arm C: Compute ΔE, ΔF targets, train MACE on corrections
    4. For Arm D: Same as C + smoothness regularization from full XTB pool
    5. Evaluate all models on data/valid.xyz (held-out)
    6. Report energy MAE, force MAE, force cosine similarity, R²
"""

import subprocess
import json
from pathlib import Path

DFT_BUDGETS = [50, 100, 200, 500]
RANDOM_SEEDS = [42, 123, 7]
MACE_SEEDS = [0, 1, 2]  # For ensemble within each arm

for N in DFT_BUDGETS:
    for data_seed in RANDOM_SEEDS:
        # Sample N structures deterministically
        np.random.seed(data_seed)
        all_train = read("data/train.xyz", ":")
        indices = np.random.choice(len(all_train), N, replace=False)
        subset = [all_train[i] for i in indices]

        # --- ARM B: Direct-MACE ---
        write(f"data/direct_N{N}_s{data_seed}.xyz", subset)
        subprocess.run([
            "python", "mace_train.py",
            "--train_file", f"data/direct_N{N}_s{data_seed}.xyz",
            "--valid_file", "data/valid.xyz",
            "--work_dir", "runs/hypothesis_test",
            "--name", f"direct_N{N}_s{data_seed}",
            "--seed", str(data_seed),
            "--extra",
            "--max_num_epochs", "200",
            "--model", "MACE",
            "--hidden_irreps", "128x0e+128x1o+128x2e",
            "--r_max", "5.0",
            "--num_interactions", "2",
            "--correlation", "3",
        ])

        # --- ARM C: Δ-MACE ---
        # Compute corrections for sampled structures
        delta_subset = compute_corrections(subset)  # ΔE, ΔF
        write(f"data/delta_N{N}_s{data_seed}.xyz", delta_subset)
        subprocess.run([
            "python", "mace_train.py",
            "--train_file", f"data/delta_N{N}_s{data_seed}.xyz",
            "--valid_file", "data/valid_delta.xyz",  # Also corrected
            "--work_dir", "runs/hypothesis_test",
            "--name", f"delta_N{N}_s{data_seed}",
            "--seed", str(data_seed),
            "--extra",
            "--max_num_epochs", "200",
            "--model", "MACE",
            "--hidden_irreps", "128x0e+128x1o+128x2e",
            "--r_max", "5.0",
            "--num_interactions", "2",
            "--correlation", "3",
        ])
```

**Key Implementation Details**:

1. **Energy Reference Alignment** (Critical):
```python
def align_energy_references(structures):
    """
    Subtract per-element atomic energies from both DFT and XTB
    before computing corrections to remove basis-set-dependent offsets.
    """
    # Compute isolated atom energies for each element
    e0_dft = {"H": -13.587, "O": -432.241}  # From DFT pseudopotential
    e0_xtb = {}  # Compute from XTB isolated atom calculations

    for atoms in structures:
        n_atoms = {s: 0 for s in set(atoms.get_chemical_symbols())}
        for s in atoms.get_chemical_symbols():
            n_atoms[s] += 1

        e_dft_shifted = atoms.info["REF_energy"] - sum(e0_dft[s]*n for s,n in n_atoms.items())
        e_xtb_shifted = atoms.info["XTB_energy"] - sum(e0_xtb[s]*n for s,n in n_atoms.items())

        atoms.info["energy"] = e_dft_shifted - e_xtb_shifted  # This is ΔE
        atoms.arrays["forces"] = atoms.arrays["REF_forces"] - atoms.arrays["XTB_forces"]  # ΔF
```

2. **Inference Pipeline for Δ-MACE**:
```python
def delta_mace_predict(atoms, xtb_calc, mace_model):
    """
    E_predicted = E_XTB + ΔE_MACE
    F_predicted = F_XTB + ΔF_MACE
    """
    atoms.calc = xtb_calc
    e_xtb = atoms.get_potential_energy()
    f_xtb = atoms.get_forces()

    atoms.calc = mace_model
    delta_e = atoms.get_potential_energy()
    delta_f = atoms.get_forces()

    return e_xtb + delta_e, f_xtb + delta_f
```

---

### PHASE 3: STATISTICAL ANALYSIS

**Objective**: Determine if the difference between methods is statistically significant.

```python
"""
For each DFT budget N:
    - Compute mean and std of MAE across 3 data seeds × 3 model seeds
    - Perform paired t-test: H₀ (direct MAE = delta MAE)
    - Compute effect size (Cohen's d)
    - Plot learning curves: MAE vs N_DFT for each method
"""

from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

results = {}  # results[method][N] = [mae_values]

for N in DFT_BUDGETS:
    direct_maes = results["direct"][N]
    delta_maes = results["delta"][N]

    t_stat, p_value = ttest_rel(direct_maes, delta_maes)
    cohens_d = (np.mean(direct_maes) - np.mean(delta_maes)) / np.sqrt(
        (np.std(direct_maes)**2 + np.std(delta_maes)**2) / 2
    )

    print(f"N={N}: Direct MAE={np.mean(direct_maes):.3f}±{np.std(direct_maes):.3f}")
    print(f"       Delta  MAE={np.mean(delta_maes):.3f}±{np.std(delta_maes):.3f}")
    print(f"       p={p_value:.4f}, Cohen's d={cohens_d:.2f}")
    print(f"       Significant (p<0.05): {p_value < 0.05}")

# Learning curve plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for method, label, color in [
    ("direct", "Direct MACE", "blue"),
    ("delta", "Δ-MACE", "red"),
    ("delta_smooth", "Δ-MACE + Smoothness", "green"),
]:
    means = [np.mean(results[method][N]) for N in DFT_BUDGETS]
    stds = [np.std(results[method][N]) for N in DFT_BUDGETS]
    axes[0].errorbar(DFT_BUDGETS, means, yerr=stds, label=label, color=color,
                     marker='o', capsize=5)

axes[0].set_xlabel("Number of DFT Structures")
axes[0].set_ylabel("Force MAE (meV/Å)")
axes[0].set_title("Learning Curves: Force Accuracy vs DFT Budget")
axes[0].legend()
axes[0].set_xscale("log")
axes[0].set_yscale("log")

plt.tight_layout()
plt.savefig("testing_hypothesis/learning_curves.png", dpi=300)
```

---

### PHASE 4: ABLATION STUDY

**Objective**: Isolate the contribution of each component.

| Experiment | Delta-Learning | Smoothness Reg. | Curriculum Weighting | Active Learning |
|-----------|:-:|:-:|:-:|:-:|
| Baseline (direct) | | | | |
| +Δ only | x | | | |
| +Δ+smooth | x | x | | |
| +Δ+smooth+curriculum | x | x | x | |
| Full protocol | x | x | x | x |

For each ablation, use N_DFT = 200 (the regime where differences should be most visible).

---

### PHASE 5: REPORT TEMPLATE

Generate a final report with the following structure:

```
═══════════════════════════════════════════════════════════════════
  DELTA-LEARNING MACE HYPOTHESIS TEST — FINAL REPORT
  Date: [DATE]
  System: Liquid Water (64 H₂O, periodic)
  Reference: DFT [functional]
  Baseline: GFN2-xTB
═══════════════════════════════════════════════════════════════════

1. PREREQUISITE: XTB ERROR SYSTEMATICITY
─────────────────────────────────────────────
  XTB vs DFT energy correlation (R²):     [VALUE]
  ΔE coefficient of variation:             [VALUE]
  ΔE mean ± std:                           [VALUE] eV
  ΔF mean ± std:                           [VALUE] meV/Å
  Decision: [PROCEED / ABORT / CAUTION]

2. LEARNING CURVE RESULTS
─────────────────────────────────────────────
  N_DFT    Direct MAE (meV/Å)    Δ-MACE MAE (meV/Å)    Improvement
  ─────    ──────────────────     ───────────────────    ───────────
  50       [VALUE ± STD]          [VALUE ± STD]          [X]×
  100      [VALUE ± STD]          [VALUE ± STD]          [X]×
  200      [VALUE ± STD]          [VALUE ± STD]          [X]×
  500      [VALUE ± STD]          [VALUE ± STD]          [X]×

3. STATISTICAL SIGNIFICANCE
─────────────────────────────────────────────
  Paired t-test p-values:
    N=50:  p=[VALUE]  [SIGNIFICANT/NOT]
    N=100: p=[VALUE]  [SIGNIFICANT/NOT]
    N=200: p=[VALUE]  [SIGNIFICANT/NOT]
    N=500: p=[VALUE]  [SIGNIFICANT/NOT]

  Effect sizes (Cohen's d):
    N=50:  d=[VALUE]  [SMALL/MEDIUM/LARGE]
    N=100: d=[VALUE]  [SMALL/MEDIUM/LARGE]
    N=200: d=[VALUE]  [SMALL/MEDIUM/LARGE]
    N=500: d=[VALUE]  [SMALL/MEDIUM/LARGE]

4. ABLATION RESULTS (N_DFT = 200)
─────────────────────────────────────────────
  Component Added           Force MAE    Δ from Baseline
  ───────────────           ─────────    ───────────────
  Direct MACE (baseline)    [VALUE]      —
  + Δ-learning              [VALUE]      [-%]
  + smoothness reg.         [VALUE]      [-%]
  + curriculum weighting    [VALUE]      [-%]
  + active learning         [VALUE]      [-%]

5. CONCLUSION
─────────────────────────────────────────────
  Hypothesis [SUPPORTED / PARTIALLY SUPPORTED / REJECTED]:

  [If supported]:
  Delta-learning with MACE trained on XTB corrections achieves
  [X]× lower force MAE than direct training at N_DFT=[N], with
  statistical significance (p=[VALUE]). The improvement is largest
  in the low-data regime (N≤200), consistent with the theoretical
  prediction that the correction field has lower intrinsic
  dimensionality than absolute energies.

  [If rejected]:
  Direct MACE training matches or exceeds Δ-learning at all DFT
  budgets tested. Possible explanations: (a) XTB errors on this
  system are not sufficiently systematic, (b) MACE architecture
  is already efficient enough to learn from limited DFT directly,
  (c) the inference overhead of XTB does not justify marginal gains.

  Implications for microelectronics:
  [Assessment based on water results + XTB reliability analysis]

6. RECOMMENDATIONS
─────────────────────────────────────────────
  [Actionable next steps based on results]

═══════════════════════════════════════════════════════════════════
```

---

### EXECUTION NOTES

1. **Computational Budget**: Estimate ~4–8 GPU-hours for full experiment (4 budgets × 3 seeds × 4 arms × 200 epochs)
2. **XTB Dependency**: Requires `xtb-python` or `tblite` package. Install via: `pip install tblite`
3. **Validation Set**: Always use the same `data/valid.xyz` across all arms for fair comparison
4. **Random Seed Protocol**: Data sampling seeds (42, 123, 7) control *which* structures are selected. MACE seeds (0, 1, 2) control network initialization. Both are varied independently.
5. **Early Stopping**: Use patience=30 on validation force MAE to prevent overfitting, especially critical at small N_DFT

---

### EXPECTED OUTCOMES

Based on literature precedent:
- **N=50**: Δ-MACE should show 2–5× improvement over direct (largest gap)
- **N=100**: Δ-MACE should show 1.5–3× improvement
- **N=200**: Δ-MACE should show 1.2–2× improvement
- **N=500**: Gap narrows; direct MACE may approach Δ-MACE performance

If these ranges are not observed, investigate:
1. Whether energy reference alignment was done correctly
2. Whether XTB errors are truly systematic on your system
3. Whether the MACE model capacity is appropriate for corrections (may need smaller model for ΔE than for E_DFT)
