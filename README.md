# Delta-Learning MACE Hypothesis Test — Complete Package

Self-contained folder with all data, models, results, and scripts from the hypothesis test run (2026-02-25).

## Contents

```
hypothesis_test_complete/
├── data/                 # DFT + XTB-augmented structures
│   ├── train.xyz         # Original DFT training (1484 structures)
│   ├── valid.xyz         # Original DFT validation (159 structures)
│   ├── train_with_xtb.xyz
│   ├── valid_with_xtb.xyz
│   ├── valid_direct.xyz
│   ├── valid_delta.xyz
│   ├── direct_full.xyz
│   ├── direct_N{50,100,200,500}_s{42,123,7}.xyz
│   └── delta_N{50,100,200,500}_s{42,123,7}.xyz
├── models/               # 25 trained MACE models
│   ├── direct_N50_s42/
│   ├── delta_N50_s42/
│   ├── ... (all 24 budgeted models)
│   └── direct_full/
├── results/              # Phase outputs
│   ├── phase1_analysis.json
│   ├── phase2_results.json
│   ├── phase3_analysis.json
│   ├── master_log.txt
│   └── ...
├── figures/              # Learning curves, improvement bars
├── run_all.py            # Master pipeline
├── phase1_xtb_systematicity.py
├── phase2_experiment.py
├── phase3_analysis.py
├── phase5_final_report.py
├── xtb_calculator.py
├── resume_training.py
├── FINAL_REPORT.txt
└── README.md
```

## Requirements

- Python 3.10+
- ASE, numpy, scipy, matplotlib
- MACE (`mace-torch`)
- XTB (`xtb-python` or `xtb` binary) — for Phase 1 only
- CUDA GPU — for training/evaluation

## Usage

**Re-run full pipeline** (skips Phase 1 if XTB data exists, skips training if models exist):

```bash
cd hypothesis_test_complete
python run_all.py [cuda|cpu]
```

**Re-run evaluation only** (Phase 2 eval + Phase 3 + Phase 5):

```bash
python phase2_experiment.py --skip-training --device cuda
python phase3_analysis.py
python phase5_final_report.py
```

**Resume training** (trains only models that aren't done):

```bash
python resume_training.py [cuda|cpu]
```

## Result Summary

Hypothesis **REJECTED** for liquid water: Direct MACE outperforms Δ-MACE at all budgets (50, 100, 200, 500 structures). See `FINAL_REPORT.txt` for details.
