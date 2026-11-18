# Katabatic manuscript analysis pipeline

This folder reproduces the Peyto Glacier (2008) and Athabasca Glacier (2015) turbulent-flux, katabatic-transfer, and surface-ablation analysis. Verified 18 November 2026 against the November 17 v0.3.0 code and v0.2.0 dataset snapshots.

## Folder structure

- `data/raw/peyto/` — 11 Peyto input files (four profile workbooks, EC flux products, radiation forcing, Qn/Qp, SR50, Profile B)
- `data/raw/athabasca/` — 6 Athabasca input files (profile, EC, ice-AWS, SR50, Qn/Qp)
- `scripts/` — processing, calibration, sensitivity, figure-generation scripts and `src/` physics/readers
- `run_pipeline.py` — runs stages in their tested order
- `requirements.txt` — software dependencies

The pipeline creates `data/interim/`, `tables/`, and `figures/` when run. Those outputs need not be stored as input files.

## How to run

From this folder in a terminal (Python 3.11+):

```bash
python -m pip install -r requirements.txt
python run_pipeline.py --check
python run_pipeline.py --stage all
python -m unittest discover -s tests -v  # only available in the original release, not this simplified folder
```

The last command is optional and belongs to the source release's additional test folder. The full local source-release integration run and its four regression tests passed on 18 November 2026.

## Scripts in execution order

| File | Purpose |
|---|---|
| `01_validate_inputs.py` | Check all raw inputs |
| `02_calibrate_katabatic.py` | EC-based background Kb calibration |
| `03_build_flux_scenarios.py` | Recalculate bulk/katabatic flux scenarios |
| `04_align_sr50_observations.py` | Align measured SR50 surface lowering |
| `05_kb_gamma_sensitivity.py` | Kb/lapse-rate sensitivity grids |
| `07_compile_manuscript_tables.py` | Flux and original analysis diagnostics |
| `08_partition_surface_ablation.py` | Final phase-consistent melt/sublimation and summary numbers |
| `fig02_...py`, `fig04_...py` to `fig09_...py` | Regenerate manuscript figures 2 and 4–9 |
| `06_bulk_transfer_sensitivity.py` | Optional *historical* diagnostic; not the main Figure 9 |
| `pipeline_common.py`, `project_config.py`, `src/` | Shared logic, path configuration and physical models |

## Included data

The 17 input files are versioned directly in `data/raw/` so GitHub users can reproduce the pipeline without obtaining a second archive. This release includes six Athabasca inputs and eleven Peyto inputs. The `.gitignore` excludes regenerated outputs, not the input data.

