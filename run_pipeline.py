#!/usr/bin/env python3
"""Run the manuscript reproduction workflow using a portable repository layout.

Usage:
    python run_pipeline.py --check
    python run_pipeline.py --stage core
    python run_pipeline.py --stage ablation
    python run_pipeline.py --stage figures
    python run_pipeline.py --stage all
    python run_pipeline.py --root /path/to/separate/data-root --stage all

The equations and scenario choices live in the original scientific scripts.
This runner controls execution order and paths only.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent

CORE = [
    "01_validate_inputs.py", "02_calibrate_katabatic.py", "03_build_flux_scenarios.py",
    "04_align_sr50_observations.py", "05_kb_gamma_sensitivity.py",
    "07_compile_manuscript_tables.py", "08_partition_surface_ablation.py",
]
FIGURES = [
    "fig02_meteorological_profiles.py", "fig04_turbulent_flux_comparison.py",
    "fig05_energy_partition.py", "fig06_cumulative_ablation.py",
    "fig07_ablation_timeseries.py", "fig08_kb_gamma_sensitivity.py",
    "fig09_bulk_transfer_sensitivity.py",
]
# This is an earlier pre-ablation sensitivity diagnostic. Main Figure 9
# is created with the corrected Code-12-consistent fig09_* script.
LEGACY_DIAGNOSTICS = ["06_bulk_transfer_sensitivity.py"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO, help="Data/output root (default: repository directory)")
    p.add_argument("--stage", choices=["all", "core", "ablation", "figures", "legacy-diagnostics"],
                   default="all", help="What to execute; ablation runs core stages")
    p.add_argument("--check", action="store_true", help="Validate required input files without computing outputs")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    env = os.environ.copy()
    env["KATABATIC_ROOT"] = str(root)
    env.setdefault("MPLBACKEND", "Agg")
    for d in [root / "data" / "interim", root / "tables", root / "figures"]:
        if not args.dry_run:
            d.mkdir(parents=True, exist_ok=True)
    if args.check:
        filenames=["01_validate_inputs.py"]
    elif args.stage in ("core","ablation"):
        filenames=CORE
    elif args.stage == "figures":
        filenames=FIGURES
    elif args.stage == "legacy-diagnostics":
        filenames=LEGACY_DIAGNOSTICS
    else:
        filenames=CORE+FIGURES

    for filename in filenames:
        script=REPO / "scripts" / filename
        if not script.is_file():
            raise FileNotFoundError(script)
        cmd=[sys.executable, str(script)]
        print("$", ' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO, env=env, check=True)
    print("Pipeline stage completed:", "check" if args.check else args.stage, flush=True)

if __name__ == "__main__":
    main()
