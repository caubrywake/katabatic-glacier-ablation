"""Repository-relative paths for the katabatic manuscript analysis.

By default all data and outputs live in the repository containing this module.
For an external data/output root, set the KATABATIC_ROOT environment variable.
This module changes paths only; it does not alter scientific calculations.
"""
from __future__ import annotations
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("KATABATIC_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PEYTO_RAW = RAW_ROOT / "peyto"
ATHA_RAW = RAW_ROOT / "athabasca"
INTERIM = PROJECT_ROOT / "data" / "interim"
TABLES = PROJECT_ROOT / "tables"
FIGURES = PROJECT_ROOT / "figures"


def make_output_dirs() -> None:
    """Create output directories without manufacturing missing input files."""
    for p in (INTERIM, TABLES, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
