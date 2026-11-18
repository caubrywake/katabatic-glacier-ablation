"""Shared configuration and QC rules for the final katabatic manuscript pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd

from project_config import PROJECT_ROOT, PEYTO_RAW, ATHA_RAW, INTERIM, TABLES, FIGURES

# Final manuscript QC decision: this last-hour Peyto latent-heat value is an
# obvious end-of-record outlier in the corrected EC series (226.325 W m-2).
PEYTO_BAD_QE_TIMES = [pd.Timestamp("2008-08-25 00:00:00")]


def apply_peyto_ec_qc(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply final Peyto EC QC and a common Qh/Qe observation mask.

    Rules
    -----
    1. Set the known 2008-08-25 00:00 AdjEC2QE outlier to NaN.
    2. Retain EC observations only where BOTH EC2QH and AdjEC2QE are finite.
       Both columns are set to NaN outside this common mask.

    This ensures Peyto Qh and Qe calibration/evaluation use the same hours.
    Model-only time series remain unchanged.
    """
    out = df.copy()
    for t in PEYTO_BAD_QE_TIMES:
        if t in out.index:
            out.loc[t, "AdjEC2QE"] = np.nan

    qh = pd.to_numeric(out["EC2QH"], errors="coerce")
    qe = pd.to_numeric(out["AdjEC2QE"], errors="coerce")
    common = qh.notna() & qe.notna()
    out.loc[~common, ["EC2QH", "AdjEC2QE"]] = np.nan

    if verbose:
        print(f"Peyto EC QC: {int(common.sum())} common hourly Qh/Qe observations retained")
    return out


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file:\n{path}")
    return path
