"""04 - Compare modeled melt against independent SR50 observations.

This script uses the hourly melt scenarios produced by script 03 and compares
them with independent surface-height (SR50) observations.

Peyto
-----
The original MATLAB manuscript analysis compared melt over four EC/SR50
segments, excluding the very short third segment:

    P1 = MATLAB 13:62
    P2 = MATLAB 275:373
    P4 = MATLAB 492:583
    P5 = MATLAB 733:766

These are retained exactly here (converted to Python 0-based indexing).
Measured melt is calculated from the change in SR50 surface height using
rho_ice = 918 kg m-3, as in the legacy manuscript analysis.

Athabasca
---------
The full 18-Jun-2015 to 30-Jun-2015 analysis period is used. Measured melt is
calculated from surface lowering relative to the first SR50 observation.

Required model inputs
---------------------
data/interim/Peyto_melt_scenarios_hourly.csv
data/interim/Athabasca_melt_scenarios_hourly.csv

Required measured-melt inputs
-----------------------------
Preferred CSV format (place anywhere under the corresponding raw folder):

    PeytoSR50_hourly.csv
    AthabascaSR50_hourly.csv

Each CSV must contain:
    time
    sr50_m

where `sr50_m` is the processed SR50 surface-height/change series in metres,
with the same sign convention as the legacy files:
- Peyto: increasing cumulative surface lowering gives positive melt change.
- Athabasca: the processed AthaSR50 series is cumulative surface lowering
  and increases positively, so melt = delta(sr50) * 918.

If your SR50 files are still .mat, convert them first rather than silently
guessing MATLAB datetime objects.

Outputs
-------
tables/Peyto_measured_melt_comparison.csv
tables/Athabasca_measured_melt_comparison.csv
tables/Measured_melt_comparison_summary.csv
data/interim/Peyto_SR50_aligned.csv
data/interim/Athabasca_SR50_aligned.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from project_config import PROJECT_ROOT
from project_config import PEYTO_RAW, ATHA_RAW, INTERIM, TABLES

RHO_ICE = 918.0  # kg m-3; 1 kg m-2 = 1 mm w.e.

# Exact segment indices from the final legacy MATLAB cumulative-melt analysis.
# MATLAB:
# id1 = 13:62
# id2 = 275:373
# id4 = 492:583
# id5 = 733:766
# Convert inclusive MATLAB 1-based indices to inclusive Python 0-based indices.
PEYTO_SEGMENTS = {
    "P1": (12, 61),
    "P2": (274, 372),
    "P4": (491, 582),
    "P5": (732, 765),
}

SCENARIOS = {
    "EC": "Melt_EC_mm_h",
    "Bulk": "Melt_Bulk_mm_h",
    "Kata_legacy_Kb001": "Melt_Kata_legacy_Kb001_mm_h",
    "Kata_legacy_Kbcal": "Melt_Kata_legacy_Kbcal_mm_h",
    "Kata_T6_preserveEa_Kb001": "Melt_Kata_T6_preserveEa_Kb001_mm_h",
    "Kata_T6_preserveEa_Kbcal": "Melt_Kata_T6_preserveEa_Kbcal_mm_h",
}


print("=== STARTING SCRIPT 04: SR50 MELT COMPARISON ===")
print("Project root:", PROJECT_ROOT)


def find_required(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(
            f"\nRequired file not found: {name}\n"
            f"Searched recursively under:\n{root}\n\n"
            "For SR50 data, create the site CSV with columns:\n"
            "    time,sr50_m\n"
            "and place it anywhere under the corresponding raw-data folder."
        )
    if len(hits) > 1:
        print(f"WARNING: found {len(hits)} copies of {name}; using {hits[0]}")
    return hits[0]


def read_model_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()

    missing = [c for c in SCENARIOS.values() if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path.name} is missing melt columns:\n{missing}\n"
            "Run 03_build_flux_scenarios.py first."
        )

    return df


def read_sr50_csv(path: Path, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"time", "sr50_m"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{path.name} must contain columns {sorted(required)}; "
            f"missing {sorted(missing)}."
        )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["sr50_m"] = pd.to_numeric(df["sr50_m"], errors="coerce")
    df = (
        df.dropna(subset=["time"])
        .drop_duplicates(subset=["time"])
        .set_index("time")
        .sort_index()
    )

    # Interpolate onto the exact hourly analysis grid using time interpolation.
    # Only interpolate within the observed SR50 range; no extrapolation.
    union = df.index.union(target_index).sort_values()
    aligned = (
        df.reindex(union)
        .interpolate(method="time", limit_area="inside")
        .reindex(target_index)
    )
    aligned.index.name = "time"

    return aligned


def percent_error(model, measured):
    if not np.isfinite(measured) or measured == 0:
        return np.nan
    return (model / measured - 1.0) * 100.0


def peyto_measured_melt(sr50: pd.Series, start_i: int, end_i: int):
    """
    Legacy Peyto convention:
        measured = [SR50(end) - SR50(start)] * 918
    """
    start = sr50.iloc[start_i]
    end = sr50.iloc[end_i]
    if not (np.isfinite(start) and np.isfinite(end)):
        return np.nan
    return (end - start) * RHO_ICE


def model_segment_melt(series: pd.Series, start_i: int, end_i: int):
    """
    Match legacy MATLAB segment integration:
        sum(melt(segment)) - melt(first_hour)

    This represents the N-1 hourly intervals between the starting and ending
    SR50 observations.
    """
    seg = pd.to_numeric(series.iloc[start_i:end_i + 1], errors="coerce")
    if len(seg) == 0:
        return np.nan, 0, 0

    valid = int(seg.notna().sum())
    expected_intervals = end_i - start_i

    first = seg.iloc[0]
    total = seg.sum(skipna=True)

    if np.isfinite(first):
        total -= first

    return float(total), valid, expected_intervals


def run_peyto():
    print("\n=== PEYTO MEASURED MELT ===")

    model_path = INTERIM / "Peyto_melt_scenarios_hourly.csv"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}\nRun script 03 first."
        )

    model = read_model_hourly(model_path)

    # Ensure we are using exactly the legacy 768-hour manuscript grid.
    expected_index = pd.date_range(
        "2008-07-24 01:00",
        "2008-08-25 00:00",
        freq="1h",
    )
    model = model.reindex(expected_index)

    sr50_path = find_required(PEYTO_RAW, "PeytoSR50_hourly.csv")
    sr50 = read_sr50_csv(sr50_path, expected_index)

    sr50.to_csv(
        INTERIM / "Peyto_SR50_aligned.csv",
        index_label="time",
    )

    detail_rows = []

    for period, (i0, i1) in PEYTO_SEGMENTS.items():
        start_time = expected_index[i0]
        end_time = expected_index[i1]

        measured = peyto_measured_melt(sr50["sr50_m"], i0, i1)

        for scenario, melt_col in SCENARIOS.items():
            model_melt, n_valid, expected_intervals = model_segment_melt(
                model[melt_col], i0, i1
            )

            detail_rows.append(
                {
                    "glacier": "Peyto",
                    "period": period,
                    "start": start_time,
                    "end": end_time,
                    "expected_intervals": expected_intervals,
                    "scenario": scenario,
                    "n_valid_model_hours": n_valid,
                    "measured_melt_mm": measured,
                    "modeled_melt_mm": model_melt,
                    "difference_mm": (
                        model_melt - measured
                        if np.isfinite(model_melt) and np.isfinite(measured)
                        else np.nan
                    ),
                    "percent_error": percent_error(model_melt, measured),
                }
            )

    detail = pd.DataFrame(detail_rows)

    # Summed comparison across the four retained Peyto periods.
    sum_rows = []

    for scenario in SCENARIOS:
        sub = detail[detail["scenario"] == scenario].copy()

        measured_total = sub["measured_melt_mm"].sum(min_count=len(PEYTO_SEGMENTS))
        model_total = sub["modeled_melt_mm"].sum(min_count=len(PEYTO_SEGMENTS))

        sum_rows.append(
            {
                "glacier": "Peyto",
                "comparison_period": "P1+P2+P4+P5",
                "scenario": scenario,
                "measured_melt_mm": measured_total,
                "modeled_melt_mm": model_total,
                "difference_mm": (
                    model_total - measured_total
                    if np.isfinite(model_total) and np.isfinite(measured_total)
                    else np.nan
                ),
                "percent_error": percent_error(model_total, measured_total),
                "coverage_note": (
                    f"{int(sub['n_valid_model_hours'].sum())} valid model hours "
                    f"across {int(sub['expected_intervals'].sum())} SR50 intervals"
                ),
            }
        )

    summary = pd.DataFrame(sum_rows)

    detail.to_csv(
        TABLES / "Peyto_measured_melt_comparison.csv",
        index=False,
    )

    print("\nPeyto period-by-period:")
    print(detail.to_string(index=False))

    print("\nPeyto summed periods:")
    print(summary.to_string(index=False))

    return detail, summary


def athabasca_measured_melt(sr50: pd.Series):
    """
    Processed Athabasca AthaSR50 is cumulative surface lowering and increases
    positively through the campaign. Convert its change directly to mm w.e.:

        melt = (SR50(end) - SR50(start)) * 918
    """
    valid = sr50.dropna()
    if len(valid) < 2:
        return np.nan

    return (valid.iloc[-1] - valid.iloc[0]) * RHO_ICE


def run_athabasca():
    print("\n=== ATHABASCA MEASURED MELT ===")

    model_path = INTERIM / "Athabasca_melt_scenarios_hourly.csv"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}\nRun script 03 first."
        )

    model = read_model_hourly(model_path)

    expected_index = pd.date_range(
        "2015-06-18 00:00",
        "2015-06-30 00:00",
        freq="1h",
    )
    model = model.reindex(expected_index)

    sr50_path = find_required(ATHA_RAW, "AthabascaSR50_hourly.csv")
    sr50 = read_sr50_csv(sr50_path, expected_index)

    sr50.to_csv(
        INTERIM / "Athabasca_SR50_aligned.csv",
        index_label="time",
    )

    measured = athabasca_measured_melt(sr50["sr50_m"])

    rows = []

    # The independent measured period spans from first to last SR50 observation.
    sr_valid = sr50["sr50_m"].dropna()
    if len(sr_valid) < 2:
        start = pd.NaT
        end = pd.NaT
        expected_hours = 0
    else:
        start = sr_valid.index[0]
        end = sr_valid.index[-1]
        expected_hours = int((end - start) / pd.Timedelta(hours=1))

    for scenario, melt_col in SCENARIOS.items():
        if pd.isna(start) or pd.isna(end):
            total = np.nan
            n_valid = 0
        else:
            s = pd.to_numeric(
                model.loc[start:end, melt_col],
                errors="coerce",
            )

            # Exclude first hour for consistency with a start-to-end surface
            # displacement spanning N-1 hourly intervals.
            first = s.iloc[0] if len(s) else np.nan
            total = s.sum(skipna=True)
            if np.isfinite(first):
                total -= first

            n_valid = int(s.notna().sum())

        coverage = (
            100.0 * max(n_valid - 1, 0) / expected_hours
            if expected_hours > 0
            else np.nan
        )

        # Avoid presenting a full-period percentage error as valid if model
        # forcing is badly incomplete (notably the current Athabasca bulk case).
        independent_error_valid = coverage >= 95.0

        rows.append(
            {
                "glacier": "Athabasca",
                "comparison_period": "full_SR50_period",
                "scenario": scenario,
                "start": start,
                "end": end,
                "expected_intervals": expected_hours,
                "n_valid_model_hours": n_valid,
                "coverage_percent": coverage,
                "measured_melt_mm": measured,
                "modeled_melt_mm": total,
                "difference_mm": (
                    total - measured
                    if independent_error_valid
                    and np.isfinite(total)
                    and np.isfinite(measured)
                    else np.nan
                ),
                "percent_error": (
                    percent_error(total, measured)
                    if independent_error_valid
                    else np.nan
                ),
                "comparison_valid": independent_error_valid,
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        TABLES / "Athabasca_measured_melt_comparison.csv",
        index=False,
    )

    print("\nAthabasca comparison:")
    print(summary.to_string(index=False))

    return summary


def run_analysis():
    TABLES.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)

    peyto_detail, peyto_summary = run_peyto()
    atha_summary = run_athabasca()

    combined = pd.concat(
        [
            peyto_summary.assign(comparison_valid=True),
            atha_summary[
                [
                    "glacier",
                    "comparison_period",
                    "scenario",
                    "measured_melt_mm",
                    "modeled_melt_mm",
                    "difference_mm",
                    "percent_error",
                    "comparison_valid",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    out = TABLES / "Measured_melt_comparison_summary.csv"
    combined.to_csv(out, index=False)

    print("\n=== FINAL MEASURED MELT SUMMARY ===")
    print(combined.to_string(index=False))

    print("\nSaved:")
    print(TABLES / "Peyto_measured_melt_comparison.csv")
    print(TABLES / "Athabasca_measured_melt_comparison.csv")
    print(out)
    print(INTERIM / "Peyto_SR50_aligned.csv")
    print(INTERIM / "Athabasca_SR50_aligned.csv")

    return combined


measured_melt_results = run_analysis()
