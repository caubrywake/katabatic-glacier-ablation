# %% RUN ENTIRE SCRIPT (VS Code: click "Run Cell" above this line)
# Code 12 v3: A SINGLE VS Code Interactive Window cell. Running the cell
# executes the full script including the visible summary at the bottom.
# Alternatively, use "Run Python File" to execute in the Terminal.

"""12 - Physically consistent melt/sublimation partition and manuscript number report.

A NON-DESTRUCTIVE downstream addition to
    <repository-root>

Run after the existing scripts 03 (hourly melt scenarios) and 04 (SR50 alignment).
Does not recalculate turbulent transfer, change Kb, or overwrite the original
hourly fluxes, melt scenarios, manuscript tables, or Figures 6/7.

Scientific approach
-------------------
The existing src/turbulent_fluxes.py parameterizations use Lv=2.5 MJ kg-1 to
convert vapour mass transfer into latent heat flux Qe (W m-2). For EC data, this
script assumes the supplied LE was also computed with Lv=2.5 MJ kg-1; check
that assumption against the original EC processing documentation.

Step 1: recover the SIGNED vapour mass flux from legacy Qe:
    m_vap = Qe_legacy / Lv_reference  [kg m-2 s-1], positive toward surface.
Step 2: apply a phase-consistent latent heat when computing melt energy:
    Qe_physical = m_vap * Ls (for negative flux: ice sublimation)
                  m_vap * Lv (for positive flux: liquid condensation at 0 C).
    Optional --positive-phase ice treats positive flux as frost deposition and
    applies Ls in that branch as well.
Step 3: compute melt, sublimation and net ICE ablation:
    Qm = Qn + Qp + Qh + Qe_physical              [W m-2]
    Melt = max(Qm, 0) * 3600 / Lf                [mm w.e. h-1]
    Sublimation = max(-m_vap, 0) * 3600          [mm w.e. h-1]
    Frost deposition = max(m_vap, 0) * 3600    [mm w.e. h-1, ice phase only]
    Net ice ablation = Melt + Sublimation - Frost deposition.

IMPORTANT PHASE CAVEAT: At melting ice (Ts about 0 C), a positive water-vapour
flux may condense to LIQUID (dew), not deposit as ice. The default positive-
phase=liquid keeps that liquid mass separate from solid surface lowering; it
contributes latent heat to the melt energy, but NOT a negative ice-ablation
term. The optional positive-phase=ice is a sensitivity assumption, not an
observed temperature-based phase determination. Negative flux is assumed to be
sublimation of ICE, not evaporation from a surface meltwater film. Neither
phase is directly verified by the available constant-Ts=0 C surface model.

Constants at approximately 0 C (choose internally consistent rounded values):
    Lv  = 2,500,000 J/kg (legacy script src/turbulent_fluxes.py)
    Lf  =   334,000 J/kg (ice -> liquid, physical; note legacy script 03
                         incorrectly/alternatively used 316,900 J/kg)
    Ls  = Lv + Lf = 2,834,000 J/kg (ice -> vapour).

The script retains *original* and *corrected* melt side by side, to separate
changes due to Lf and latent-vapour treatment. The source data assume zero ice
surface temperature, negligible subsurface storage and no refreezing; these
assumptions are retained and must be discussed as limitations.

Inputs (existing, unchanged):
  data/interim/Peyto_melt_scenarios_hourly.csv
  data/interim/Athabasca_melt_scenarios_hourly.csv
  data/interim/Peyto_SR50_aligned.csv
  data/interim/Athabasca_SR50_aligned.csv

Outputs (NEW, no original files overwritten):
  data/interim/Peyto_ablation_partition_hourly.csv
  data/interim/Athabasca_ablation_partition_hourly.csv
  tables/Ablation_partition_by_period.csv
  tables/Ablation_partition_summary.csv
  tables/Ablation_partition_physics_assumptions.txt
  tables/Manuscript_ablation_numbers.csv  (all numbers for Results paragraphs)
  tables/Manuscript_ablation_numbers_full_precision.csv  (numerical audit)

Example (Windows / VS Code):
  python scripts/08_partition_surface_ablation.py

Examples for alternate root / sensitivity:
  python 08_partition_surface_ablation.py --root "D:/my_katabatic_copy"
  python 08_partition_surface_ablation.py --positive-phase ice

This script can be executed independently with numpy and pandas.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from project_config import PROJECT_ROOT as DEFAULT_ROOT
LV_REFERENCE = 2_500_000.0    # J/kg: legacy flux energy-to-vapour-mass convention
LF_FUSION = 334_000.0        # J/kg: physical ice-to-liquid fusion near 0 C
LS_SUBLIMATION = LV_REFERENCE + LF_FUSION  # J/kg: 2,834,000 (phase consistent)
RHO_ICE = 918.0              # kg/m3: retained original SR50 conversion
SECONDS_PER_HOUR = 3600.0

# Exact scenario-specific Qh and Qe columns from script 03, not assumed names.
SCENARIOS_BY_GLACIER = {
    "Peyto": {
        "EC": ("EC2QH", "AdjEC2QE"),
        "Bulk": ("Qh_Bulk", "Qe_Bulk"),
        "Kata_legacy_Kb001": ("Qh_Kata_legacy_Kb001", "Qe_Kata_legacy_Kb001"),
        "Kata_legacy_Kbcal": ("Qh_Kata_legacy_Kbcal", "Qe_Kata_legacy_Kbcal"),
        "Kata_T6_preserveEa_Kb001": (
            "Qh_Kata_T6_preserveEa_Kb001", "Qe_Kata_T6_preserveEa_Kb001"
        ),
        "Kata_T6_preserveEa_Kbcal": (
            "Qh_Kata_T6_preserveEa_Kbcal", "Qe_Kata_T6_preserveEa_Kbcal"
        ),
    },
    "Athabasca": {
        "EC": ("Qh_EC", "Qe_EC"),
        "Bulk": ("Qh_Bulk", "Qe_Bulk"),
        "Kata_legacy_Kb001": ("Qh_Kata_legacy_Kb001", "Qe_Kata_legacy_Kb001"),
        "Kata_legacy_Kbcal": ("Qh_Kata_legacy_Kbcal", "Qe_Kata_legacy_Kbcal"),
        "Kata_T6_preserveEa_Kb001": (
            "Qh_Kata_T6_preserveEa_Kb001", "Qe_Kata_T6_preserveEa_Kb001"
        ),
        "Kata_T6_preserveEa_Kbcal": (
            "Qh_Kata_T6_preserveEa_Kbcal", "Qe_Kata_T6_preserveEa_Kbcal"
        ),
    },
}

# Retains the 2008 Peyto SR50 sampling periods; 1-based MATLAB 13:62, etc.
# Tuple values are inclusive, Python 0-based index positions.
PEYTO_SEGMENTS = {"P1": (12, 61), "P2": (274, 372),
                  "P4": (491, 582), "P5": (732, 765)}
EXPECTED_GRIDS = {
    "Peyto": ("2008-07-24 01:00", "2008-08-25 00:00"),
    "Athabasca": ("2015-06-18 00:00", "2015-06-30 00:00"),
}


def read_time_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    x = pd.read_csv(path, parse_dates=["time"])
    if x["time"].isna().any() or x["time"].duplicated().any():
        raise ValueError(f"Invalid or duplicate timestamps: {path}")
    return x.set_index("time").sort_index()


def require_columns(df: pd.DataFrame, cols: set[str], source: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {source}: {sorted(missing)}")


def partition_one_scenario(df: pd.DataFrame, scenario: str,
                           qh_col: str, qe_col: str,
                           positive_phase: str) -> pd.DataFrame:
    """Preserve input NaNs, and return only *new* scenario output columns."""
    qh = pd.to_numeric(df[qh_col], errors="coerce").to_numpy(float)
    qe_old = pd.to_numeric(df[qe_col], errors="coerce").to_numpy(float)
    qn = pd.to_numeric(df["Qn"], errors="coerce").to_numpy(float)
    qp = pd.to_numeric(df["Qp"], errors="coerce").to_numpy(float)

    # Mass transfer is latent-energy flux divided by the SAME coefficient used
    # in the existing Qe parameterization. Merely dividing Qe_old by Ls while
    # retaining Qe_old in Qm would not be a self-consistent phase treatment.
    vapor_mass_flux_kg_m2_s = qe_old / LV_REFERENCE
    sub_mm = np.where(np.isfinite(vapor_mass_flux_kg_m2_s),
                      np.maximum(-vapor_mass_flux_kg_m2_s, 0) * SECONDS_PER_HOUR,
                      np.nan)
    inward_mm = np.where(np.isfinite(vapor_mass_flux_kg_m2_s),
                         np.maximum(vapor_mass_flux_kg_m2_s, 0) * SECONDS_PER_HOUR,
                         np.nan)

    if positive_phase == "ice":
        frost_deposition_mm = inward_mm
        liquid_condensation_mm = np.where(np.isfinite(inward_mm), 0.0, np.nan)
        positive_latent_heat = LS_SUBLIMATION
    elif positive_phase == "liquid":
        frost_deposition_mm = np.where(np.isfinite(inward_mm), 0.0, np.nan)
        liquid_condensation_mm = inward_mm
        positive_latent_heat = LV_REFERENCE
    else:
        raise ValueError("positive_phase must be 'liquid' or 'ice'")

    # Negative Qe: actual ice -> vapour (Ls); positive: dew or frost, selectable.
    phase_latent_heat = np.where(vapor_mass_flux_kg_m2_s < 0,
                                 LS_SUBLIMATION, positive_latent_heat)
    qe_physical = vapor_mass_flux_kg_m2_s * phase_latent_heat
    qm_physical = qn + qp + qh + qe_physical
    melt_mm = np.where(np.isfinite(qm_physical),
                       np.maximum(qm_physical, 0) * SECONDS_PER_HOUR / LF_FUSION,
                       np.nan)
    net_vapor_ice_loss_mm = sub_mm - frost_deposition_mm
    ablation_mm = melt_mm + net_vapor_ice_loss_mm

    # Unused negative melt energy indicates cooling/storage, not negative melt.
    cooling_residual = np.where(np.isfinite(qm_physical),
                                np.minimum(qm_physical, 0), np.nan)

    return pd.DataFrame({
        f"Vapor_mass_flux_{scenario}_kg_m2_s": vapor_mass_flux_kg_m2_s,
        f"Qe_phase_{scenario}_W_m2": qe_physical,
        f"Qm_phase_{scenario}_W_m2": qm_physical,
        f"Cooling_residual_{scenario}_W_m2": cooling_residual,
        f"Melt_physical_{scenario}_mm_h": melt_mm,
        f"Sublimation_{scenario}_mm_h": sub_mm,
        f"Frost_deposition_{scenario}_mm_h": frost_deposition_mm,
        f"Liquid_condensation_{scenario}_mm_h": liquid_condensation_mm,
        f"Net_ice_vapor_loss_{scenario}_mm_h": net_vapor_ice_loss_mm,
        f"Ablation_{scenario}_mm_h": ablation_mm,
    }, index=df.index)


def partition_glacier(df: pd.DataFrame, glacier: str, phase: str) -> pd.DataFrame:
    expected = pd.date_range(*EXPECTED_GRIDS[glacier], freq="1h")
    if not df.index.equals(expected):
        raise ValueError(
            f"{glacier}: expected hourly grid {expected[0]} to {expected[-1]} "
            f"({len(expected)} rows), but found {len(df)} different timestamps. "
            "Do not silently shift the SR50 interval indices."
        )
    required = {"Qn", "Qp"}
    for qh, qe in SCENARIOS_BY_GLACIER[glacier].values():
        required.update((qh, qe))
    require_columns(df, required, glacier)
    out = df.copy()
    for scenario, (qh, qe) in SCENARIOS_BY_GLACIER[glacier].items():
        added = partition_one_scenario(df, scenario, qh, qe, phase)
        for col in added.columns:
            out[col] = added[col]
    return out


def get_validation_periods(glacier: str, sr50: pd.DataFrame,
                           model: pd.DataFrame):
    require_columns(sr50, {"sr50_m"}, f"{glacier} SR50")
    if not sr50.index.equals(model.index):
        raise ValueError(f"{glacier}: aligned SR50 timestamps differ from model grid")
    if glacier == "Peyto":
        return PEYTO_SEGMENTS
    finite = np.flatnonzero(np.isfinite(pd.to_numeric(sr50["sr50_m"], errors="coerce")))
    if len(finite) < 2:
        raise ValueError("Athabasca: fewer than two valid SR50 measurements")
    return {"full_SR50_period": (int(finite[0]), int(finite[-1]))}


def period_sum(series: pd.Series, start: int, end: int, glacier: str):
    """Integrate N-1 hours using source-specific flux timestamp conventions.

    Peyto: original MATLAB validation uses hours (start,end], retained to
        replicate the four period integrations exactly.
    Athabasca: the 288 archived energy-flux hours are stamped at the START
        (June 18 00:00 to June 29 23:00), while the 289 SR50 endpoints end
        June 30 00:00. Use flux hours [start,end) to capture 288 intervals.
        Older script 04 instead used (start,end], silently omitting the final
        NaN energy record and dropping the first (available) flux hour.
    """
    if glacier == "Peyto":
        values = pd.to_numeric(series.iloc[start + 1:end + 1], errors="coerce")
    elif glacier == "Athabasca":
        values = pd.to_numeric(series.iloc[start:end], errors="coerce")
    else:
        raise ValueError(f"Unknown glacier {glacier}")
    n_required = end - start
    n_valid = int(np.isfinite(values.to_numpy(float)).sum())
    return (float(values.sum()) if n_valid == n_required else np.nan,
            n_valid, n_required)


def safe_error(model: float, observed: float) -> float:
    return ((model / observed - 1.0) * 100.0
            if np.isfinite(model) and np.isfinite(observed) and observed != 0
            else np.nan)


def compare_sr50(glacier: str, model: pd.DataFrame, sr50: pd.DataFrame):
    periods = get_validation_periods(glacier, sr50, model)
    records = []
    names = SCENARIOS_BY_GLACIER[glacier]
    for period, (start, end) in periods.items():
        h0, h1 = sr50["sr50_m"].iloc[[start, end]]
        measured = ((float(h1) - float(h0)) * RHO_ICE
                    if np.isfinite(h0) and np.isfinite(h1) else np.nan)
        for name in names:
            columns = {
                "modeled_melt_mmwe": f"Melt_physical_{name}_mm_h",
                "modeled_sublimation_mmwe": f"Sublimation_{name}_mm_h",
                "modeled_frost_deposition_mmwe": f"Frost_deposition_{name}_mm_h",
                "modeled_liquid_condensation_mmwe": f"Liquid_condensation_{name}_mm_h",
                "modeled_net_ice_vapor_loss_mmwe": f"Net_ice_vapor_loss_{name}_mm_h",
                "modeled_ablation_mmwe": f"Ablation_{name}_mm_h",
                "legacy_modeled_melt_mmwe": f"Melt_{name}_mm_h",
            }
            result = {"glacier": glacier, "period": period, "scenario": name,
                      "start": model.index[start], "end": model.index[end],
                      "measured_SR50_ablation_mmwe": measured}
            for label, col in columns.items():
                # Original legacy melt column is optional in future reruns.
                result[label] = (period_sum(model[col], start, end, glacier)[0]
                                 if col in model.columns else np.nan)
            nvalid, expected = period_sum(model[columns["modeled_ablation_mmwe"]],
                                          start, end, glacier)[1:]
            result["valid_hourly_intervals"] = nvalid
            result["expected_hourly_intervals"] = expected
            result["coverage_valid"] = bool(nvalid == expected and np.isfinite(measured))
            result["ablation_bias_mmwe"] = (
                result["modeled_ablation_mmwe"] - measured
                if result["coverage_valid"] else np.nan
            )
            result["ablation_percent_error"] = (
                safe_error(result["modeled_ablation_mmwe"], measured)
                if result["coverage_valid"] else np.nan
            )
            records.append(result)
    return pd.DataFrame(records)


def summarize_periods(period_df: pd.DataFrame) -> pd.DataFrame:
    summable = ["measured_SR50_ablation_mmwe", "modeled_melt_mmwe",
                "modeled_sublimation_mmwe", "modeled_frost_deposition_mmwe",
                "modeled_liquid_condensation_mmwe", "modeled_net_ice_vapor_loss_mmwe",
                "modeled_ablation_mmwe", "legacy_modeled_melt_mmwe"]
    rows = []
    for (glacier, scenario), group in period_df.groupby(["glacier", "scenario"], sort=False):
        record = {"glacier": glacier, "scenario": scenario,
                  "comparison_period": "P1+P2+P4+P5" if glacier == "Peyto" else "full_SR50_period",
                  "valid_hourly_intervals": int(group["valid_hourly_intervals"].sum()),
                  "expected_hourly_intervals": int(group["expected_hourly_intervals"].sum()),
                  "coverage_valid": bool(group["coverage_valid"].all())}
        for c in summable:
            record[c] = group[c].sum(min_count=len(group))
        if not record["coverage_valid"]:
            record["modeled_ablation_mmwe"] = np.nan
        record["ablation_bias_mmwe"] = (
            record["modeled_ablation_mmwe"] - record["measured_SR50_ablation_mmwe"]
            if record["coverage_valid"] else np.nan
        )
        record["ablation_percent_error"] = (
            safe_error(record["modeled_ablation_mmwe"], record["measured_SR50_ablation_mmwe"])
            if record["coverage_valid"] else np.nan
        )
        if np.isfinite(record["legacy_modeled_melt_mmwe"]) and np.isfinite(record["modeled_melt_mmwe"]):
            record["melt_change_vs_legacy_mmwe"] = (record["modeled_melt_mmwe"] -
                                                     record["legacy_modeled_melt_mmwe"])
        else:
            record["melt_change_vs_legacy_mmwe"] = np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def run(root: Path, positive_phase: str) -> pd.DataFrame:
    interim = root / "data" / "interim"
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    details = []
    for glacier in ("Peyto", "Athabasca"):
        model = read_time_csv(interim / f"{glacier}_melt_scenarios_hourly.csv")
        sr50 = read_time_csv(interim / f"{glacier}_SR50_aligned.csv")
        updated = partition_glacier(model, glacier, positive_phase)
        out = interim / f"{glacier}_ablation_partition_hourly.csv"
        updated.to_csv(out, index_label="time")
        print(f"Saved hourly components: {out}")
        details.append(compare_sr50(glacier, updated, sr50))

    by_period = pd.concat(details, ignore_index=True)
    by_period.to_csv(tables / "Ablation_partition_by_period.csv", index=False)
    summary = summarize_periods(by_period)
    summary.to_csv(tables / "Ablation_partition_summary.csv", index=False)

    notes = (
        "Katabatic ablation partition: assumptions\n"
        "Source: output of clean pipeline 03, no Kb recalibration.\n"
        f"Lv_reference = {LV_REFERENCE:g} J/kg (legacy Qe conversion)\n"
        f"Lf_physical = {LF_FUSION:g} J/kg (ice melt; legacy Lf was 316900 J/kg)\n"
        f"Ls_physical = {LS_SUBLIMATION:g} J/kg (= Lv_reference + Lf_physical)\n"
        f"Positive vapour flux phase = {positive_phase}\n"
        "Qe sign positive toward ice. Negative flux treated as ice sublimation.\n"
        "Default positive flux treated as liquid condensation (dew); mass does not\n"
        "reduce solid SR50 lowering. --positive-phase ice treats frost deposition.\n"
        "At 0 C, phase assignment is an assumption; measured surface temperature\n"
        "and meltwater-film state are unavailable in these hourly input files.\n"
        "Underlying EC LE is assumed encoded using Lv_reference; VERIFY against\n"
        "original EC processing before treating results as definitive.\n"
        "SR50 ice density = 918 kg/m3; Peyto excludes non-retained periods.\n"
        "Peyto intervals are (start,end] (original MATLAB convention).\n"
        "Athabasca intervals are [start,end) to match 288 start-stamped flux hours.\n"
        "Missing-hour coverage invalidates the period statistics; no zero filling.\n"
        "Legacy melt values are retained for direct comparisons.\n"
        "Assumes Ts=0 C, Qg=0, dU/dt=0, no refreezing or surface energy storage.\n"
    )
    (tables / "Ablation_partition_physics_assumptions.txt").write_text(notes, encoding="utf-8")
    cols = ["glacier", "scenario", "measured_SR50_ablation_mmwe",
            "modeled_melt_mmwe", "modeled_sublimation_mmwe",
            "modeled_frost_deposition_mmwe", "modeled_liquid_condensation_mmwe",
            "modeled_ablation_mmwe", "legacy_modeled_melt_mmwe",
            "ablation_percent_error", "coverage_valid"]
    print("\nPHYSICAL PARTITION vs SR50 (same periods as original manuscript):")
    print(summary[cols].round(4).to_string(index=False))
    print("\nSaved:", tables / "Ablation_partition_by_period.csv")
    print("Saved:", tables / "Ablation_partition_summary.csv")
    print("Saved:", tables / "Ablation_partition_physics_assumptions.txt")
    export_manuscript_ablation_numbers(root, summary)
    return summary



# -----------------------------------------------------------------------------
# NEW IN V3: one manuscript-ready, reproducible numerical reference table.
# Does not modify the melt/sublimation partition, masks, SR50 conventions,
# calibrations, or existing Code 12 output files.
# -----------------------------------------------------------------------------

MAIN_MANUSCRIPT_SCENARIOS = {
    "EC": "EC",
    "Bulk transfer": "Bulk",
    "Calibrated katabatic": "Kata_T6_preserveEa_Kbcal",
}


def _manuscript_intervals(df: pd.DataFrame, sr50: pd.DataFrame, glacier: str) -> pd.DataFrame:
    """Exact N-1 intervals already used by the Code 12 SR50 validation."""
    periods = get_validation_periods(glacier, sr50, df)
    pieces = []
    for start, end in periods.values():
        if glacier == "Peyto":
            pieces.append(df.iloc[start + 1:end + 1])
        else:
            pieces.append(df.iloc[start:end])
    return pd.concat(pieces, axis=0)


def build_manuscript_ablation_table(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Produce all numbers for both Section 4.4 manuscript paragraphs.

    Cumulative quantities and radiation fractions: SR50 validation windows
      (Peyto's four retained periods and Athabasca's 288 hourly intervals).
    Hourly mean and maximum: every valid modeled hourly value in the series
      shown in the Figure 7 time-series panels. This deliberate distinction
      reproduces the statistics in the revised manuscript passage.
    The one missing first Athabasca bulk-transfer hour is NOT imputed; for that
      method the 287-hour integrated total is reported against the SR50 period.
      Coverage counts are retained in the output for transparent provenance.
    """
    rows = []
    interim = root / "data" / "interim"
    for glacier in ("Athabasca", "Peyto"):
        df = read_time_csv(interim / f"{glacier}_ablation_partition_hourly.csv")
        sr = read_time_csv(interim / f"{glacier}_SR50_aligned.csv")
        subset = _manuscript_intervals(df, sr, glacier)
        for method, scenario in MAIN_MANUSCRIPT_SCENARIOS.items():
            hit = summary.loc[(summary["glacier"] == glacier) &
                              (summary["scenario"] == scenario)]
            if len(hit) != 1:
                raise ValueError(f"Expected exactly one summary row for {glacier}/{scenario}")
            rec = hit.iloc[0]
            obs = float(rec["measured_SR50_ablation_mmwe"])
            ablation_name = f"Ablation_{scenario}_mm_h"
            melt_name = f"Melt_physical_{scenario}_mm_h"
            sub_name = f"Sublimation_{scenario}_mm_h"
            dep_name = f"Frost_deposition_{scenario}_mm_h"
            dew_name = f"Liquid_condensation_{scenario}_mm_h"
            vapor_name = f"Net_ice_vapor_loss_{scenario}_mm_h"
            need = (ablation_name, melt_name, sub_name, dep_name,
                    dew_name, vapor_name, "Qn")
            if not all(k in df.columns for k in need):
                raise KeyError(f"Missing ablation component or Qn columns: {glacier}/{scenario}")

            # Code 12 intentionally sets incomplete-period component totals to
            # NaN. The manuscript report integrates the *available* hourly
            # values, with the coverage count retained (no gap infilling).
            melt = float(subset[melt_name].sum(min_count=1))
            sub = float(subset[sub_name].sum(min_count=1))
            dep = float(subset[dep_name].sum(min_count=1))
            dew = float(subset[dew_name].sum(min_count=1))
            vapor_loss = float(subset[vapor_name].sum(min_count=1))
            # For fully covered scenarios, the aggregate must agree with Code 12.
            for name, computed in (("modeled_melt_mmwe", melt),
                                   ("modeled_sublimation_mmwe", sub),
                                   ("modeled_frost_deposition_mmwe", dep),
                                   ("modeled_liquid_condensation_mmwe", dew),
                                   ("modeled_net_ice_vapor_loss_mmwe", vapor_loss)):
                original = float(rec[name])
                if np.isfinite(original) and not np.isclose(original, computed, atol=1e-7):
                    raise AssertionError(f"Manuscript component mismatch for {glacier}/{scenario}/{name}")

            # In Code 12 v2 the one-hour-incomplete Athabasca bulk-transfer
            # ablation total is deliberately NaN. The manuscript has elected
            # to report the available-hour integrated estimate. All components
            # are already summed over those same available hours in the v2
            # summary; no extrapolation or zero-filling is performed.
            modeled = melt + vapor_loss
            tabulated = rec["modeled_ablation_mmwe"]
            if np.isfinite(tabulated) and not np.isclose(modeled, float(tabulated), atol=1e-7):
                raise AssertionError(f"Mass balance mismatch: {glacier}/{scenario}")
            direct = subset[ablation_name].sum(min_count=1)
            if not np.isclose(modeled, direct, atol=1e-7):
                raise AssertionError(f"Hourly ablation does not sum to {modeled}: {glacier}/{scenario}")
            bias = modeled - obs
            n_valid = int(rec["valid_hourly_intervals"])
            expected = int(rec["expected_hourly_intervals"])

            # Radiation energy percentages: Qn over SAME comparison intervals
            # and with the SAME scenario coverage mask. Only the calibrated
            # katabatic entries are cited in the current manuscript prose,
            # but the table reports all three methods for completeness.
            available = subset[melt_name].notna() & subset[sub_name].notna()
            rad = float(subset.loc[available, "Qn"].sum())
            if not np.isfinite(rad) or not (np.isfinite(melt) and np.isfinite(sub)):
                rad_of_melt = np.nan
                rad_of_combined = np.nan
            else:
                # Convert cumulative mm w.e. back to J/m2, and hourly Qn in
                # W/m2 into J/m2 with 3600 seconds per integration interval.
                rad_of_melt = 100 * rad * SECONDS_PER_HOUR / (LF_FUSION * melt) if melt > 0 else np.nan
                den = LF_FUSION * melt + LS_SUBLIMATION * sub
                rad_of_combined = 100 * rad * SECONDS_PER_HOUR / den if den > 0 else np.nan

            hourly = pd.to_numeric(df[ablation_name], errors="coerce")
            rows.append({
                "glacier": glacier,
                "method": method,
                "scenario": scenario,
                "measured_SR50_ablation_mmwe": obs,
                "modeled_melt_mmwe": melt,
                "modeled_sublimation_mmwe": sub,
                "modeled_frost_deposition_mmwe": dep,
                "modeled_liquid_condensation_mmwe": dew,
                "modeled_net_ice_vapor_loss_mmwe": vapor_loss,
                "modeled_total_ablation_mmwe": modeled,
                "ablation_bias_mmwe": bias,
                "ablation_percent_error": 100.0 * bias / obs if obs > 0 else np.nan,
                "sublimation_over_melt_pct": 100.0 * sub / melt if melt > 0 else np.nan,
                "sublimation_share_of_ablation_pct": 100.0 * sub / modeled if modeled > 0 else np.nan,
                "melt_share_of_ablation_pct": 100.0 * melt / modeled if modeled > 0 else np.nan,
                "max_hourly_ablation_mmwe_h": float(hourly.max()),
                "mean_hourly_ablation_mmwe_h": float(hourly.mean()),
                "net_radiation_share_of_melt_energy_pct": rad_of_melt,
                "net_radiation_share_of_melt_plus_sublimation_energy_pct": rad_of_combined,
                "sr50_valid_intervals": n_valid,
                "sr50_expected_intervals": expected,
                "all_valid_hourly_rate_values": int(hourly.count()),
                "all_expected_hourly_rate_values": len(hourly),
                "cumulative_statistic_basis": (
                    "Four retained SR50 periods P1+P2+P4+P5"
                    if glacier == "Peyto" else "Full SR50 observation period"
                ),
                "hourly_statistic_basis": "All available hourly values (Figure 7)",
            })
    return pd.DataFrame.from_records(rows)


def export_manuscript_ablation_numbers(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Write one report table in published precision and an audit copy."""
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    raw = build_manuscript_ablation_table(root, summary)
    raw_path = tables / "Manuscript_ablation_numbers_full_precision.csv"
    raw.to_csv(raw_path, index=False, float_format="%.12g")

    manuscript = raw.copy()
    # One-decimal-place numbers reproduce the reported manuscript values.
    one_decimal = [
        "measured_SR50_ablation_mmwe", "modeled_melt_mmwe",
        "modeled_sublimation_mmwe", "modeled_frost_deposition_mmwe",
        "modeled_liquid_condensation_mmwe", "modeled_net_ice_vapor_loss_mmwe",
        "modeled_total_ablation_mmwe", "ablation_bias_mmwe",
        "ablation_percent_error", "max_hourly_ablation_mmwe_h",
        "mean_hourly_ablation_mmwe_h", "net_radiation_share_of_melt_energy_pct",
        "net_radiation_share_of_melt_plus_sublimation_energy_pct",
    ]
    manuscript[one_decimal] = manuscript[one_decimal].round(1)
    manuscript["sublimation_over_melt_pct"] = manuscript["sublimation_over_melt_pct"].round(3)
    manuscript["sublimation_share_of_ablation_pct"] = manuscript["sublimation_share_of_ablation_pct"].round(3)
    manuscript["melt_share_of_ablation_pct"] = manuscript["melt_share_of_ablation_pct"].round(2)
    path = tables / "Manuscript_ablation_numbers.csv"
    manuscript.to_csv(path, index=False)

    print("\n" + "=" * 80, flush=True)
    print("MANUSCRIPT NUMBERS — cumulative ablation and hourly rates", flush=True)
    print("=" * 80, flush=True)
    cols1 = ["glacier", "method", "measured_SR50_ablation_mmwe", "modeled_melt_mmwe",
             "modeled_sublimation_mmwe", "modeled_total_ablation_mmwe",
             "ablation_bias_mmwe", "ablation_percent_error",
             "max_hourly_ablation_mmwe_h", "mean_hourly_ablation_mmwe_h"]
    print(manuscript[cols1].to_string(index=False), flush=True)
    print("\nMANUSCRIPT NUMBERS — sublimation fractions and radiation", flush=True)
    cols2 = ["glacier", "method", "sublimation_over_melt_pct",
             "sublimation_share_of_ablation_pct", "melt_share_of_ablation_pct",
             "net_radiation_share_of_melt_energy_pct",
             "net_radiation_share_of_melt_plus_sublimation_energy_pct"]
    print(manuscript[cols2].to_string(index=False), flush=True)
    print("\nCumulative totals and radiation: SR50 validation periods only."
          " Hourly rate maxima/means: all available Figure 7 hourly values.", flush=True)
    print(f"Saved manuscript reporting table: {path}", flush=True)
    print(f"Saved full-precision audit table: {raw_path}", flush=True)
    return manuscript


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="Root katabatic folder, with data/interim and tables subfolders")
    p.add_argument("--positive-phase", choices=["liquid", "ice"], default="liquid",
                   help="Positive vapour flux: liquid dew (default, melting surface) or ice frost")
    args = p.parse_args()
    print(f"Project root: {args.root}\nPositive vapour phase: {args.positive_phase}")
    run(args.root, args.positive_phase)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 72, flush=True)
    print("CODE 12 STARTED — melt / sublimation / ablation partition", flush=True)
    print("=" * 72, flush=True)

    # The VS Code Interactive Window is a Jupyter kernel. Its command line
    # contains notebook arguments, so calling argparse.parse_args() there may
    # fail. Run directly instead, with the same default root as the CLI.
    if "ipykernel" in sys.modules:
        print("Mode: VS Code / Jupyter Interactive Window", flush=True)
        print(f"Input project folder: {DEFAULT_ROOT}", flush=True)
        summary12 = run(DEFAULT_ROOT, positive_phase="liquid")
        try:
            from IPython.display import display
            shown_cols = ["glacier", "scenario", "modeled_melt_mmwe",
                          "modeled_sublimation_mmwe", "modeled_ablation_mmwe",
                          "measured_SR50_ablation_mmwe", "ablation_percent_error",
                          "coverage_valid"]
            display(summary12[shown_cols].round(3))
            manuscript_table = pd.read_csv(DEFAULT_ROOT / "tables" / "Manuscript_ablation_numbers.csv")
            display(manuscript_table[[
                "glacier", "method", "measured_SR50_ablation_mmwe",
                "modeled_melt_mmwe", "modeled_sublimation_mmwe",
                "modeled_total_ablation_mmwe", "ablation_bias_mmwe",
                "ablation_percent_error", "max_hourly_ablation_mmwe_h",
                "mean_hourly_ablation_mmwe_h",
                "net_radiation_share_of_melt_plus_sublimation_energy_pct",
            ]])
        except ImportError:
            print("Interactive table not available; printed summary is above.")
    else:
        print("Mode: Python terminal", flush=True)
        main()

    print("CODE 12 FINISHED — output CSV files saved in data/interim and tables.",
          flush=True)
