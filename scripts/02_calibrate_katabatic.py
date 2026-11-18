"""02 - Calibrate katabatic forcing for Peyto and Athabasca.

Revised forcing uses 6 m air temperature while preserving the vapour pressure
observed at the reliable near-surface humidity level.

This is important because relative humidity depends on temperature. Simply
combining RH measured at 2 m (or 1.5 m) with T measured at 6 m would imply a
different vapour pressure than was actually observed.

For each glacier, compare:
1. Legacy forcing + Kb = 0.01
2. Legacy forcing + EC-calibrated Kb
3. Revised 6 m Ta / preserved ea + Kb = 0.01
4. Revised 6 m Ta / preserved ea + EC-calibrated Kb

Primary calibration criterion:
    mean[(Qh + Qe)_Kata] = mean[(Qh + Qe)_EC]

Least-squares Kb is retained as a diagnostic.

Outputs
-------
tables/Kb_forcing_comparison.csv
data/interim/Peyto_kb_forcing_hourly.csv
data/interim/Athabasca_kb_forcing_hourly.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

from project_config import PROJECT_ROOT, PEYTO_RAW, ATHA_RAW, INTERIM, TABLES, FIGURES
# Import shared modules from the shipped scripts/ directory (not from a machine-specific root).
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline_common import apply_peyto_ec_qc

print("=== STARTING SCRIPT 02: KATABATIC CALIBRATION ===")
print("Project root:", PROJECT_ROOT)

from src.io_peyto import build_profile_a_hourly, build_ec_hourly
from src.thermodynamics import saturation_vapor_pressure_hpa
from src.turbulent_fluxes import (
    fit_kb_zero_mean_bias,
    fit_kb_least_squares,
    katabatic_transfer,
)
from src.metrics import pairwise_metrics


PROFILE_NAMES = [
    "ProfileA_tue_20080721_20080728.xlsx",
    "ProfileA_tue_20080803_20080808.xlsx",
    "ProfileA_tue_20080812_20080817.xlsx",
    "ProfileA_tue_20080823_20080825.xlsx",
]


def find_required(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(
            f"Could not find required file:\n{name}\n\n"
            f"Searched recursively under:\n{root}"
        )
    if len(hits) > 1:
        print(f"WARNING: found {len(hits)} copies of {name}; using {hits[0]}")
    return hits[0]


def preserve_vapour_pressure_rh(T_source_C, RH_source_pct, T_target_C):
    """
    Convert RH measured at a source temperature into the RH that gives the
    same vapour pressure at a target temperature.

    ea = RH_source/100 * es(T_source)

    RH_target = 100 * ea / es(T_target)

    This preserves atmospheric vapour pressure while changing only the
    temperature used to drive the katabatic sensible-heat term.
    """
    T_source_K = np.asarray(T_source_C, dtype=float) + 273.15
    T_target_K = np.asarray(T_target_C, dtype=float) + 273.15
    RH_source = np.asarray(RH_source_pct, dtype=float)

    es_source = saturation_vapor_pressure_hpa(T_source_K)
    es_target = saturation_vapor_pressure_hpa(T_target_K)

    ea_hpa = (RH_source / 100.0) * es_source
    RH_target = 100.0 * ea_hpa / es_target

    return RH_target, ea_hpa


def read_atha_profile_hourly(path: Path) -> pd.DataFrame:
    """Read Athabasca Campbell profile and return hourly 1.5 m and 6 m Ta/RH."""
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=[0, 2, 3],
        na_values=["NAN", "NaN", "nan"],
    )

    df["time"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")

    # Manuscript profile levels:
    # T/RH: 0.75, 1.5, 3, 4.5, 6 m
    out = pd.DataFrame(
        {
            "time": df["time"],
            "T_1p5m_C": pd.to_numeric(df["AirTemp_Avg(2)"], errors="coerce"),
            "RH_1p5m_pct": pd.to_numeric(df["RH_Avg(2)"], errors="coerce"),
            "T_6m_C": pd.to_numeric(df["AirTemp_Avg(5)"], errors="coerce"),
        }
    )

    return (
        out.dropna(subset=["time"])
        .set_index("time")
        .resample("1h")
        .mean()
    )


def read_atha_ec_hourly(path: Path) -> pd.DataFrame:
    """Read Conway Athabasca EC workbook and convert to manuscript sign convention."""
    df = pd.read_excel(path)

    df["time"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df["H"] = pd.to_numeric(df["H"], errors="coerce")
    df["LE"] = pd.to_numeric(df["LE"], errors="coerce")

    # Original Conway file sign convention is opposite the manuscript convention.
    df["Qh_EC"] = -df["H"]
    df["Qe_EC"] = -df["LE"]

    return (
        df.dropna(subset=["time"])
        .set_index("time")[["Qh_EC", "Qe_EC"]]
        .resample("1h")
        .mean()
    )


def fit_and_evaluate(
    glacier,
    forcing_name,
    T_C,
    RH_pct,
    qh_obs,
    qe_obs,
    altitude_m,
):
    """Fit Kb and evaluate fixed, mean-match, and least-squares scenarios."""
    T_K = np.asarray(T_C, dtype=float) + 273.15
    RH = np.asarray(RH_pct, dtype=float)
    qh = np.asarray(qh_obs, dtype=float)
    qe = np.asarray(qe_obs, dtype=float)

    valid = (
        np.isfinite(T_K)
        & np.isfinite(RH)
        & np.isfinite(qh)
        & np.isfinite(qe)
    )

    T_K = T_K[valid]
    RH = RH[valid]
    qh = qh[valid]
    qe = qe[valid]

    kb_mean = fit_kb_zero_mean_bias(
        T_K,
        RH,
        qh,
        qe,
        altitude_m=altitude_m,
        Pr=5.0,
        gamma=0.005,
        pressure_mode="legacy_fixed",
    )

    kb_ls = fit_kb_least_squares(
        T_K,
        RH,
        qh,
        qe,
        altitude_m=altitude_m,
        Pr=5.0,
        gamma=0.005,
        pressure_mode="legacy_fixed",
    )

    obs = qh + qe
    rows = []

    for scenario, kb in [
        ("Kb_0.01", 0.01),
        ("Kb_mean_match", kb_mean),
        ("Kb_least_squares", kb_ls),
    ]:
        model = katabatic_transfer(
            T_K,
            RH,
            altitude_m=altitude_m,
            Kb=kb,
            Pr=5.0,
            gamma=0.005,
            pressure_mode="legacy_fixed",
        )

        mod = model["Qh"] + model["Qe"]
        metrics = pairwise_metrics(obs, mod)

        rows.append(
            {
                "glacier": glacier,
                "forcing": forcing_name,
                "scenario": scenario,
                "Kb_m_s": kb,
                **metrics,
            }
        )

    return pd.DataFrame(rows), kb_mean, kb_ls


def add_model_columns(df, label, T_C, RH_pct, altitude_m, kb):
    model = katabatic_transfer(
        np.asarray(T_C, dtype=float) + 273.15,
        np.asarray(RH_pct, dtype=float),
        altitude_m=altitude_m,
        Kb=kb,
        Pr=5.0,
        gamma=0.005,
        pressure_mode="legacy_fixed",
    )

    df[f"Qh_{label}"] = model["Qh"]
    df[f"Qe_{label}"] = model["Qe"]
    df[f"Qtot_{label}"] = model["Qh"] + model["Qe"]


def run_analysis():
    INTERIM.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    all_results = []

    # =========================================================
    # PEYTO
    # =========================================================
    print("\n=== PEYTO ===")

    profile_paths = [
        find_required(PEYTO_RAW, name)
        for name in PROFILE_NAMES
    ]

    moh = find_required(PEYTO_RAW, "PeytoMNH_2008_MOH.obs")
    ec_orig = find_required(PEYTO_RAW, "EC_Munro2008_QH_QE.xlsx")
    ec_corr = find_required(
        PEYTO_RAW,
        "EC_Munro2008_QE_corrected20220412.xlsx",
    )

    peyto_met = build_profile_a_hourly(profile_paths, moh)
    peyto_ec = build_ec_hourly(ec_orig, ec_corr)

    peyto = peyto_met.join(peyto_ec, how="left")
    peyto = apply_peyto_ec_qc(peyto)

    peyto_required = [
        "T_2m_C",
        "T_6m_C",
        "RH_2m_pct",
        "EC2QH",
        "AdjEC2QE",
    ]

    peyto_valid = peyto.dropna(subset=peyto_required).copy()

    # Preserve 2 m vapour pressure while using 6 m Ta.
    peyto_valid["RH_equiv_6m_pct"], peyto_valid["ea_source_hPa"] = (
        preserve_vapour_pressure_rh(
            peyto_valid["T_2m_C"],
            peyto_valid["RH_2m_pct"],
            peyto_valid["T_6m_C"],
        )
    )

    print("Matched Peyto hours:", len(peyto_valid))
    print(
        "Mean RH: source 2 m =",
        round(peyto_valid["RH_2m_pct"].mean(), 2),
        "%; equivalent at 6 m =",
        round(peyto_valid["RH_equiv_6m_pct"].mean(), 2),
        "%",
    )

    # Legacy forcing: 2 m Ta + 2 m RH
    res, peyto_kb_legacy, peyto_kb_legacy_ls = fit_and_evaluate(
        glacier="Peyto",
        forcing_name="legacy_T2m_RH2m",
        T_C=peyto_valid["T_2m_C"],
        RH_pct=peyto_valid["RH_2m_pct"],
        qh_obs=peyto_valid["EC2QH"],
        qe_obs=peyto_valid["AdjEC2QE"],
        altitude_m=2169,
    )
    all_results.append(res)

    # Revised forcing: 6 m Ta, with RH transformed to preserve 2 m ea.
    res, peyto_kb_revised, peyto_kb_revised_ls = fit_and_evaluate(
        glacier="Peyto",
        forcing_name="revised_T6m_preserve_ea2m",
        T_C=peyto_valid["T_6m_C"],
        RH_pct=peyto_valid["RH_equiv_6m_pct"],
        qh_obs=peyto_valid["EC2QH"],
        qe_obs=peyto_valid["AdjEC2QE"],
        altitude_m=2169,
    )
    all_results.append(res)

    print("Peyto mean-match Kb:")
    print("  legacy forcing:", peyto_kb_legacy)
    print("  revised 6 m / preserved ea forcing:", peyto_kb_revised)

    add_model_columns(
        peyto_valid,
        "legacy_Kb001",
        peyto_valid["T_2m_C"],
        peyto_valid["RH_2m_pct"],
        2169,
        0.01,
    )
    add_model_columns(
        peyto_valid,
        "legacy_Kbcal",
        peyto_valid["T_2m_C"],
        peyto_valid["RH_2m_pct"],
        2169,
        peyto_kb_legacy,
    )
    add_model_columns(
        peyto_valid,
        "T6m_preserveEa_Kb001",
        peyto_valid["T_6m_C"],
        peyto_valid["RH_equiv_6m_pct"],
        2169,
        0.01,
    )
    add_model_columns(
        peyto_valid,
        "T6m_preserveEa_Kbcal",
        peyto_valid["T_6m_C"],
        peyto_valid["RH_equiv_6m_pct"],
        2169,
        peyto_kb_revised,
    )

    peyto_valid.to_csv(
        INTERIM / "Peyto_kb_forcing_hourly.csv",
        index_label="time",
    )

    # =========================================================
    # ATHABASCA
    # =========================================================
    print("\n=== ATHABASCA ===")

    atha_profile_path = find_required(ATHA_RAW, "AthaIceProfile.txt")
    atha_ec_path = find_required(ATHA_RAW, "LE_H_atha2015.xlsx")

    atha_met = read_atha_profile_hourly(atha_profile_path)
    atha_ec = read_atha_ec_hourly(atha_ec_path)

    atha = atha_met.join(atha_ec, how="inner")

    atha_required = [
        "T_1p5m_C",
        "T_6m_C",
        "RH_1p5m_pct",
        "Qh_EC",
        "Qe_EC",
    ]

    atha_valid = atha.dropna(subset=atha_required).copy()

    # Preserve 1.5 m vapour pressure while using 6 m Ta.
    atha_valid["RH_equiv_6m_pct"], atha_valid["ea_source_hPa"] = (
        preserve_vapour_pressure_rh(
            atha_valid["T_1p5m_C"],
            atha_valid["RH_1p5m_pct"],
            atha_valid["T_6m_C"],
        )
    )

    print("Matched Athabasca hours:", len(atha_valid))
    print(
        "Mean RH: source 1.5 m =",
        round(atha_valid["RH_1p5m_pct"].mean(), 2),
        "%; equivalent at 6 m =",
        round(atha_valid["RH_equiv_6m_pct"].mean(), 2),
        "%",
    )

    # Legacy forcing
    res, atha_kb_legacy, atha_kb_legacy_ls = fit_and_evaluate(
        glacier="Athabasca",
        forcing_name="legacy_T1.5m_RH1.5m",
        T_C=atha_valid["T_1p5m_C"],
        RH_pct=atha_valid["RH_1p5m_pct"],
        qh_obs=atha_valid["Qh_EC"],
        qe_obs=atha_valid["Qe_EC"],
        altitude_m=2175,
    )
    all_results.append(res)

    # Revised forcing: 6 m Ta, preserving 1.5 m ea.
    res, atha_kb_revised, atha_kb_revised_ls = fit_and_evaluate(
        glacier="Athabasca",
        forcing_name="revised_T6m_preserve_ea1.5m",
        T_C=atha_valid["T_6m_C"],
        RH_pct=atha_valid["RH_equiv_6m_pct"],
        qh_obs=atha_valid["Qh_EC"],
        qe_obs=atha_valid["Qe_EC"],
        altitude_m=2175,
    )
    all_results.append(res)

    print("Athabasca mean-match Kb:")
    print("  legacy forcing:", atha_kb_legacy)
    print("  revised 6 m / preserved ea forcing:", atha_kb_revised)

    add_model_columns(
        atha_valid,
        "legacy_Kb001",
        atha_valid["T_1p5m_C"],
        atha_valid["RH_1p5m_pct"],
        2175,
        0.01,
    )
    add_model_columns(
        atha_valid,
        "legacy_Kbcal",
        atha_valid["T_1p5m_C"],
        atha_valid["RH_1p5m_pct"],
        2175,
        atha_kb_legacy,
    )
    add_model_columns(
        atha_valid,
        "T6m_preserveEa_Kb001",
        atha_valid["T_6m_C"],
        atha_valid["RH_equiv_6m_pct"],
        2175,
        0.01,
    )
    add_model_columns(
        atha_valid,
        "T6m_preserveEa_Kbcal",
        atha_valid["T_6m_C"],
        atha_valid["RH_equiv_6m_pct"],
        2175,
        atha_kb_revised,
    )

    atha_valid.to_csv(
        INTERIM / "Athabasca_kb_forcing_hourly.csv",
        index_label="time",
    )

    # =========================================================
    # COMBINED TABLE
    # =========================================================
    result = pd.concat(all_results, ignore_index=True)

    output_path = TABLES / "Kb_forcing_comparison.csv"
    result.to_csv(output_path, index=False)

    print("\n=== Kb FORCING COMPARISON COMPLETE ===")
    print(result.to_string(index=False))

    print("\nSaved:")
    print(output_path)
    print(INTERIM / "Peyto_kb_forcing_hourly.csv")
    print(INTERIM / "Athabasca_kb_forcing_hourly.csv")

    return result


kb_forcing_results = run_analysis()
