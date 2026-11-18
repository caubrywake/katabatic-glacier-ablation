"""03 - Build final hourly turbulent-flux and melt scenarios.

Key correction from v2
----------------------
The Athabasca bulk-transfer calculation must use the glacier AWS meteorology,
not the profile-mast wind channel.

Legacy/manuscript forcing:
    T, RH : ice AWS (~1.5 m)
    U     : ice AWS (~2.0 m)
    bulk reference height z = 1.5 m (matching the active manuscript-era script)

The previous v2 script used the profile 1.5-m wind channel, which has long
zero/missing periods and caused incomplete bulk-transfer coverage.

Required Athabasca AWS input
-----------------------------
Preferred:
    data/raw/athabasca/Athabasca_AWSice_hourly.csv

with columns:
    time, T_C, RH_pct, U_ms

Alternative:
    data/raw/athabasca/iceAWs_Conway.obs

The OBS parser assumes the legacy CRHM file layout used by the MATLAB scripts:
first six data columns after the timestamp, where columns 1/2/3 are T/RH/U.

Outputs overwrite the v2 melt products:
    data/interim/Peyto_melt_scenarios_hourly.csv
    data/interim/Athabasca_melt_scenarios_hourly.csv
    tables/Melt_forcing_comparison.csv
"""

from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd

from project_config import PROJECT_ROOT, PEYTO_RAW, ATHA_RAW, INTERIM, TABLES, FIGURES
# Import shared modules from the shipped scripts/ directory (not from a machine-specific root).
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline_common import apply_peyto_ec_qc

from src.io_peyto import build_profile_a_hourly, build_ec_hourly
from src.thermodynamics import saturation_vapor_pressure_hpa
from src.turbulent_fluxes import katabatic_transfer, bulk_transfer

LF = 3.169e5

PROFILE_NAMES = [
    "ProfileA_tue_20080721_20080728.xlsx",
    "ProfileA_tue_20080803_20080808.xlsx",
    "ProfileA_tue_20080812_20080817.xlsx",
    "ProfileA_tue_20080823_20080825.xlsx",
]

print("=== STARTING SCRIPT 03: BUILD MELT SCENARIOS ===")
print("Project root:", PROJECT_ROOT)


def find_optional(root: Path, name: str):
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def find_required(root: Path, name: str) -> Path:
    p = find_optional(root, name)
    if p is None:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return p


def preserve_vapour_pressure_rh(T_source_C, RH_source_pct, T_target_C):
    Ts = np.asarray(T_source_C, float) + 273.15
    Tt = np.asarray(T_target_C, float) + 273.15
    RH = np.asarray(RH_source_pct, float)
    ea = RH / 100.0 * saturation_vapor_pressure_hpa(Ts)
    return 100.0 * ea / saturation_vapor_pressure_hpa(Tt)


def read_calibrated_kb(glacier, forcing):
    tab = pd.read_csv(TABLES / "Kb_forcing_comparison.csv")
    hit = tab[
        (tab["glacier"] == glacier)
        & (tab["forcing"] == forcing)
        & (tab["scenario"] == "Kb_mean_match")
    ]
    if len(hit) != 1:
        raise ValueError(f"Expected one Kb row for {glacier}/{forcing}")
    return float(hit.iloc[0]["Kb_m_s"])


def read_atha_profile_hourly(path):
    df = pd.read_csv(
        path, sep="\t", skiprows=[0, 2, 3],
        na_values=["NAN", "NaN", "nan"]
    )
    df["time"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    out = pd.DataFrame({
        "time": df["time"],
        "T_1p5m_C": pd.to_numeric(df["AirTemp_Avg(2)"], errors="coerce"),
        "RH_1p5m_pct": pd.to_numeric(df["RH_Avg(2)"], errors="coerce"),
        "T_6m_C": pd.to_numeric(df["AirTemp_Avg(5)"], errors="coerce"),
    })
    return out.dropna(subset=["time"]).set_index("time").resample("1h").mean()


def read_atha_ec_hourly(path):
    df = pd.read_excel(path)
    df["time"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df["Qh_EC"] = -pd.to_numeric(df["H"], errors="coerce")
    df["Qe_EC"] = -pd.to_numeric(df["LE"], errors="coerce")
    return (
        df.dropna(subset=["time"])
        .set_index("time")[["Qh_EC", "Qe_EC"]]
        .resample("1h").mean()
    )


def read_atha_aws_csv(path):
    df = pd.read_csv(path)
    required = {"time", "T_C", "RH_pct", "U_ms"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{path.name} must contain {sorted(required)}; missing {sorted(missing)}"
        )
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["T_C", "RH_pct", "U_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (
        df.dropna(subset=["time"])
        .set_index("time")[["T_C", "RH_pct", "U_ms"]]
        .resample("1h").mean()
        .rename(columns={
            "T_C": "T_AWS_C",
            "RH_pct": "RH_AWS_pct",
            "U_ms": "U_AWS_ms",
        })
    )


def read_atha_aws_obs(path):
    """
    Parse legacy CRHM iceAWs_Conway.obs.
    Expected data rows begin with:
        YYYY MM DD HH MM  T  RH  U ...
    or an equivalent whitespace-delimited timestamp followed by values.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = re.split(r"\s+", s)

            # Try YYYY MM DD HH MM followed by at least T RH U.
            if len(parts) >= 8:
                try:
                    y, mo, d, h, mi = map(int, parts[:5])
                    t = pd.Timestamp(y, mo, d, h, mi)
                    vals = [float(parts[5]), float(parts[6]), float(parts[7])]
                    rows.append((t, *vals))
                    continue
                except Exception:
                    pass

            # Try a quoted/unquoted datetime in first two fields:
            # YYYY-MM-DD HH:MM ...
            if len(parts) >= 5:
                try:
                    t = pd.to_datetime(parts[0] + " " + parts[1])
                    vals = [float(parts[2]), float(parts[3]), float(parts[4])]
                    rows.append((t, *vals))
                    continue
                except Exception:
                    pass

    if not rows:
        raise ValueError(
            f"Could not parse {path}. Convert the old AWS file to "
            "Athabasca_AWSice_hourly.csv with columns time,T_C,RH_pct,U_ms."
        )

    df = pd.DataFrame(rows, columns=["time", "T_AWS_C", "RH_AWS_pct", "U_AWS_ms"])
    return df.set_index("time").resample("1h").mean()


def read_atha_aws_hourly():
    csv_path = find_optional(ATHA_RAW, "Athabasca_AWSice_hourly.csv")
    if csv_path is not None:
        print("Using Athabasca AWS:", csv_path)
        return read_atha_aws_csv(csv_path)

    obs_path = find_optional(ATHA_RAW, "iceAWs_Conway.obs")
    if obs_path is not None:
        print("Using Athabasca AWS:", obs_path)
        return read_atha_aws_obs(obs_path)

    raise FileNotFoundError(
        "\nAthabasca bulk transfer requires the ICE AWS forcing.\n"
        "Neither of these was found:\n"
        "  Athabasca_AWSice_hourly.csv\n"
        "  iceAWs_Conway.obs\n\n"
        "The old project stored the processed source as:\n"
        r"  D:\10_TurbulentFluxes\2_dataprocess\Athat2015_ConwayData_AWSice.mat"
        "\nand the CRHM version as:\n"
        r"  D:\10_TurbulentFluxes\7_CRHMexp\athabasca\iceAWs_Conway.obs"
    )


def read_energy_components(root, glacier, source_start, expected_rows, expected_end):
    """Read precomputed common Qn/Qp forcing without computing or modifying it.

    Each component has one energy-flux column in W m-2 and a timestamp that
    was assigned when extracting the legacy model output. Require the full
    expected timestamp series, rather than silently reconstructing it at run time.
    """
    expected = pd.date_range(source_start, periods=expected_rows, freq="1h")
    if expected[-1] != pd.Timestamp(expected_end):
        raise ValueError(f"{glacier}: expected energy-input date range is inconsistent")
    components = {}
    for term in ("Qn", "Qp"):
        path = find_required(root, f"{glacier}_{term}_hourly.csv")
        tab = pd.read_csv(path)
        col = f"{term}_W_m2"
        if list(tab.columns) != ["time", col]:
            raise ValueError(f"{path}: expected exactly columns ['time', '{col}']")
        if len(tab) != expected_rows:
            raise ValueError(f"{path}: expected {expected_rows} rows, found {len(tab)}")
        times = pd.to_datetime(tab["time"], errors="raise")
        if not np.array_equal(times.to_numpy(), expected.to_numpy()):
            raise ValueError(f"{path}: timestamps do not match the historical hourly forcing series")
        values = pd.to_numeric(tab[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{path}: missing or nonnumeric {term} data")
        components[term] = values.to_numpy(dtype=float)
    return pd.DataFrame(components, index=expected)


def add_katabatic(df, prefix, T_C, RH_pct, altitude_m, kb):
    m = katabatic_transfer(
        np.asarray(T_C, float) + 273.15,
        np.asarray(RH_pct, float),
        altitude_m=altitude_m,
        Kb=kb, Pr=5.0, gamma=0.005,
        pressure_mode="legacy_fixed",
    )
    df[f"Qh_{prefix}"] = m["Qh"]
    df[f"Qe_{prefix}"] = m["Qe"]


def add_bulk(df, prefix, T_C, RH_pct, U_ms, altitude_m, z_m):
    m = bulk_transfer(
        np.asarray(T_C, float) + 273.15,
        np.asarray(U_ms, float),
        np.asarray(RH_pct, float),
        altitude_m=altitude_m,
        z_m=z_m,
    )
    df[f"Qh_{prefix}"] = m["Qh"]
    df[f"Qe_{prefix}"] = m["Qe"]


def add_melt(df, scenario, qh_col, qe_col):
    df[f"Qm_{scenario}"] = df["Qn"] + df["Qp"] + df[qh_col] + df[qe_col]
    qm = df[f"Qm_{scenario}"].to_numpy(float)
    df[f"Melt_{scenario}_mm_h"] = np.maximum(qm, 0.0) * 3600.0 / LF


def summarize(df, glacier, scenario):
    melt = pd.to_numeric(df[f"Melt_{scenario}_mm_h"], errors="coerce")
    qm = pd.to_numeric(df[f"Qm_{scenario}"], errors="coerce")
    return {
        "glacier": glacier,
        "scenario": scenario,
        "n_valid_hours": int(melt.notna().sum()),
        "mean_Qm_W_m2": float(qm.mean()),
        "mean_melt_mm_h": float(melt.mean()),
        "max_melt_mm_h": float(melt.max()),
        "cumulative_melt_mm": float(melt.sum()),
    }


def build_peyto():
    target = pd.date_range("2008-07-24 01:00", "2008-08-25 00:00", freq="1h")

    energy = read_energy_components(
        PEYTO_RAW, "Peyto", "2008-07-22 01:00", 816, "2008-08-25 00:00",
    ).reindex(target)

    profiles = [find_required(PEYTO_RAW, n) for n in PROFILE_NAMES]
    met = build_profile_a_hourly(
        profiles,
        find_required(PEYTO_RAW, "PeytoMNH_2008_MOH.obs"),
    ).reindex(target)
    ec = build_ec_hourly(
        find_required(PEYTO_RAW, "EC_Munro2008_QH_QE.xlsx"),
        find_required(PEYTO_RAW, "EC_Munro2008_QE_corrected20220412.xlsx"),
    ).reindex(target)

    df = met.join(ec, how="left").join(energy, how="left")
    df = apply_peyto_ec_qc(df)
    df["RH_equiv_6m_pct"] = preserve_vapour_pressure_rh(
        df["T_2m_C"], df["RH_2m_pct"], df["T_6m_C"]
    )

    kb_legacy = read_calibrated_kb("Peyto", "legacy_T2m_RH2m")
    kb_revised = read_calibrated_kb("Peyto", "revised_T6m_preserve_ea2m")

    add_bulk(df, "Bulk", df["T_2m_C"], df["RH_2m_pct"], df["u_2m"], 2169, 2.0)
    add_katabatic(df, "Kata_legacy_Kb001", df["T_2m_C"], df["RH_2m_pct"], 2169, 0.01)
    add_katabatic(df, "Kata_legacy_Kbcal", df["T_2m_C"], df["RH_2m_pct"], 2169, kb_legacy)
    add_katabatic(df, "Kata_T6_preserveEa_Kb001", df["T_6m_C"], df["RH_equiv_6m_pct"], 2169, 0.01)
    add_katabatic(df, "Kata_T6_preserveEa_Kbcal", df["T_6m_C"], df["RH_equiv_6m_pct"], 2169, kb_revised)

    scenarios = [
        ("EC", "EC2QH", "AdjEC2QE"),
        ("Bulk", "Qh_Bulk", "Qe_Bulk"),
        ("Kata_legacy_Kb001", "Qh_Kata_legacy_Kb001", "Qe_Kata_legacy_Kb001"),
        ("Kata_legacy_Kbcal", "Qh_Kata_legacy_Kbcal", "Qe_Kata_legacy_Kbcal"),
        ("Kata_T6_preserveEa_Kb001", "Qh_Kata_T6_preserveEa_Kb001", "Qe_Kata_T6_preserveEa_Kb001"),
        ("Kata_T6_preserveEa_Kbcal", "Qh_Kata_T6_preserveEa_Kbcal", "Qe_Kata_T6_preserveEa_Kbcal"),
    ]
    rows = []
    for s, qh, qe in scenarios:
        add_melt(df, s, qh, qe)
        rows.append(summarize(df, "Peyto", s))
    return df, rows


def build_athabasca():
    target = pd.date_range("2015-06-18 00:00", "2015-06-30 00:00", freq="1h")

    energy = read_energy_components(
        ATHA_RAW, "Athabasca", "2015-06-18 00:00", 288, "2015-06-29 23:00",
    ).reindex(target)

    profile = read_atha_profile_hourly(
        find_required(ATHA_RAW, "AthaIceProfile.txt")
    ).reindex(target)
    aws = read_atha_aws_hourly().reindex(target)
    ec = read_atha_ec_hourly(
        find_required(ATHA_RAW, "LE_H_atha2015.xlsx")
    ).reindex(target)

    df = profile.join(aws, how="left").join(ec, how="left").join(energy, how="left")

    print("Athabasca AWS valid hours:")
    print("  T :", int(df["T_AWS_C"].notna().sum()))
    print("  RH:", int(df["RH_AWS_pct"].notna().sum()))
    print("  U :", int(df["U_AWS_ms"].notna().sum()))

    df["RH_equiv_6m_pct"] = preserve_vapour_pressure_rh(
        df["T_1p5m_C"], df["RH_1p5m_pct"], df["T_6m_C"]
    )

    kb_legacy = read_calibrated_kb("Athabasca", "legacy_T1.5m_RH1.5m")
    kb_revised = read_calibrated_kb("Athabasca", "revised_T6m_preserve_ea1.5m")

    # Corrected bulk forcing: AWS T/RH/U, not profile wind.
    add_bulk(
        df, "Bulk",
        df["T_AWS_C"], df["RH_AWS_pct"], df["U_AWS_ms"],
        altitude_m=2175, z_m=1.5
    )

    # Katabatic experiments retain the profile-based forcing used in code 05.
    add_katabatic(df, "Kata_legacy_Kb001", df["T_1p5m_C"], df["RH_1p5m_pct"], 2175, 0.01)
    add_katabatic(df, "Kata_legacy_Kbcal", df["T_1p5m_C"], df["RH_1p5m_pct"], 2175, kb_legacy)
    add_katabatic(df, "Kata_T6_preserveEa_Kb001", df["T_6m_C"], df["RH_equiv_6m_pct"], 2175, 0.01)
    add_katabatic(df, "Kata_T6_preserveEa_Kbcal", df["T_6m_C"], df["RH_equiv_6m_pct"], 2175, kb_revised)

    scenarios = [
        ("EC", "Qh_EC", "Qe_EC"),
        ("Bulk", "Qh_Bulk", "Qe_Bulk"),
        ("Kata_legacy_Kb001", "Qh_Kata_legacy_Kb001", "Qe_Kata_legacy_Kb001"),
        ("Kata_legacy_Kbcal", "Qh_Kata_legacy_Kbcal", "Qe_Kata_legacy_Kbcal"),
        ("Kata_T6_preserveEa_Kb001", "Qh_Kata_T6_preserveEa_Kb001", "Qe_Kata_T6_preserveEa_Kb001"),
        ("Kata_T6_preserveEa_Kbcal", "Qh_Kata_T6_preserveEa_Kbcal", "Qe_Kata_T6_preserveEa_Kbcal"),
    ]
    rows = []
    for s, qh, qe in scenarios:
        add_melt(df, s, qh, qe)
        rows.append(summarize(df, "Athabasca", s))
    return df, rows


def run_analysis():
    INTERIM.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    peyto, p_rows = build_peyto()
    atha, a_rows = build_athabasca()

    peyto.to_csv(INTERIM / "Peyto_melt_scenarios_hourly.csv", index_label="time")
    atha.to_csv(INTERIM / "Athabasca_melt_scenarios_hourly.csv", index_label="time")

    summary = pd.DataFrame(p_rows + a_rows)
    summary.to_csv(TABLES / "Melt_forcing_comparison.csv", index=False)

    print("\n=== MELT FORCING COMPARISON v3 ===")
    print(summary.to_string(index=False))

    return summary


melt_results = run_analysis()
