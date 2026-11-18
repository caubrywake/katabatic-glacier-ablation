"""05 - Gamma–Kb sensitivity analysis for Peyto and Athabasca.

Purpose
-------
Reproduce and extend the manuscript's original Peyto gamma–Kb sensitivity
analysis using the revised forcing adopted in the manuscript, and apply the
same analysis to Athabasca for a balanced two-glacier comparison.

Forcing used
------------
Peyto:
    Ta = 6 m
    vapour pressure preserved from the reliable 2 m T/RH observations

Athabasca:
    Ta = 6 m
    vapour pressure preserved from the reliable 1.5 m T/RH observations

These are exactly the revised forcing datasets written by:
    05_revised_katabatic_forcing_v2.py

The sensitivity analysis varies:
    Kb     : background turbulent-exchange coefficient (m s-1)
    gamma  : glacier temperature lapse-rate parameter (deg C m-1)

For every Kb–gamma pair, the script evaluates Qh and Qe separately against
eddy-covariance observations using:
    Pearson correlation coefficient r
    RMSE
    mean bias

Outputs
-------
tables/Gamma_Kb_sensitivity_all.csv
tables/Gamma_Kb_sensitivity_Peyto.csv
tables/Gamma_Kb_sensitivity_Athabasca.csv
tables/Gamma_Kb_sensitivity_optima.csv

figures/Fig08_gamma_Kb_sensitivity_both_glaciers.png
figures/Fig08_gamma_Kb_sensitivity_both_glaciers.pdf

Figure layout
-------------
Top row:    Pearson r
Bottom row: RMSE

Columns:
    Peyto Qh | Peyto Qe | Athabasca Qh | Athabasca Qe

Markers:
    open circle = manuscript reference parameters:
                  Kb = 0.01 m s-1, gamma = 0.005 deg C m-1

    x = revised glacier-specific calibrated Kb from script 02,
        evaluated at gamma = 0.005 deg C m-1

Notes
-----
- gamma must be > 0 because the katabatic parameterization contains
  sqrt(1/gamma).
- Correlation is relatively insensitive to Kb because Kb mainly changes the
  amplitude/offset of modeled turbulent exchange. The full surfaces are still
  useful because the manuscript historically evaluated both r and RMSE.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project_config import PROJECT_ROOT
from project_config import INTERIM, TABLES, FIGURES
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.turbulent_fluxes import katabatic_transfer

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

print("=== STARTING SCRIPT 05 ===")
print("Project root:", PROJECT_ROOT)


# ---------------------------------------------------------------------
# USER-ADJUSTABLE SENSITIVITY GRID
# ---------------------------------------------------------------------
# Chosen to reproduce the range of the original manuscript figure while
# providing smooth surfaces. Start gamma above zero to avoid division by zero.
KB_VALUES = np.linspace(0.001, 0.020, 77)
GAMMA_VALUES = np.linspace(0.00025, 0.00600, 70)

REFERENCE_KB = 0.010
REFERENCE_GAMMA = 0.005

PR = 5.0
K_COEFF = 0.0004


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{path}\n\n"
            "Run 05_revised_katabatic_forcing_v2.py first."
        )
    return path


def pair_metrics(obs, mod):
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(mod)
    obs = obs[mask]
    mod = mod[mask]

    if len(obs) == 0:
        return np.nan, np.nan, np.nan, 0

    rmse = np.sqrt(np.mean((mod - obs) ** 2))
    bias = np.mean(mod - obs)

    if len(obs) > 1 and np.std(obs) > 0 and np.std(mod) > 0:
        r = np.corrcoef(obs, mod)[0, 1]
    else:
        r = np.nan

    return float(r), float(rmse), float(bias), int(len(obs))


def load_site(glacier):
    if glacier == "Peyto":
        path = require(INTERIM / "Peyto_kb_forcing_hourly.csv")
        df = pd.read_csv(path, parse_dates=["time"])

        required = [
            "T_6m_C",
            "RH_equiv_6m_pct",
            "EC2QH",
            "AdjEC2QE",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{path.name} missing columns: {missing}")

        return {
            "glacier": "Peyto",
            "df": df,
            "T_C": df["T_6m_C"],
            "RH_pct": df["RH_equiv_6m_pct"],
            "Qh_obs": df["EC2QH"],
            "Qe_obs": df["AdjEC2QE"],
            "altitude_m": 2169,
            "forcing_label": "T6m_preserve_ea2m",
        }

    if glacier == "Athabasca":
        path = require(INTERIM / "Athabasca_kb_forcing_hourly.csv")
        df = pd.read_csv(path, parse_dates=["time"])

        required = [
            "T_6m_C",
            "RH_equiv_6m_pct",
            "Qh_EC",
            "Qe_EC",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{path.name} missing columns: {missing}")

        return {
            "glacier": "Athabasca",
            "df": df,
            "T_C": df["T_6m_C"],
            "RH_pct": df["RH_equiv_6m_pct"],
            "Qh_obs": df["Qh_EC"],
            "Qe_obs": df["Qe_EC"],
            "altitude_m": 2175,
            "forcing_label": "T6m_preserve_ea1.5m",
        }

    raise ValueError(glacier)


def calibrated_kb(glacier):
    tab = pd.read_csv(require(TABLES / "Kb_forcing_comparison.csv"))

    forcing = (
        "revised_T6m_preserve_ea2m"
        if glacier == "Peyto"
        else "revised_T6m_preserve_ea1.5m"
    )

    hit = tab[
        (tab["glacier"] == glacier)
        & (tab["forcing"] == forcing)
        & (tab["scenario"] == "Kb_mean_match")
    ]

    if len(hit) != 1:
        raise ValueError(
            f"Expected one calibrated Kb for {glacier}/{forcing}; "
            f"found {len(hit)}."
        )

    return float(hit.iloc[0]["Kb_m_s"])


def sensitivity_for_site(site):
    glacier = site["glacier"]

    T = pd.to_numeric(site["T_C"], errors="coerce").to_numpy(float)
    RH = pd.to_numeric(site["RH_pct"], errors="coerce").to_numpy(float)
    qh_obs = pd.to_numeric(site["Qh_obs"], errors="coerce").to_numpy(float)
    qe_obs = pd.to_numeric(site["Qe_obs"], errors="coerce").to_numpy(float)

    # One common mask for Qh and Qe so the two fluxes are evaluated over the
    # same observations at every grid point.
    valid = (
        np.isfinite(T)
        & np.isfinite(RH)
        & np.isfinite(qh_obs)
        & np.isfinite(qe_obs)
    )

    T_K = T[valid] + 273.15
    RH = RH[valid]
    qh_obs = qh_obs[valid]
    qe_obs = qe_obs[valid]

    print(f"\n{glacier}: {valid.sum()} common paired hours")

    rows = []

    for gamma in GAMMA_VALUES:
        for kb in KB_VALUES:
            model = katabatic_transfer(
                T_K,
                RH,
                altitude_m=site["altitude_m"],
                Kb=float(kb),
                Pr=PR,
                gamma=float(gamma),
                K=K_COEFF,
                pressure_mode="legacy_fixed",
            )

            qh_mod = model["Qh"]
            qe_mod = model["Qe"]

            qh_r, qh_rmse, qh_bias, qh_n = pair_metrics(qh_obs, qh_mod)
            qe_r, qe_rmse, qe_bias, qe_n = pair_metrics(qe_obs, qe_mod)

            rows.append(
                {
                    "glacier": glacier,
                    "forcing": site["forcing_label"],
                    "Kb_m_s": float(kb),
                    "gamma_C_m": float(gamma),
                    "n": min(qh_n, qe_n),
                    "Qh_r": qh_r,
                    "Qh_RMSE_W_m2": qh_rmse,
                    "Qh_bias_W_m2": qh_bias,
                    "Qe_r": qe_r,
                    "Qe_RMSE_W_m2": qe_rmse,
                    "Qe_bias_W_m2": qe_bias,
                }
            )

    return pd.DataFrame(rows)


def nearest_grid_row(df, kb, gamma):
    dist = (
        ((df["Kb_m_s"] - kb) / (KB_VALUES.max() - KB_VALUES.min())) ** 2
        + ((df["gamma_C_m"] - gamma) / (GAMMA_VALUES.max() - GAMMA_VALUES.min())) ** 2
    )
    return df.loc[dist.idxmin()]


def summarize_optima(df, glacier):
    """
    Summarize the optimum location for each diagnostic.

    Every output row contains the full set of diagnostics evaluated at that
    Kb-gamma pair. This avoids the misleading NaNs that appeared in the first
    version, where only the selected metric was populated.
    """
    rows = []

    diagnostic_cols = [
        "Qh_r",
        "Qe_r",
        "Qh_RMSE_W_m2",
        "Qe_RMSE_W_m2",
        "Qh_bias_W_m2",
        "Qe_bias_W_m2",
    ]

    for metric, objective in [
        ("Qh_r", "max"),
        ("Qe_r", "max"),
        ("Qh_RMSE_W_m2", "min"),
        ("Qe_RMSE_W_m2", "min"),
        ("Qh_bias_W_m2", "abs_min"),
        ("Qe_bias_W_m2", "abs_min"),
    ]:
        if objective == "max":
            idx = df[metric].idxmax()
        elif objective == "min":
            idx = df[metric].idxmin()
        else:
            idx = df[metric].abs().idxmin()

        r = df.loc[idx]

        row = {
            "glacier": glacier,
            "metric": metric,
            "objective": objective,
            "Kb_m_s": r["Kb_m_s"],
            "gamma_C_m": r["gamma_C_m"],
            "metric_value": r[metric],
        }
        for c in diagnostic_cols:
            row[c] = r[c]

        rows.append(row)

    # Also record the manuscript reference and revised calibrated point.
    kb_cal = calibrated_kb(glacier)

    for name, kb, gamma in [
        ("reference_Kb001_gamma005", REFERENCE_KB, REFERENCE_GAMMA),
        ("revised_calibrated_Kb_gamma005", kb_cal, REFERENCE_GAMMA),
    ]:
        r = nearest_grid_row(df, kb, gamma)

        row = {
            "glacier": glacier,
            "metric": name,
            "objective": "diagnostic_point",
            "Kb_m_s": r["Kb_m_s"],
            "gamma_C_m": r["gamma_C_m"],
            "metric_value": np.nan,
        }
        for c in diagnostic_cols:
            row[c] = r[c]

        rows.append(row)

    return pd.DataFrame(rows)


def grid_for_plot(df, metric):
    pivot = df.pivot(
        index="gamma_C_m",
        columns="Kb_m_s",
        values=metric,
    )

    x = pivot.columns.to_numpy(float)
    y = pivot.index.to_numpy(float)
    z = pivot.to_numpy(float)

    return x, y, z


def draw_surface(ax, df, metric, title, kb_cal):
    x, y, z = grid_for_plot(df, metric)

    mesh = ax.pcolormesh(
        x,
        y,
        z,
        shading="auto",
    )

    # Original/reference manuscript parameter pair
    ax.plot(
        REFERENCE_KB,
        REFERENCE_GAMMA,
        marker="o",
        markersize=7,
        markerfacecolor="none",
        markeredgewidth=1.5,
        linestyle="none",
        label="Reference",
    )

    # Revised calibrated Kb at the manuscript gamma
    ax.plot(
        kb_cal,
        REFERENCE_GAMMA,
        marker="x",
        markersize=8,
        markeredgewidth=1.8,
        linestyle="none",
        label="Revised calibrated Kb",
    )

    ax.set_xlabel(r"$K_b$ (m s$^{-1}$)")
    ax.set_ylabel(r"$\gamma$ ($^\circ$C m$^{-1}$)")
    ax.set_title(title)

    return mesh


def make_figure(peyto, atha):
    p_kb = calibrated_kb("Peyto")
    a_kb = calibrated_kb("Athabasca")

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.5, 7.2),
        sharex=False,
        sharey=True,
    )

    panels = [
        # top row: correlation
        (axes[0, 0], peyto, "Qh_r", "(a) Peyto $Q_h$: r", p_kb),
        (axes[0, 1], peyto, "Qe_r", "(b) Peyto $Q_e$: r", p_kb),
        (axes[0, 2], atha, "Qh_r", "(c) Athabasca $Q_h$: r", a_kb),
        (axes[0, 3], atha, "Qe_r", "(d) Athabasca $Q_e$: r", a_kb),

        # bottom row: RMSE
        (axes[1, 0], peyto, "Qh_RMSE_W_m2", "(e) Peyto $Q_h$: RMSE", p_kb),
        (axes[1, 1], peyto, "Qe_RMSE_W_m2", "(f) Peyto $Q_e$: RMSE", p_kb),
        (axes[1, 2], atha, "Qh_RMSE_W_m2", "(g) Athabasca $Q_h$: RMSE", a_kb),
        (axes[1, 3], atha, "Qe_RMSE_W_m2", "(h) Athabasca $Q_e$: RMSE", a_kb),
    ]

    for ax, df, metric, title, kb_cal in panels:
        mesh = draw_surface(ax, df, metric, title, kb_cal)
        cbar = fig.colorbar(mesh, ax=ax)
        if metric.endswith("_r"):
            cbar.set_label("Pearson r")
        else:
            cbar.set_label(r"RMSE (W m$^{-2}$)")

    # A single legend, using handles from first panel.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.suptitle(
        r"Sensitivity of katabatic turbulent-flux performance to $K_b$ and $\gamma$",
        y=1.04,
    )

    fig.tight_layout()

    png = FIGURES / "Fig08_gamma_Kb_sensitivity_both_glaciers.png"
    pdf = FIGURES / "Fig08_gamma_Kb_sensitivity_both_glaciers.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.show()

    print("\nSaved figure:")
    print(png)
    print(pdf)


def run_analysis():
    peyto_site = load_site("Peyto")
    atha_site = load_site("Athabasca")

    print("\nRunning Peyto sensitivity grid...")
    peyto = sensitivity_for_site(peyto_site)

    print("\nRunning Athabasca sensitivity grid...")
    atha = sensitivity_for_site(atha_site)

    all_df = pd.concat([peyto, atha], ignore_index=True)

    p_path = TABLES / "Gamma_Kb_sensitivity_Peyto.csv"
    a_path = TABLES / "Gamma_Kb_sensitivity_Athabasca.csv"
    all_path = TABLES / "Gamma_Kb_sensitivity_all.csv"

    peyto.to_csv(p_path, index=False)
    atha.to_csv(a_path, index=False)
    all_df.to_csv(all_path, index=False)

    optima = pd.concat(
        [
            summarize_optima(peyto, "Peyto"),
            summarize_optima(atha, "Athabasca"),
        ],
        ignore_index=True,
        sort=False,
    )

    opt_path = TABLES / "Gamma_Kb_sensitivity_optima.csv"
    optima.to_csv(opt_path, index=False)

    print("\n=== SENSITIVITY OPTIMA / DIAGNOSTIC POINTS ===")
    print(optima.to_string(index=False))

    make_figure(peyto, atha)

    print("\nSaved tables:")
    print(p_path)
    print(a_path)
    print(all_path)
    print(opt_path)

    return all_df, optima


gamma_kb_sensitivity, gamma_kb_optima = run_analysis()
