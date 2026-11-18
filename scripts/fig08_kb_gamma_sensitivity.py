# %%
"""Figure 8 v3 — RMSE sensitivity to Kb and gamma for two glaciers.

B09 manuscript revision
-----------------------
Retains only four root-mean-square-error (RMSE) surfaces:
    (a) Peyto Qh, (b) Peyto Qe,
    (c) Athabasca Qh, (d) Athabasca Qe.

Dashed zero-mean-bias contours are shown ONLY if the sensitivity grid actually
contains both negative and positive mean bias values. Note that Athabasca Qe
is positively biased at every tested parameter combination, and consequently
has no zero-bias contour.

Markers show:
   white open circle = reference Kb=0.01 m s^-1, gamma=0.005 deg C m^-1
   orange x = EXACT calibrated Kb from Kb_forcing_comparison.csv at the
              reference gamma (not rounded to the sensitivity grid).

This is a plotting-only script: it does NOT recompute the sensitivity analysis
and does not change model outputs, calibration or source CSVs. Run Code 05
first if Gamma_Kb_sensitivity_all.csv does not exist.

Inputs from katabatic/tables/
    Gamma_Kb_sensitivity_all.csv
    Kb_forcing_comparison.csv

Outputs in katabatic/figures/
    Fig08_RMSE_bias_sensitivity_v3_viridis_r.png
    Fig08_RMSE_bias_sensitivity_v3_viridis_r.pdf
    Fig08_RMSE_bias_sensitivity_v3_viridis_r_validation.csv

Windows usage:
    Run from the repository root, after executing the parameter sensitivity stage.
    python fig08_kb_gamma_sensitivity.py

Alternative for local data copies:
    python fig08_kb_gamma_sensitivity.py --root 'C:/path/to/katabatic'
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from project_config import PROJECT_ROOT
FONT_SIZE_PT = 14
REFERENCE_KB = 0.010
REFERENCE_GAMMA = 0.005
STEM = "Fig08_RMSE_bias_sensitivity_v3_viridis_r"

# Plot order: top row Peyto, bottom row Athabasca; Qh left and Qe right.
PANELS = [
    ("Peyto", "Qh", "a"),
    ("Peyto", "Qe", "b"),
    ("Athabasca", "Qh", "c"),
    ("Athabasca", "Qe", "d"),
]


def resolve_dirs(root_arg=None):
    project = Path(root_arg).expanduser().resolve() if root_arg else PROJECT_ROOT
    tables, figures = project / "tables", project / "figures"
    if not (tables / "Gamma_Kb_sensitivity_all.csv").exists():
        raise FileNotFoundError(f"Missing {tables / 'Gamma_Kb_sensitivity_all.csv'}; run stage 05 first")
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def load_data(tables):
    sensitivity_file = tables / "Gamma_Kb_sensitivity_all.csv"
    kb_file = tables / "Kb_forcing_comparison.csv"
    if not sensitivity_file.is_file() or not kb_file.is_file():
        raise FileNotFoundError(
            "Figure 8 requires these two files in the tables folder:\n"
            f"  {sensitivity_file}\n  {kb_file}\n"
            "Run Code 05 (sensitivity) and Code 02 (calibration) first."
        )
    df = pd.read_csv(sensitivity_file)
    kb_table = pd.read_csv(kb_file)
    required = {
        "glacier", "Kb_m_s", "gamma_C_m", "n",
        "Qh_RMSE_W_m2", "Qe_RMSE_W_m2", "Qh_bias_W_m2", "Qe_bias_W_m2",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Sensitivity CSV missing required columns: {missing}")
    if df[list(required - {"glacier"})].isnull().any().any():
        raise ValueError("Sensitivity CSV contains missing values in required numeric columns")
    kb_values = {}
    for glacier, forcing in [
        ("Peyto", "revised_T6m_preserve_ea2m"),
        ("Athabasca", "revised_T6m_preserve_ea1.5m"),
    ]:
        q = kb_table.loc[
            (kb_table["glacier"] == glacier)
            & (kb_table["forcing"] == forcing)
            & (kb_table["scenario"] == "Kb_mean_match"),
            "Kb_m_s",
        ]
        if len(q) != 1:
            raise ValueError(f"Expected exactly one mean-match calibration for {glacier}: found {len(q)}")
        kb_values[glacier] = float(q.iloc[0])
    return df, kb_values


def grid_from_rows(rows, variable):
    if rows.duplicated(["Kb_m_s", "gamma_C_m"]).any():
        raise ValueError("Duplicate Kb/gamma parameter pairs in sensitivity results")
    grid = rows.pivot(index="gamma_C_m", columns="Kb_m_s", values=variable)
    if grid.isna().any().any():
        raise ValueError(f"Missing grid cells for {variable}")
    return grid.columns.to_numpy(float), grid.index.to_numpy(float), grid.to_numpy(float)


def plot_panel(ax, rows, glacier, flux, letter, kb_cal):
    kb, gamma, z_rmse = grid_from_rows(rows, f"{flux}_RMSE_W_m2")
    _, _, z_bias = grid_from_rows(rows, f"{flux}_bias_W_m2")
    if len(rows) != len(kb) * len(gamma):
        raise ValueError(f"Incomplete sensitivity grid for {glacier}")
    if rows["n"].nunique() != 1:
        raise ValueError(f"Comparison sample size varies across parameter grid for {glacier}")

    # Separate colorbars preserve detail in panels with different flux scales.
    # Share consistent labeling (RMSE, W m-2) rather than implying common units
    # can be represented by a single color scale across very different ranges.
    im = ax.pcolormesh(kb, gamma, z_rmse, shading="auto", cmap="viridis_r",
                       rasterized=True)  # avoids hairline cell seams in vector PDF

    zero_crossed = bool(np.nanmin(z_bias) < 0 < np.nanmax(z_bias))
    if zero_crossed:
        co = ax.contour(kb, gamma, z_bias, levels=[0.0], colors="white",
                        linewidths=1.9, linestyles="--")
        # Version-safe path effects: Matplotlib's QuadContourSet API changed.
        effects = [pe.Stroke(linewidth=2.6, foreground="black"), pe.Normal()]
        if hasattr(co, "set_path_effects"):
            co.set_path_effects(effects)
        else:
            # Matplotlib releases where individual contour collections
            # own the Artist path effects instead of ContourSet itself.
            for collection in getattr(co, "collections", []):
                collection.set_path_effects(effects)

    # Add exact reference and calibrated points on every subplot.
    ax.plot(REFERENCE_KB, REFERENCE_GAMMA, linestyle="none", marker="o",
            markersize=8, markeredgewidth=1.3, markeredgecolor="black",
            markerfacecolor="white", zorder=10)
    cross, = ax.plot(kb_cal, REFERENCE_GAMMA, linestyle="none", marker="x",
                     markersize=10, markeredgewidth=2.2, color="#e66101", zorder=11)
    cross.set_path_effects([pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])

    ax.set_xlim(float(kb.min()), float(kb.max()))
    ax.set_ylim(float(gamma.min()), float(gamma.max()))
    ax.set_title(f"({letter}) {glacier}  $Q_{{{flux[-1]}}}$",
                 loc="left", fontsize=FONT_SIZE_PT, weight="normal", pad=7)
    ax.set_xlabel(r"$K_b$ (m s$^{-1}$)")
    ax.set_ylabel(r"$\gamma$ ($^\circ$C m$^{-1}$)")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(labelsize=FONT_SIZE_PT)

    return im, {
        "glacier": glacier,
        "flux": flux,
        "n": int(rows["n"].iloc[0]),
        "calibrated_Kb_m_s": kb_cal,
        "reference_Kb_m_s": REFERENCE_KB,
        "gamma_reference_C_per_m": REFERENCE_GAMMA,
        "RMSE_min_W_m2": float(z_rmse.min()),
        "RMSE_max_W_m2": float(z_rmse.max()),
        "bias_min_W_m2": float(z_bias.min()),
        "bias_max_W_m2": float(z_bias.max()),
        "zero_bias_contour_shown": zero_crossed,
    }


def make_figure(df, calibrated, out_dir):
    plt.rcParams.update({
        "font.size": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), constrained_layout=False)
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.09, top=0.89,
                        wspace=0.34, hspace=0.33)
    validation = []

    for ax, (glacier, flux, letter) in zip(axes.ravel(), PANELS):
        rows = df[df["glacier"] == glacier].copy()
        if rows.empty:
            raise ValueError(f"No sensitivity results for {glacier}")
        im, info = plot_panel(ax, rows, glacier, flux, letter, calibrated[glacier])
        # The right panels share the same gamma scale as the left panels.
        # Omitting repeated y-axis titles avoids crowding the adjacent colorbars.
        if flux == "Qe":
            ax.set_ylabel("")
        validation.append(info)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.027, aspect=19)
        cb.set_label(r"RMSE (W m$^{-2}$)", fontsize=FONT_SIZE_PT)
        cb.ax.tick_params(labelsize=FONT_SIZE_PT)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markeredgecolor="black", markerfacecolor="white", label="Reference"),
        Line2D([0], [0], marker="x", linestyle="none", markersize=10,
               markeredgewidth=2.2, color="#e66101", label="Calibrated $K_b$"),
        Line2D([0], [0], linestyle="--", linewidth=1.9,
               color="0.2", label="Zero mean bias"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.97),
               ncol=3, frameon=False, columnspacing=1.8, handletextpad=0.6)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / (STEM + "." + ext)
        fig.savefig(path, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.11, facecolor="white")
        print("Saved:", path)
    plt.close(fig)
    vfile = out_dir / (STEM + "_validation.csv")
    pd.DataFrame(validation).to_csv(vfile, index=False)
    print("Saved:", vfile)
    print(pd.DataFrame(validation).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    return pd.DataFrame(validation)


def main(root=None):
    tables, figures = resolve_dirs(root)
    sensitivity, calibrated_kb = load_data(tables)
    print("Figure 8 using", tables)
    print("Calibrated Kb values:", calibrated_kb)
    make_figure(sensitivity, calibrated_kb, figures)


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main()  # VS Code Interactive Window
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", default=None, help="katabatic project root or CSV directory")
        main(parser.parse_args().root)
