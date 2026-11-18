# %% Figure 7 v5 — panel names inside, two-line y-axis labels
"""Figure 7 v5: panel labels inside, y-axis labels on two lines, EC in front.

Revised Figure 7 using Code 12's phase-consistent ablation series.

Preserves the four panels and the continuous Peyto modelled series:
  (a) Athabasca hourly
  (b) Athabasca daily
  (c) Peyto hourly
  (d) Peyto daily

IMPORTANT: The black line is *EC-derived model ablation*, calculated using
observed eddy-covariance turbulent fluxes and other energy-balance inputs.
It is NOT measured SR50 surface lowering (shown in Figure 6 instead).
Peyto EC-derived black series is intermittent; the other two parameterizations
run continuously over available meteorological forcing.

Daily sums require >=20 valid hourly estimates within a calendar day.
Peyto's individual EC field campaigns are therefore shown as valid observations
without suppressing continuous bulk-transfer / katabatic model estimates.
Only the graph is generated; source data/calibrations are never modified.

Run from VS Code or:
    python fig07_ablation_timeseries.py
    python fig07_ablation_timeseries.py --root "<repository-root>"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from project_config import PROJECT_ROOT
OUTPUT_STEM = "Fig07_hourly_daily_ablation_v5_EC_on_top"

# Same columns as Code 13 (from Code 12) and same colors as previous figures.
SCENARIOS = {
    "EC-derived": ("Ablation_EC_mm_h", "#171717"),
    "Bulk transfer": ("Ablation_Bulk_mm_h", "#40b39c"),
    "Katabatic": ("Ablation_Kata_T6_preserveEa_Kbcal_mm_h", "#e66101"),
}
MIN_DAILY_HOURS = 20
FONT_PT = 14


def resolve_root(explicit_root=None):
    root = Path(explicit_root).expanduser().resolve() if explicit_root is not None else PROJECT_ROOT
    if not (root / "data" / "interim" / "Peyto_ablation_partition_hourly.csv").is_file():
        raise FileNotFoundError(f"Cannot find ablation partition data under {root}; run stage 08 first")
    return root


def load_inputs(root: Path):
    interim = root / "data" / "interim"
    data = {}
    for glacier in ("Athabasca", "Peyto"):
        path = interim / f"{glacier}_ablation_partition_hourly.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run Code 12 first.")
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
        if not df.index.is_unique or df.index.isna().any():
            raise ValueError(f"{path.name}: invalid or duplicate timestamps")
        missing = [c for c, _ in SCENARIOS.values() if c not in df.columns]
        if missing:
            raise KeyError(f"{path.name} missing Code 12 columns: {missing}")
        data[glacier] = df
    return data


def daily_sum_with_coverage(series: pd.Series, min_hours: int = MIN_DAILY_HOURS):
    hourly = pd.to_numeric(series, errors="coerce")
    daily = hourly.resample("D").sum(min_count=1)
    count = hourly.resample("D").count()
    daily.loc[count < min_hours] = np.nan
    return daily


def style_axis(ax, glacier: str, daily: bool, letter: str):
    ax.grid(True, color="0.85", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=FONT_PT)
    # Put panel identifier inside the plot, without repeating the frequency.
    ax.text(0.015, 0.96, f"({letter}) {glacier}", transform=ax.transAxes,
            ha="left", va="top", fontsize=FONT_PT, fontweight="normal",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            zorder=12)
    ax.set_ylabel(
        "Daily ablation\n" + r"(mm w.e. day$^{-1}$)"
        if daily else "Hourly ablation\n" + r"(mm w.e. h$^{-1}$)",
        fontsize=FONT_PT,
        labelpad=8
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    if daily:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2 if glacier == "Athabasca" else 4))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2 if glacier == "Athabasca" else 4))
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(bottom=0)  # surface ablation rates cannot be negative
    ax.margins(x=0.01)


def create_figure(data, output: Path):
    matplotlib.rcParams.update({
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
    })
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 10.8))
    panels = [
        ("Athabasca", False, "a"),
        ("Athabasca", True, "b"),
        ("Peyto", False, "c"),
        ("Peyto", True, "d"),
    ]
    for ax, (glacier, is_daily, letter) in zip(axes, panels):
        df = data[glacier]
        for name, (column, color) in SCENARIOS.items():
            values = pd.to_numeric(df[column], errors="coerce")
            if is_daily:
                values = daily_sum_with_coverage(values)
            # NaN gaps in EC-derived black line remain gaps. Model series
            # are shown whenever their respective forcing is available.
            # Always draw the black EC-derived series ON TOP of both model curves.
            # Make it slightly thicker so coincident curves stay distinguishable.
            is_ec = (name == "EC-derived")
            ax.plot(values.index, values.to_numpy(dtype=float),
                    lw=1.65 if is_ec else 1.2,
                    zorder=6 if is_ec else (3 if name == "Bulk transfer" else 2),
                    color=color, label=name,
                    marker="o" if is_daily else None,
                    markersize=3.0 if is_daily and is_ec else (2.8 if is_daily else None),
                    markeredgewidth=0.0)
        style_axis(ax, glacier, is_daily, letter)
        if letter == "a":
            ax.legend(ncol=3, frameon=False, loc="upper right", fontsize=FONT_PT)

    # Keep the four original panels. No time-axis compression or
    # campaign masking is applied to Peyto's continuous model curves.
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.06, hspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output.with_suffix('.png')}")
    print(f"Saved: {output.with_suffix('.pdf')}")


def main(root=None):
    project = resolve_root(root)
    df = load_inputs(project)
    for glacier, data in df.items():
        print(f"{glacier}: {len(data)} hourly timestamps")
        for name, (col, _) in SCENARIOS.items():
            valid_hourly = int(data[col].notna().sum())
            valid_daily = int(daily_sum_with_coverage(data[col]).notna().sum())
            print(f"  {name:15s}: {valid_hourly:4d} hourly, {valid_daily:2d} daily values")
    output_dir = project / "figures" if (project / "data" / "interim").exists() else project
    create_figure(df, output_dir / OUTPUT_STEM)


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=Path, default=None)
        args = parser.parse_args()
        main(args.root)
