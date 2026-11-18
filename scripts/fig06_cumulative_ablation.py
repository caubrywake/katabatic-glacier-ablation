# %% Figure 6: cumulative surface ablation (EC, BT, calibrated katabatic vs SR50)
"""Code 13 / Figure 6 revision (v4): one Athabasca panel + four Peyto period panels.

Manuscript Figure 6: five panels:
  (a) Athabasca: full 12-day SR50 comparison interval
  (b)-(e) Peyto: one panel per retained SR50 comparison period,
       with each period reset to a zero ablation baseline.

There is NO combined Peyto overview panel.

Scientifically important conventions:
  Athabasca: 288 start-stamped hourly fluxes over [Jun18 00, Jun30 00).
  Peyto: four (start, end] intervals, in the exact Code 12 convention.
  Positive Qe condensation is liquid water, and ablation = melt + net
  ice-to-vapour loss as computed upstream in Code 12 (no recomputation).
  The Athabasca bulk-transfer scenario is missing ONLY its initial hour
  (Jun18 00 to 01); the plot shows its 287-hour cumulative sum starting
  at 01:00. That curve is not a complete 288-hour modeled ablation total.

The historical Code 12 period identifiers P1, P2, P4, P5 are RELABELLED
P1, P2, P3, P4 solely for plotting to match the revised manuscript Fig. 5.
No data are deleted or renumbered in the source files.

Run on the user's Windows machine after Code 12 (Code 13 not required):
  python fig06_cumulative_ablation.py
or via VS Code Interactive Window: Run Cell on this first # %% cell.

Optional:
  python fig06_cumulative_ablation.py --root "<repository-root>"

Inputs (Code 12):
  data/interim/{Athabasca,Peyto}_ablation_partition_hourly.csv
  data/interim/{Athabasca,Peyto}_SR50_aligned.csv
  tables/Ablation_partition_summary.csv
  tables/Ablation_partition_by_period.csv

Outputs (under <root>/figures):
  Fig06_cumulative_ablation_v4_five_panels.png/.pdf
  Fig06_cumulative_ablation_v4_checks.csv (includes BT coverage warning)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

from project_config import PROJECT_ROOT as DEFAULT_ROOT
RHO_ICE = 918.0  # SR50 displacement: m ice -> mm w.e.  (918 kg m-3)

FLUX_COLUMNS = {
    "EC": "Ablation_EC_mm_h",
    "BT": "Ablation_Bulk_mm_h",
    "Katabatic": "Ablation_Kata_T6_preserveEa_Kbcal_mm_h",
}
COLORS = {
    "EC": "#171717",
    "BT": "#40b39c",
    "Katabatic": "#e66101",
    "SR50": "#666666",
}
SCENARIO_TABLE_LABEL = {"EC": "EC", "BT": "Bulk", "Katabatic": "Kata_T6_preserveEa_Kbcal"}
# Validated Code 12 source slices (first timestamp = SR50 baseline).
PEYTO_SEGMENTS = [
    ("P1", "P1", 12, 61),
    ("P2", "P2", 274, 372),
    ("P3", "P4", 491, 582),
    ("P4", "P5", 732, 765),
]

FONT_SIZE = 14


def project_root() -> Path:
    return DEFAULT_ROOT


def load(root: Path):
    in_dir = root / "data" / "interim"
    summary_dir = root / "tables"
    models, sr = {}, {}
    for glacier in ("Athabasca", "Peyto"):
        for dest, suffix in [(models, "ablation_partition_hourly"), (sr, "SR50_aligned")]:
            f = in_dir / f"{glacier}_{suffix}.csv"
            if not f.exists():
                raise FileNotFoundError(f"Missing {f}. Run Code 12 first.")
            d = pd.read_csv(f, parse_dates=["time"]).set_index("time").sort_index()
            if not d.index.is_unique:
                raise ValueError(f"Duplicated time in {f}")
            dest[glacier] = d
        if not models[glacier].index.equals(sr[glacier].index):
            raise ValueError(f"{glacier}: model/SR50 time grids differ")
        for col in FLUX_COLUMNS.values():
            if col not in models[glacier].columns:
                raise ValueError(f"{glacier} is missing {col}; rerun Code 12")
    summaries = pd.read_csv(summary_dir / "Ablation_partition_summary.csv")
    periods = pd.read_csv(summary_dir / "Ablation_partition_by_period.csv")
    return models, sr, summaries, periods


def observed_ablation(sr: pd.DataFrame, first: int, last: int):
    # Each period has its own arbitrary zero because the SR50 was not tracked
    # continuously through the gaps; do NOT integrate across missing periods.
    vals = pd.to_numeric(sr["sr50_m"].iloc[first:last+1], errors="coerce").to_numpy(float)
    if not np.isfinite(vals).all():
        raise ValueError("Missing SR50 within a validated period")
    return (vals - vals[0]) * RHO_ICE


def ablation_period(model: pd.DataFrame, name: str, first: int, last: int):
    # Peyto intervals are (first, last], so initial baseline is zero.
    flux = pd.to_numeric(model[FLUX_COLUMNS[name]].iloc[first+1:last+1], errors="coerce").to_numpy(float)
    if not np.isfinite(flux).all():
        raise ValueError(f"{name}: missing flux inside Peyto validated segment {first}:{last}")
    return np.r_[0.0, np.cumsum(flux)]


def athabasca_series(model: pd.DataFrame, name: str):
    # These fluxes are START stamped, so the 288 intervals end at time[288].
    interval_flux = pd.to_numeric(model[FLUX_COLUMNS[name]].iloc[:-1], errors="coerce")
    if name == "BT":
        # One missing initial BT hourly interval, Jun18 00 -> 01.
        # No interpolated/assumed flux: accumulate the remaining 287 hours.
        bad = np.flatnonzero(~np.isfinite(interval_flux.to_numpy(float)))
        if not np.array_equal(bad, np.array([0])):
            raise ValueError(f"Athabasca BT unexpected missing intervals: {bad.tolist()}")
        remaining = interval_flux.iloc[1:].to_numpy(float)
        return model.index[1:], np.r_[0.0, np.cumsum(remaining)], 287, 288
    if not np.isfinite(interval_flux.to_numpy(float)).all():
        raise ValueError(f"Athabasca {name}: invalid hourly fluxes")
    return model.index, np.r_[0.0, np.cumsum(interval_flux.to_numpy(float))], 288, 288


def check_summaries(models, sr, summaries, periods):
    rows = []
    for glacier, model in models.items():
        for name in FLUX_COLUMNS:
            if glacier == "Peyto":
                modeled = sum(ablation_period(model, name, first, last)[-1]
                              for _, _, first, last in PEYTO_SEGMENTS)
                measured = sum(observed_ablation(sr[glacier], first, last)[-1]
                               for _, _, first, last in PEYTO_SEGMENTS)
                used, total = 271, 271
            else:
                _, cumul, used, total = athabasca_series(model, name)
                modeled = float(cumul[-1])
                measured = float(observed_ablation(sr[glacier], 0, len(sr[glacier]) - 1)[-1])

            tab = summaries[(summaries.glacier == glacier) &
                            (summaries.scenario == SCENARIO_TABLE_LABEL[name])]
            if len(tab) != 1:
                raise ValueError(f"Missing unique Code 12 summary for {glacier} {name}")
            tab = tab.iloc[0]
            if np.isfinite(tab["modeled_ablation_mmwe"]):
                if not np.isclose(modeled, tab["modeled_ablation_mmwe"], atol=0.01):
                    raise ValueError(f"{glacier} {name}: model {modeled:.5f} vs Code 12 {tab['modeled_ablation_mmwe']}")
            elif name != "BT" or glacier != "Athabasca":
                raise ValueError(f"Missing Code 12 total for {glacier} {name}")
            if not np.isclose(measured, tab["measured_SR50_ablation_mmwe"], atol=.01):
                raise ValueError(f"{glacier}: SR50 mismatch {measured:.5f} vs {tab['measured_SR50_ablation_mmwe']}")
            rows.append(dict(glacier=glacier, scenario=name,
                             hourly_intervals_used=used, hourly_intervals_expected=total,
                             modeled_ablation_mmwe=modeled,
                             observed_sr50_ablation_mmwe=measured,
                             difference_mmwe=modeled - measured,
                             percent_difference=100 * (modeled - measured) / measured,
                             complete_coverage=(used == total),
                             note=("Athabasca BT first hour missing; valid-hours-only partial total"
                                   if glacier == "Athabasca" and name == "BT" else "")))
    # Additionally verify each Peyto period against Code12's per-period outputs.
    for _, legacy_label, first, last in PEYTO_SEGMENTS:
        for name in FLUX_COLUMNS:
            d = periods[(periods.glacier == "Peyto") &
                        (periods.period == legacy_label) &
                        (periods.scenario == SCENARIO_TABLE_LABEL[name])]
            if len(d) != 1:
                raise ValueError(f"Missing Code 12 Peyto period {legacy_label}/{name}")
            m = ablation_period(models["Peyto"], name, first, last)[-1]
            if not np.isclose(m, float(d.iloc[0]["modeled_ablation_mmwe"]), atol=.01):
                raise ValueError(f"Peyto {legacy_label}/{name}: mismatch {m} vs {float(d.iloc[0]['modeled_ablation_mmwe'])}")
    out = pd.DataFrame(rows)
    print("\nVerified Code 12 cumulative ablation: \n", out.round(2).to_string(index=False))
    return out


def common_style(ax):
    ax.grid(alpha=.23, color=".65", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.set_ylabel("Cumulative ablation\n (mm w.e.)", fontsize=FONT_SIZE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))


def draw_line(ax, x, values, name, *, label=True, lw=None):
    kwargs = {"color": COLORS[name], "lw": lw or (1.9 if name == "SR50" else 1.55),
              "label": "SR50 ablation" if name == "SR50" else
                       ("Katabatic" if name == "Katabatic" else name)}
    if name == "SR50":
        kwargs["linestyle"] = (0, (1.4, 2.2))
    ax.plot(x, values, **kwargs)




def nice_top(y):
    """Rounded independent y-limits, with headroom for the panel label."""
    if y <= 120:
        step = 20
    elif y <= 400:
        step = 50
    else:
        step = 100
    return step * int(np.ceil((y * 1.14) / step))


def plot_figure_6(models, sr, out_dir: Path):
    """One wide Athabasca panel, four Peyto panels (no combined overview)."""
    with plt.rc_context({
        "font.size": FONT_SIZE,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig = plt.figure(figsize=(11.8, 9.2), facecolor="white")
        gs = fig.add_gridspec(3, 2, height_ratios=[1.13, 1, 1])
        axes = [fig.add_subplot(gs[0, :])]
        axes.extend([
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
        ])
        fig.subplots_adjust(left=.115, right=.985, bottom=.083, top=.985,
                            hspace=.40, wspace=.30)

        # Panel (a): Athabasca. Model starts at measurement interval start;
        # Athabasca BT lacks the FIRST of 288 model intervals, not infilled.
        ax = axes[0]
        model = models["Athabasca"]
        for name in FLUX_COLUMNS:
            time, cumulative, _, _ = athabasca_series(model, name)
            draw_line(ax, time, cumulative, name)
        observed = observed_ablation(sr["Athabasca"], 0, len(sr["Athabasca"]) - 1)
        draw_line(ax, sr["Athabasca"].index, observed, "SR50")
        common_style(ax)
        ax.set_ylim(0, 1000)
        ax.set_xlim(model.index[0], model.index[-1])
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.text(.013, .94, "(a) Athabasca", transform=ax.transAxes,
                ha="left", va="top", fontsize=FONT_SIZE)
        ax.legend(loc="upper left", bbox_to_anchor=(.23, 1.0), ncol=4,
                  frameon=False, columnspacing=1.0, handlelength=2.4,
                  borderaxespad=0.5)
        
        # Panels (b)-(e): Peyto period plots, with their own ZERO baseline
        # and their own y ranges. There is deliberately no overview panel.
        peyto_model = models["Peyto"]
        peyto_sr = sr["Peyto"]
        records = []
        for ax, panel_letter, (period_label, original_label, first, last) in zip(
            axes[1:], "bcde", PEYTO_SEGMENTS
        ):
            dates = peyto_model.index[first:last + 1]
            maxima = []
            for name in FLUX_COLUMNS:
                c = ablation_period(peyto_model, name, first, last)
                draw_line(ax, dates, c, name)
                maxima.append(float(np.nanmax(c)))
                records.append({"glacier": "Peyto", "period": period_label,
                                "original_period": original_label,
                                "method": name, "start": str(dates[0]),
                                "end": str(dates[-1]),
                                "interval_count": last - first,
                                "cumulative_ablation_mmwe": c[-1]})
            measured = observed_ablation(peyto_sr, first, last)
            draw_line(ax, dates, measured, "SR50")
            maxima.append(float(np.nanmax(measured)))
            records.append({"glacier": "Peyto", "period": period_label,
                            "original_period": original_label,
                            "method": "SR50", "start": str(dates[0]),
                            "end": str(dates[-1]),
                            "interval_count": last - first,
                            "cumulative_ablation_mmwe": measured[-1]})
            common_style(ax)
            ax.set_ylim(0, nice_top(max(maxima)))
            ax.set_xlim(dates[0], dates[-1])
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.text(.025, .94, f"({panel_letter}) Peyto {period_label}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=FONT_SIZE)
            if ax in (axes[2], axes[4]):
                ax.set_ylabel("")   # left column carries y-axis titles
            # X ticks present in all four small panels, formatted MM/DD.
            ax.tick_params(axis="x", rotation=0)

        path = out_dir / "Fig06_cumulative_ablation_v4_five_panels"
        fig.savefig(path.with_suffix(".png"), dpi=300,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print("Saved:", path.with_suffix(".png"), "and", path.with_suffix(".pdf"))
        pd.DataFrame(records).to_csv(
            out_dir / "Fig06_cumulative_ablation_v4_period_totals.csv", index=False)


def run(root: Path):
    models, sr, summaries, periods = load(root)
    checked = check_summaries(models, sr, summaries, periods)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plot_figure_6(models, sr, output)
    checked.to_csv(output / "Fig06_cumulative_ablation_v4_checks.csv", index=False)
    print("Complete. Source data and Code 12 outputs have NOT been modified.")


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        run(project_root())
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=Path, default=project_root(),
                            help="Root of data/output tree; repository by default")
        run(parser.parse_args().root)
