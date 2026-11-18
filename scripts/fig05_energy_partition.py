"""Manuscript Figure 5 (v7): phase-consistent energy flux with packed Peyto periods.

Inputs: Code 12's *_ablation_partition_hourly.csv files under data/interim.
Outputs: figures/Fig05_energy_partition_v7.png and .pdf (originals untouched).

Peyto periods (origAinal P1, P2, P4, P5) are displayed as P1-P4. The plotting
clock preserves actual hour-of-day in every retained observation and aligns
all midnight boundaries with the Athabasca panel. To fit the periods onto the
12-day Athabasca axis, small boundary crops affect the FIGURE ONLY:
  P1: last 2 hours omitted; P2: first 2 and last 1 hours omitted;
  P3: first 1 hour omitted; P4: no omissions.
No input file, data analysis, period statistic, or cumulative integral is changed.
The artificial breaks between campaigns are not real elapsed time.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Publication figure typography: every text element is 12 pt.
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
})

from project_config import PROJECT_ROOT, INTERIM as INPUT_DIR, FIGURES as OUTPUT_DIR

PEYTO_PERIODS = [
    # Label, inclusive source 0-based indices, display start in synthetic days,
    # number of observations removed from the beginning / end for plotting only.
    ("P1", 12, 61, 0, 0, 2),
    ("P2", 274, 372, 2, 2, 1),
    ("P3", 491, 582, 6, 1, 0),
    ("P4", 732, 765, 10, 0, 0),
]

COLORS = {"Qn": "#b2df8a", "Qh": "#1f78b4", "Qe": "#a6cee3"}


def load_hourly(glacier):
    fn = INPUT_DIR / f"{glacier}_ablation_partition_hourly.csv"
    if not fn.is_file():
        raise FileNotFoundError(f"Missing Code 12 hourly output: {fn}")
    df = pd.read_csv(fn, parse_dates=["time"]).set_index("time").sort_index()
    needed = {"Qn", "Qe_phase_EC_W_m2", "Qm_phase_EC_W_m2", "Qp",
              "Qh_EC" if glacier == "Athabasca" else "EC2QH"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{glacier} input is missing columns: {sorted(missing)}")
    return df


def site_fluxes(df, glacier):
    qh_col = "Qh_EC" if glacier == "Athabasca" else "EC2QH"
    return [df[col].astype(float).to_numpy() for col in
            ("Qn", qh_col, "Qe_phase_EC_W_m2", "Qm_phase_EC_W_m2")]


def athabasca_plot_data(df):
    start_midnight = df.index[0].normalize()
    x = ((df.index - start_midnight) / pd.Timedelta(days=1)).to_numpy(float)
    days = list(pd.date_range(start_midnight, periods=12, freq="D"))
    return x, days


def peyto_plot_data(df):
    frames = []
    period_positions = []
    date_by_day = {i: [] for i in range(12)}
    prev_last_x = None

    for label, start_idx, end_idx, display_day, remove_first, remove_last in PEYTO_PERIODS:
        original = df.iloc[start_idx:end_idx+1].copy()
        start_date = original.index[0].normalize()
        end_slice = len(original) - remove_last if remove_last else len(original)
        seg = original.iloc[remove_first:end_slice].copy()
        if seg.empty:
            raise ValueError(f"Plot trimming removed all records in {label}")
        x = display_day + ((seg.index - start_date) / pd.Timedelta(days=1)).to_numpy(float)
        seg["plot_x"] = x

        if prev_last_x is not None:
            if x[0] <= prev_last_x:
                raise ValueError(f"Display periods overlap at {label}: {x[0]} <= {prev_last_x}")
            # NaN data row guarantees lines and filled areas never connect.
            separator = seg.iloc[0:1].copy()
            separator.iloc[:, :] = np.nan
            separator["plot_x"] = 0.5 * (prev_last_x + x[0])
            frames.append(separator)

        frames.append(seg)
        period_positions.append((label, float(x[0]), float(x[-1]),
                                 len(original), len(seg),
                                 original.index[0], original.index[-1],
                                 seg.index[0], seg.index[-1]))
        prev_last_x = float(x[-1])

        # Label the midnight-to-midnight day consistently, including the days
        # representing two different original dates at a campaign boundary.
        for date in sorted(set(seg.index.normalize())):
            syn_day = display_day + int((date - start_date) / pd.Timedelta(days=1))
            if not 0 <= syn_day < 12:
                raise ValueError(f"Unexpected synthetic day {syn_day} in {label}")
            label_date = date.strftime("%m/%d")  # Windows compatible.
            if label_date not in date_by_day[syn_day]:
                date_by_day[syn_day].append(label_date)

    plotted = pd.concat(frames, axis=0).sort_values("plot_x")
    tick_labels = ["\n".join(date_by_day[i]) for i in range(12)]
    return plotted, period_positions, tick_labels


def signed_stack(ax, x, qn, qh, qe):
    positive = np.zeros(len(x))
    negative = np.zeros(len(x))
    for values, color in zip((qn, qh, qe), COLORS.values()):
        values = np.asarray(values, dtype=float)
        pos = np.where(np.isfinite(values) & (values > 0), values, 0.0)
        neg = np.where(np.isfinite(values) & (values < 0), values, 0.0)
        ax.fill_between(x, positive, positive + pos, where=pos != 0,
                        color=color, linewidth=0)
        ax.fill_between(x, negative, negative + neg, where=neg != 0,
                        color=color, linewidth=0)
        positive += pos
        negative += neg


def plot():
    atha = load_hourly("Athabasca")
    peyto = load_hourly("Peyto")
    xa, atha_days = athabasca_plot_data(atha)
    display_peyto, period_positions, peyto_day_labels = peyto_plot_data(peyto)
    xp = display_peyto["plot_x"].to_numpy(dtype=float)

    qa = site_fluxes(atha, "Athabasca")
    qp = site_fluxes(display_peyto, "Peyto")
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.3), sharex=False, sharey=True)
    plt.subplots_adjust(left=0.083, right=0.988, top=0.95, bottom=0.19, hspace=0.14)
    ax_a, ax_p = axes

    for ax, x, fluxes, label in [(ax_a, xa, qa, "(a)"),
                                 (ax_p, xp, qp, "(b)")]:
        qn, qh, qe, qm = fluxes
        signed_stack(ax, x, qn, qh, qe)
        ax.plot(x, qm, color="black", linewidth=1.1, zorder=5)
        ax.axhline(0, color="0.45", linewidth=0.7)
        ax.set_xlim(0, 12)
        ax.set_ylabel("Energy flux (W m$^{-2}$)")
        ax.text(0.012, 0.965, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=14, fontweight="normal")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22)
        # Every integer x is midnight on both synthetic axes.
        for midnight in range(13):
            ax.axvline(midnight, linewidth=0.65, color="0.85", zorder=0)

    # Same y range based on Athabasca, as Scott requested.
    vals = np.concatenate([arr[np.isfinite(arr)] for arr in qa])
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * (hi - lo)
    for ax in axes:
        ax.set_ylim(lo-pad, hi+pad)

    # Place date labels at day *centres*, but midnight grid lines at integers.
    tick_positions = np.arange(12) + 0.5
    ax_a.set_xticks(tick_positions)
    ax_a.set_xticklabels([d.strftime("%m/%d") for d in atha_days], fontsize=14)
    ax_a.tick_params(axis="x", labelbottom=True, pad=4, length=0)
    ax_p.set_xticks(tick_positions)
    ax_p.set_xticklabels(peyto_day_labels, fontsize=14)
    ax_p.tick_params(axis="x", pad=5, length=0)

    # Distinct campaign-boundary rules, using midpoints between last/first
    # plotted observations so the short break stays visible.
    for earlier, later in zip(period_positions[:-1], period_positions[1:]):
        x_boundary = (earlier[2] + later[1]) / 2
        ax_p.axvline(x_boundary, linewidth=1.0, color="0.38", ls="--")

    # Campaign labels below the lower date axis, centred on the shown data.
    for label, start_x, end_x, *_ in period_positions:
        ax_p.text(0.5*(start_x+end_x), -0.18, label,
                  transform=ax_p.get_xaxis_transform(), ha="center",
                  va="top", fontsize=14, fontweight="semibold", clip_on=False)

    legend_handles = [
        Patch(facecolor=COLORS["Qn"], label=r"$Q_n$"),
        Patch(facecolor=COLORS["Qh"], label=r"$Q_h$"),
        Patch(facecolor=COLORS["Qe"], label=r"$Q_e$"),
        Line2D([0], [0], color="black", lw=1.1, label=r"$Q_m$"),
    ]
    # One compact legend inside panel (a), immediately to the right of (a).
    # Its axes-relative anchor keeps it aligned with the panel rather than
    # occupying a separate band above the figure.
    ax_a.legend(handles=legend_handles, loc="upper left", ncol=4,
                bbox_to_anchor=(0.06, 0.99), borderaxespad=0,
                columnspacing=1.35, handlelength=1.6,
                frameon=False, fontsize=16)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = OUTPUT_DIR / f"Fig05_energy_partition_v7.{ext}"
        fig.savefig(dest, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("Saved:", dest)
    plt.close(fig)

    print("\nPeyto display-only trimming (no scientific input files changed):")
    for label, start_x, end_x, n0, n1, original_start, original_end, shown_start, shown_end in period_positions:
        print(f"  {label}: {n1}/{n0} hourly points shown, display x={start_x:.3f}–{end_x:.3f}; "
              f"{shown_start:%Y-%m-%d %H:%M} to {shown_end:%Y-%m-%d %H:%M}")
    print("Midnight-aligned synthetic figure extent: 12 days for both panels.")


# %% Run Figure 5 from the VS Code Interactive Window
plot()
