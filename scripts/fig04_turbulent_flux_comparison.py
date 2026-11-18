"""Figure 4 v5: continuous model curves with verified paired-sample statistics.

Inputs (from Code 03):
  data/interim/Athabasca_melt_scenarios_hourly.csv
  data/interim/Peyto_melt_scenarios_hourly.csv

Uses the common six-component validity mask from Code 07, requiring valid
Qh/Qe EC, Qh/Qe bulk transfer, and Qh/Qe calibrated katabatic values.
The EC observations, scatter plots and Table 2 are based on common paired
hours. The model time series continue through periods lacking valid EC
observations. QC applied to Peyto EC by Code 03 is preserved.

Outputs: figures/Fig04_flux_comparison_v5_continuous_models.{png,pdf}
         figures/Fig04_flux_comparison_v5_statistics.csv

To restrict all three time-series lines to paired hours instead, set
TIMESERIES_COMMON_ONLY=True. The scatter statistics never change.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from project_config import PROJECT_ROOT, INTERIM, FIGURES
FIGURES.mkdir(parents=True, exist_ok=True)

COL_EC = "black"
COL_BT = "#40b39c"
COL_KATA = "#e66101"
TIMESERIES_COMMON_ONLY = False
FONT_SIZE = 11
plt.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
                     "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
                     "legend.fontsize": FONT_SIZE})

SPECS = {
    "Athabasca": {
        "file": "Athabasca_melt_scenarios_hourly.csv",
        "obs_h": "Qh_EC", "obs_e": "Qe_EC"
    },
    "Peyto": {
        "file": "Peyto_melt_scenarios_hourly.csv",
        "obs_h": "EC2QH", "obs_e": "AdjEC2QE"
    },
}
BT_H, BT_E = "Qh_Bulk", "Qe_Bulk"
KATA_H = "Qh_Kata_T6_preserveEa_Kbcal"
KATA_E = "Qe_Kata_T6_preserveEa_Kbcal"


def load_site(name):
    spec = SPECS[name]
    path = INTERIM / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"Cannot find final Code 03 output: {path}")
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    cols = [spec["obs_h"], spec["obs_e"], BT_H, BT_E, KATA_H, KATA_E]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}")
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    common = pd.Series(np.isfinite(df[cols].to_numpy(dtype=float)).all(axis=1), index=df.index)
    return df, common


def get_metrics(obs, model):
    o = np.asarray(obs, float)
    m = np.asarray(model, float)
    return {"n": len(o), "mean_obs": float(o.mean()), "mean_model": float(m.mean()),
            "bias": float(np.mean(m - o)), "rmse": float(np.sqrt(np.mean((m - o)**2))),
            "r": float(np.corrcoef(o, m)[0, 1])}


def add_row(fig, spec, row, df, mask, flux, show_legend=False):
    obs_col = spec["obs_h"] if flux == "h" else spec["obs_e"]
    bt_col = BT_H if flux == "h" else BT_E
    kata_col = KATA_H if flux == "h" else KATA_E
    ax_ts, ax_sc = row
    times = df.index
    obs = df[obs_col].where(mask)
    bt = df[bt_col].where(mask) if TIMESERIES_COMMON_ONLY else df[bt_col]
    kata = df[kata_col].where(mask) if TIMESERIES_COMMON_ONLY else df[kata_col]

    ax_ts.plot(times, obs, label="EC", color=COL_EC, lw=1.2)
    ax_ts.plot(times, bt, label="BT", color=COL_BT, lw=1.05)
    ax_ts.plot(times, kata, label="Katabatic", color=COL_KATA, lw=1.05)
    ax_ts.grid(True, color="0.85", lw=0.7)
    ax_ts.set_axisbelow(True)
    ax_ts.margins(x=0)
    ax_ts.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    for tick in ax_ts.get_xticklabels():
        tick.set_rotation(0)
    ax_ts.set_ylabel((r"$Q_h$" if flux == "h" else r"$Q_e$") + r" (W m$^{-2}$)")
    if show_legend:
        ax_ts.legend(loc="upper left", bbox_to_anchor=(0.22, 0.99),
                     ncol=3, frameon=True, fancybox=False,
                     edgecolor="0.3", framealpha=1.0,
                     columnspacing=1.0, handlelength=2.0, borderaxespad=0.2)

    compared = df.loc[mask, [obs_col, bt_col, kata_col]]
    ax_sc.scatter(compared[obs_col], compared[bt_col], s=12, color=COL_BT, alpha=0.85)
    ax_sc.scatter(compared[obs_col], compared[kata_col], s=12, color=COL_KATA, alpha=0.85)
    all_values = compared.to_numpy(float)
    low = float(np.min(all_values)); high = float(np.max(all_values))
    pad = max((high-low)*0.055, 1.0)
    lo, hi = low-pad, high+pad
    ax_sc.plot([lo, hi], [lo, hi], ls="--", color="0.45", lw=1.0)
    ax_sc.set_xlim(lo, hi); ax_sc.set_ylim(lo, hi)
    ax_sc.grid(True, color="0.85", lw=0.7)
    ax_sc.set_axisbelow(True)
    ax_sc.set_xlabel(r"EC (W m$^{-2}$)")
    ax_sc.set_ylabel(r"Model (W m$^{-2}$)")
    ax_sc.set_box_aspect(1)
    ax_sc.text(0.04, 0.85, f"n = {len(compared)}", transform=ax_sc.transAxes,
               fontsize=FONT_SIZE, ha="left", va="top",
               bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.75))
    return [
        {"flux": "Q"+flux, "method": method, **get_metrics(compared[obs_col], compared[column])}
        for method, column in [("BT", bt_col), ("Katabatic", kata_col)]
    ]


def main():
    datasets = {name: load_site(name) for name in SPECS}
    fig = plt.figure(figsize=(12.2, 12.3))
    gs = fig.add_gridspec(4, 2, width_ratios=[2.5, 1.0], hspace=0.34, wspace=0.34)
    axes = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(4)]
    stats = []
    rows = [("Athabasca", "h"), ("Athabasca", "e"), ("Peyto", "h"), ("Peyto", "e")]
    for i, (name, flux) in enumerate(rows):
        df, mask = datasets[name]
        print(f"{name} Q{flux}: {int(mask.sum())} paired comparison hours")
        rowstats = add_row(fig, SPECS[name], axes[i], df, mask, flux, show_legend=(i==0))
        for d in rowstats:
            stats.append({"glacier": name, **d})
        axes[i][0].text(0.012, 0.94, f"({chr(ord('a')+2*i)}) {name}",
                        transform=axes[i][0].transAxes, fontsize=FONT_SIZE,
                        ha="left", va="top", fontweight="normal")
        axes[i][1].text(0.035, 0.98, f"({chr(ord('b')+2*i)})",
                        transform=axes[i][1].transAxes, fontsize=FONT_SIZE,
                        ha="left", va="top", fontweight="normal")
    axes[-1][0].set_xlabel("Time")
    fig.align_ylabels([r[0] for r in axes])
    out_csv = FIGURES / "Fig04_flux_comparison_v5_statistics.csv"
    pd.DataFrame(stats).to_csv(out_csv, index=False)
    for ext in ("png", "pdf"):
        out = FIGURES / f"Fig04_flux_comparison_v5_continuous_models.{ext}"
        fig.savefig(out, dpi=300 if ext=="png" else None,
                    bbox_inches="tight", pad_inches=0.12)
        print("Saved:", out)
    plt.close(fig)
    print("Statistics:", out_csv)
    print(pd.DataFrame(stats).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    main()
