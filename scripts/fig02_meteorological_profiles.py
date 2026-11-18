"""Figure 2 v4 — smaller vertical display offsets, unchanged QC and statistics.

Revisions relative to Code 14 v3:
  - Halved vertical plotting offsets from ±0.09 to ±0.045 m.
  - All screening and scientific statistics are unchanged.

Features carried forward from v3:
  - All available valid hours used independently at EACH sensor height and site.
    No common-hour matching is imposed on the display means.
  - Peyto water-vapour pressure now includes 4- and 6-m measurements where
    the psychrometer-implied RH is 0 < RH <= 100%, with valid temperature.
    Upper Peyto points are PROVISIONAL, shown with open symbols and dashed
    profile extensions: physical RH screening cannot establish sensor health.
  - Peyto Toe and Upwind point heights are shifted -/+ 0.045 m FOR DISPLAY
    ONLY. Computations always use the true sensor heights (1, 2, 4, 6 m).
  - Outputs: PDF/PNG figure, per-height statistics, QC counts, paired-height
    sensitivity checks, and a text record of assumptions.

Input provenance:
  Profile A (Toe): ProfileA_tue_2008*.xlsx, 4 campaign workbooks.
  Profile B (Upwind): Peyto_ProfileB_original_consolidated.csv, from
                     PGProB108.xls ... PGProB408.xls.
  Athabasca: AthaIceProfile.txt.

Peyto files store calculated vapour pressure in Pa (despite an archival
header stating hPa). e [hPa] = e_raw [Pa] / 100, and RH = 100 e / es(T).
Athabasca e [hPa] = RH/100 * es(T). Saturation formula matches project
src/thermodynamics.py (water for T >= 0 C; ice for T < 0 C).

Edit PLOT_HEIGHT_OFFSET_M and SHOW_PROVISIONAL_UPPER_E as needed.
Script will look for data beside this file, then in the original
data/raw/{peyto,athabasca} in the repository.

Required packages: numpy, pandas, matplotlib, openpyxl (for .xlsx data).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from project_config import PROJECT_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
PEYTO_A_FILES = (
    "ProfileA_tue_20080721_20080728.xlsx",
    "ProfileA_tue_20080803_20080808.xlsx",
    "ProfileA_tue_20080812_20080817.xlsx",
    "ProfileA_tue_20080823_20080825.xlsx",
)
PEYTO_B_FILE = "Peyto_ProfileB_original_consolidated.csv"
ATHABASCA_FILE = "AthaIceProfile.txt"
OUTPUT_STEM = "Fig02_vapour_pressure_profiles_v4"
FONT_PT = 11
PLOT_HEIGHT_OFFSET_M = {"Peyto Toe": -0.045, "Peyto Upwind": +0.045, "Athabasca Ice": 0.0}
SHOW_PROVISIONAL_UPPER_E = True
RH_MIN_EXCLUSIVE = 0.0
RH_MAX_INCLUSIVE = 100.0
# Use only broad nonphysical-value screening; do not remove plausible extremes
# or make subjective spike corrections. All exclusions are counted in QC CSV.
T_MIN, T_MAX = -40.0, 40.0
U_MIN, U_MAX = 0.0, 50.0

COLORS = {
    "Peyto Toe": "#1876b8",
    "Peyto Upwind": "#d45124",
    "Athabasca Ice": "#d99a00",
}
MARKERS = {"Peyto Toe": "o", "Peyto Upwind": "D", "Athabasca Ice": "s"}
HEIGHTS_PEYTO = (1, 2, 4, 6)
HEIGHTS_ATHA_MET = (0.75, 1.5, 3.0, 4.5, 6.0)
HEIGHTS_ATHA_WIND = (0.75, 1.5, 3.0, 4.5, 6.0, 8.0)


def locate_input(name: str, glacier: str) -> Path:
    candidates = [SCRIPT_DIR / name]
    if PROJECT_ROOT.is_dir():
        candidates.extend([
            PROJECT_ROOT / name,
            PROJECT_ROOT / "data" / "raw" / glacier / name,
            PROJECT_ROOT / "data" / "raw" / glacier / "profileA" / name,
        ])
    for path in candidates:
        if path.is_file():
            return path
    if PROJECT_ROOT.is_dir():
        raw = PROJECT_ROOT / "data" / "raw" / glacier
        if raw.exists():
            found = sorted(raw.rglob(name))
            if found:
                return found[0]
    raise FileNotFoundError(f"Missing {name}. Put the original file beside this script or in {PROJECT_ROOT}/data/raw/{glacier}/")


def output_dir() -> Path:
    folder = PROJECT_ROOT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def valid_temperature(s):
    s = num(s)
    return s.where(s.between(T_MIN, T_MAX))


def valid_wind(s):
    s = num(s)
    return s.where(s.between(U_MIN, U_MAX))


def saturation_hpa(temp_c):
    """Match src/thermodynamics.py / Figure 2 v2 saturation convention."""
    kelvin = np.asarray(temp_c, dtype=float) + 273.15
    L = np.where(kelvin < 273.15, 2.83e6, 2.5e6)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return 6.113 * np.exp((L / 461.5) * (1 / 273.15 - 1 / kelvin))


def peyto_qc(temp, e_raw_pa, site, height, file_label, qc_records):
    """Screen per-observation e from T and archival psychrometric e [Pa]."""
    t_raw = num(temp)
    t = valid_temperature(t_raw)
    e = num(e_raw_pa) / 100.0
    implied_rh = 100.0 * e / saturation_hpa(t)
    bad_temp = t.isna()
    bad_e = e.isna() | e.le(0)
    rh_high = implied_rh.gt(RH_MAX_INCLUSIVE)
    rh_low = implied_rh.le(RH_MIN_EXCLUSIVE)
    good = ~bad_temp & ~bad_e & implied_rh.gt(RH_MIN_EXCLUSIVE) & implied_rh.le(RH_MAX_INCLUSIVE)
    qc_records.append({
        "site": site, "file": file_label, "height_m": height,
        "n_raw": len(t), "n_good_5min": int(good.sum()),
        "n_bad_temp": int(bad_temp.sum()), "n_bad_or_nonpositive_e": int(bad_e.sum()),
        "n_RH_over_100": int(rh_high.sum()), "n_RH_at_or_below_0": int(rh_low.sum()),
        "fraction_retained": float(good.mean()),
        "mean_implied_RH_valid_pct": float(implied_rh.where(good).mean()),
        "mean_e_valid_hPa": float(e.where(good).mean()),
    })
    return e.where(good)


def dates_from_doy_hhmm(doy, hhmm):
    d = num(doy).to_numpy(float)
    t = num(hhmm).to_numpy(float)
    base = pd.Timestamp("2008-01-01")
    stamps = []
    for di, ti in zip(d, t):
        if not (np.isfinite(di) and np.isfinite(ti)):
            stamps.append(pd.NaT)
            continue
        hm = int(round(ti))
        hour, minute = hm // 100, hm % 100
        if not (1 <= di <= 366 and 0 <= hour <= 24 and 0 <= minute < 60):
            stamps.append(pd.NaT)
            continue
        stamps.append(base + pd.Timedelta(days=int(di) - 1, hours=hour, minutes=minute))
    return pd.DatetimeIndex(stamps)


def read_peyto_toe(qc_records):
    periods = []
    for name in PEYTO_A_FILES:
        raw = pd.read_excel(locate_input(name, "peyto"), header=None, skiprows=3)
        if raw.shape[1] < 14:
            raise ValueError(f"Unexpected number of Profile A columns: {name}: {raw.shape[1]}")
        times = dates_from_doy_hhmm(raw.iloc[:, 0], raw.iloc[:, 1])
        values = {}
        for j, h in enumerate(HEIGHTS_PEYTO):
            t = valid_temperature(raw.iloc[:, 6 + j].reset_index(drop=True))
            u = valid_wind(raw.iloc[:, 2 + j].reset_index(drop=True))
            e = peyto_qc(t, raw.iloc[:, 10 + j].reset_index(drop=True),
                         "Peyto Toe", h, name, qc_records)
            values[f"T_{h}"] = t.to_numpy()
            values[f"u_{h}"] = u.to_numpy()
            values[f"e_{h}"] = e.to_numpy()
        periods.append(pd.DataFrame(values, index=times))
    combined = pd.concat(periods).sort_index()
    combined = combined.loc[combined.index.notna()]
    combined = combined.loc[~combined.index.duplicated(keep="first")]
    return combined.resample("1h").mean().dropna(how="all")


def read_peyto_upwind(qc_records):
    raw = pd.read_csv(locate_input(PEYTO_B_FILE, "peyto"), parse_dates=["time"])
    raw = raw.dropna(subset=["time"]).drop_duplicates("time").set_index("time").sort_index()
    values = {}
    for h in HEIGHTS_PEYTO:
        t = valid_temperature(raw[f"T_{h}"])
        u = valid_wind(raw[f"u_{h}"])
        e = peyto_qc(t, raw[f"e_rawPa_{h}"], "Peyto Upwind", h,
                     "PGProB[1-4]08.xls (consolidated)", qc_records)
        values[f"T_{h}"] = t
        values[f"u_{h}"] = u
        values[f"e_{h}"] = e
    return pd.DataFrame(values).resample("1h").mean().dropna(how="all")


def read_athabasca(qc_records):
    raw = pd.read_csv(locate_input(ATHABASCA_FILE, "athabasca"),
                      sep="\t", skiprows=[0, 2, 3],
                      na_values=["NAN", "NaN", "nan"], low_memory=False)
    raw["time"] = pd.to_datetime(raw["TIMESTAMP"], errors="coerce")
    raw = raw.dropna(subset=["time"]).set_index("time").sort_index()
    raw = raw.loc["2015-06-17":"2015-06-30"]
    out = {}
    for i, h in enumerate(HEIGHTS_ATHA_MET, 1):
        t = valid_temperature(raw[f"AirTemp_Avg({i})"])
        rh = num(raw[f"RH_Avg({i})"])
        good = t.notna() & rh.gt(RH_MIN_EXCLUSIVE) & rh.le(RH_MAX_INCLUSIVE)
        e = pd.Series((rh / 100.0) * saturation_hpa(t), index=raw.index).where(good)
        out[f"T_{h:g}"] = t
        out[f"e_{h:g}"] = e
        qc_records.append({
            "site": "Athabasca Ice", "file": ATHABASCA_FILE, "height_m": h,
            "n_raw": len(t), "n_good_5min": int(good.sum()),
            "n_bad_temp": int(t.isna().sum()), "n_bad_or_nonpositive_e": int((rh.isna() | rh.le(0)).sum()),
            "n_RH_over_100": int(rh.gt(100).sum()), "n_RH_at_or_below_0": int(rh.le(0).sum()),
            "fraction_retained": float(good.mean()),
            "mean_implied_RH_valid_pct": float(rh.where(good).mean()),
            "mean_e_valid_hPa": float(e.mean()),
        })
    wind = {h: valid_wind(raw[f"WindSpeed_Avg({i})"])
            for i, h in enumerate(HEIGHTS_ATHA_WIND, 1)}
    # Retain the previously identified 1.5-m wind sensor dropout screen.
    zero_dropouts = wind[1.5].eq(0) & wind[0.75].ge(1) & wind[3.0].ge(1)
    wind[1.5] = wind[1.5].mask(zero_dropouts)
    for h, s in wind.items():
        out[f"u_{h:g}"] = s
    return pd.DataFrame(out).resample("1h").mean().dropna(how="all"), int(zero_dropouts.sum())


def summarise_site(site, hourly):
    out = []
    for var, units in (("T", "degC"), ("e", "hPa"), ("u", "m/s")):
        for key in hourly.columns:
            if not key.startswith(var + "_"):
                continue
            h = float(key.split("_", 1)[1])
            s = num(hourly[key]).dropna()
            if s.empty:
                continue
            out.append({
                "site": site, "variable": var, "units": units,
                "height_m": h, "n_valid_hours": len(s),
                "mean": s.mean(), "sd": s.std(ddof=1),
                "min": s.min(), "max": s.max(),
                "first_hour": str(s.index.min()), "last_hour": str(s.index.max()),
                "provisional_upper_peyto_e": bool(var == "e" and site.startswith("Peyto") and h > 2),
            })
    return out


def paired_height_diagnostics(data):
    rows = []
    for name, df in data.items():
        heights = ("e_1", "e_6") if name.startswith("Peyto") else ("e_1.5", "e_6")
        pairs = df.loc[:, list(heights)].dropna()
        delta = pairs[heights[1]] - pairs[heights[0]]
        rows.append({
            "site": name, "lower_height_m": 1.0 if name.startswith("Peyto") else 1.5,
            "upper_height_m": 6.0, "n_common_valid_hours": len(pairs),
            "mean_e_6_minus_lower_hPa": delta.mean(),
            "median_e_6_minus_lower_hPa": delta.median(),
            "rmse_e_6_minus_lower_hPa": np.sqrt(np.mean(delta ** 2)),
            "pct_hours_abs_difference_over_2hPa": 100.0 * float((delta.abs() > 2).mean()),
        })
    return pd.DataFrame(rows)


def create_figure(stats, destination):
    plt.rcParams.update({
        "font.size": FONT_PT, "font.weight": "normal", "axes.labelsize": FONT_PT,
        "axes.labelweight": "normal", "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.5), sharey=True)
    axis_specs = (
        ("T", r"Air temperature ($^\circ$C)"),
        ("e", "Water-vapour pressure (hPa)"),
        ("u", r"Wind speed (m s$^{-1}$)"),
    )
    order = ("Peyto Toe", "Peyto Upwind", "Athabasca Ice")
    for idx, (ax, (var, xlabel)) in enumerate(zip(axes, axis_specs)):
        for site in order:
            sub = stats[(stats.site == site) & (stats.variable == var)].sort_values("height_m").copy()
            if var == "e" and site.startswith("Peyto") and not SHOW_PROVISIONAL_UPPER_E:
                sub = sub[sub.height_m <= 2]
            if sub.empty:
                continue
            x = sub["mean"].to_numpy(float)
            y = sub["height_m"].to_numpy(float) + PLOT_HEIGHT_OFFSET_M[site]
            err = sub["sd"].fillna(0).to_numpy(float)
            if var != "e" or site == "Athabasca Ice":
                ax.errorbar(x, y, xerr=err,
                            fmt="-" + MARKERS[site], markersize=4.5,
                            linewidth=1.6, elinewidth=0.9, capsize=2.0,
                            color=COLORS[site], zorder=3)
            else:
                # Upper Peyto e observations passed RH-only QC, but the
                # sensors remain unverified. Distinguish those points using
                # open markers and dashed upper-level line segments.
                lower = sub.height_m.to_numpy(float) <= 2.0
                upper = ~lower
                if lower.sum() > 1:
                    ax.plot(x[lower], y[lower], "-", color=COLORS[site], lw=1.6, zorder=2)
                if upper.any():
                    start = max(0, np.flatnonzero(upper)[0] - 1)
                    ax.plot(x[start:], y[start:], "--", color=COLORS[site], lw=1.4, zorder=2)
                for x_i, y_i, sd_i, is_upper in zip(x, y, err, upper):
                    ax.errorbar([x_i], [y_i], xerr=[sd_i], fmt=MARKERS[site],
                                markersize=4.8, elinewidth=0.9, capsize=2.0,
                                color=COLORS[site], markeredgecolor=COLORS[site],
                                markerfacecolor="white" if is_upper else COLORS[site],
                                zorder=4)
        ax.set_ylim(0, 8.25)
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_xlabel(xlabel)
        ax.text(0.025, 0.97, f"({chr(97 + idx)})", transform=ax.transAxes,
                va="top", ha="left", fontsize=FONT_PT, fontweight="normal")
        ax.grid(axis="x", color="0.88", linewidth=0.65)
        ax.tick_params(axis="both", labelsize=FONT_PT)
    axes[0].set_ylabel("Height above ice (m)")
    handles = [Line2D([0], [0], color=COLORS[site], marker=MARKERS[site],
                      linewidth=1.5, markersize=5) for site in order]
    fig.legend(handles, order, loc="upper center", frameon=False, ncol=3,
               bbox_to_anchor=(0.50, 0.99), fontsize=FONT_PT)
    fig.text(0.5, 0.032,
             "Peyto plotted heights offset ±0.045 m; open symbols/dashed lines: provisional 4–6 m vapour pressure (RH-screened only)",
             ha="center", va="bottom", fontsize=FONT_PT)
    fig.subplots_adjust(top=0.87, bottom=0.195, left=0.078, right=0.985, wspace=0.20)
    for artist in fig.findobj(match=plt.Text):
        artist.set_fontsize(FONT_PT)
        artist.set_fontweight("normal")
    sizes = sorted({round(t.get_fontsize(), 2) for t in fig.findobj(match=plt.Text)})
    assert sizes == [float(FONT_PT)], f"Inconsistent text sizes in output: {sizes}"
    png = destination / f"{OUTPUT_STEM}.png"
    pdf = destination / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    qc = []
    toe = read_peyto_toe(qc)
    upwind = read_peyto_upwind(qc)
    atha, atha_zero_wind = read_athabasca(qc)
    sites = {"Peyto Toe": toe, "Peyto Upwind": upwind, "Athabasca Ice": atha}
    stats = pd.DataFrame([row for site, df in sites.items() for row in summarise_site(site, df)])
    qc_df = pd.DataFrame(qc)
    pairs = paired_height_diagnostics(sites)
    dest = output_dir()
    stats_file = dest / f"{OUTPUT_STEM}_summary.csv"
    qc_file = dest / f"{OUTPUT_STEM}_QC_counts.csv"
    pairs_file = dest / f"{OUTPUT_STEM}_paired_height_diagnostics.csv"
    stats.to_csv(stats_file, index=False)
    qc_df.to_csv(qc_file, index=False)
    pairs.to_csv(pairs_file, index=False)
    png, pdf = create_figure(stats, dest)
    peyto_upper = qc_df[(qc_df.site.str.startswith("Peyto")) & (qc_df.height_m > 2)]
    notes = [
        "Figure 2 B02 v4 — all valid observations per height; 0 < RH <= 100% screen",
        "Source: Peyto Toe/Profile A (four ProfileA_tue_*.xlsx workbooks),",
        "Peyto Upwind/Profile B (PGProB[1-4]08.xls consolidated CSV),",
        "Athabasca Ice (AthaIceProfile.txt).",
        "Panels: (a) temperature, (b) vapour pressure, (c) wind; values are means +/-1 SD of hourly means.",
        "There is NO common-time mask across heights or stations for display profiles.",
        "Peyto heights offset vertically by -0.045 (Toe), +0.045 (Upwind) m for visibility only.",
        "Peyto e calculated from psychrometric vapour pressure in original Pa columns divided by 100.",
        "Implied RH = 100 * e / es(T), where es uses project temperature-dependent formula.",
        "Exclude RH>100%, RH<=0%, invalid e (<=0 or missing), or T outside -40 to 40 C.",
        "Peyto upper (4/6 m) psychrometer records are provisional even after physical screening.",
        "Open symbols and dashed extensions distinguish those provisional points.",
        "No sensor offset correction, temporal-spike rejection, flatline correction, or RH capping performed.",
        "T and wind use valid measurements independently; wind screened 0–50 m/s.",
        f"Athabasca anomalous exact-zero 1.5-m wind half-hour readings excluded: {atha_zero_wind}.",
        "Counts are recorded at original sampling resolution in QC_counts.csv and hourly in summary.csv.",
        "Critical limitation: RH plausibility filtering alone does NOT establish upper Peyto sensor reliability.",
        "Do not use the large Peyto upper e gradient as evidence for a real atmospheric gradient without independent checks.",
    ]
    for site in ("Peyto Toe", "Peyto Upwind"):
        for h in (4, 6):
            sub = peyto_upper[(peyto_upper.site == site) & (peyto_upper.height_m == h)]
            if not sub.empty:
                notes.append(f"{site} {h}m: {int(sub.n_good_5min.sum())}/{int(sub.n_raw.sum())} original samples retained; "+
                             f"{int(sub.n_RH_over_100.sum())} with RH>100% screened.")
    notes.extend(["", "Paired e-gradient checks (on simultaneous valid hours, NOT plotting means):", pairs.to_string(index=False)])
    notes_file = dest / f"{OUTPUT_STEM}_QC_notes.txt"
    notes_file.write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))
    print(f"\nFigure: {png} and {pdf}")
    print(f"Statistics: {stats_file}\nQC counts: {qc_file}\nPaired gradient check: {pairs_file}\nNotes: {notes_file}")
    return png, pdf, stats_file, qc_file, pairs_file, notes_file


if __name__ == "__main__":
    main()
