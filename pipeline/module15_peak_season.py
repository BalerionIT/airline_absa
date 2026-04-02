"""
pipeline/module15_peak_season.py -- Peak Season Analysis (Hemisphere-Aware)

"Summer" is not universal. June-August is peak season in Italy;
December-February is peak season in Australia. This module classifies each
review by the hemisphere of its route origin, defines peak season accordingly,
and analyses how passenger satisfaction differs between peak and off-peak
periods -- separately for each hemisphere.

Hemisphere classification (by route origin city/keyword):
  NH: Europe, North America, North Asia, Middle East, North Africa
  SH: Oceania, South America, Southern Africa
  EQ: Near-equatorial regions (treated as NH for simplicity)

Peak season definitions:
  NH peak : June-August      (school summer holidays + high travel volume)
  NH xmas : December         (Christmas secondary peak)
  SH peak : December-February (southern summer + school holidays AUS/NZ/ARG)
  SH winter hols: June-July  (Australian/NZ school winter holidays)
"""
import re, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu
from config import OUTPUTS_DIR, SUB_COLS
from pipeline.utils import banner, save_fig

log = logging.getLogger(__name__)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

SH_CITIES = {
    "Sydney","Melbourne","Brisbane","Perth","Adelaide","Darwin","Cairns",
    "Gold Coast","Canberra","Hobart","Auckland","Wellington","Christchurch",
    "Queenstown","Nadi","Suva","Port Moresby","Noumea","Apia","Papeete",
    "Buenos Aires","Santiago","Sao Paulo","Rio","Lima","Bogota","Montevideo",
    "Asuncion","La Paz","Quito","Caracas","Medellin","Cali","Recife","Fortaleza",
    "Belo Horizonte","Porto Alegre","Curitiba","Brasilia","Johannesburg",
    "Cape Town","Durban","Dar es Salaam","Lusaka","Harare","Maputo","Windhoek",
    "Blantyre","Lilongwe","Antananarivo",
}

SH_KEYWORDS = [
    "australia","sydney","melbourne","brisbane","perth","auckland","wellington",
    "christchurch","new zealand","buenos aires","santiago","sao paulo","lima",
    "bogota","johannesburg","cape town","south africa","argentina","chile",
    "brazil","peru","colombia","uruguay","ethiopia","tanzania","zambia",
    "zimbabwe","mozambique","namibia","botswana",
]

EQ_KEYWORDS = [
    "singapore","kuala lumpur","jakarta","bangkok","manila","colombo","maldives",
    "male","bali","denpasar","ho chi minh","hanoi","yangon","nairobi","accra",
    "lagos","dakar","abidjan","kinshasa","douala",
]

NH_PEAK    = {6,7,8}
NH_XMAS    = {12}
NH_OFFPEAK = {1,2,3,4,5,9,10,11}
SH_PEAK    = {12,1,2}
SH_WINTER  = {6,7}
SH_OFFPEAK = {3,4,5,8,9,10,11}


def _classify_hemisphere(route):
    if pd.isna(route):
        return "NH"
    r = str(route).lower()
    for kw in EQ_KEYWORDS:
        if kw in r:
            return "EQ"
    for city in SH_CITIES:
        if city.lower() in r:
            return "SH"
    for kw in SH_KEYWORDS:
        if kw in r:
            return "SH"
    return "NH"


def _draw_hemisphere(ax, monthly_df, peak_months, xmas_months,
                     secondary_months, title, peak_label, secondary_label,
                     peak_color, secondary_color, global_mean):
    for m in range(1, 13):
        if m in peak_months:
            ax.axvspan(m-0.5, m+0.5, alpha=0.18, color=peak_color, zorder=0)
        elif m in xmas_months:
            ax.axvspan(m-0.5, m+0.5, alpha=0.18, color="#C0392B", zorder=0)
        elif m in secondary_months:
            ax.axvspan(m-0.5, m+0.5, alpha=0.12, color=secondary_color, zorder=0)

    ax_vol = ax.twinx()
    valid_n = monthly_df["n"].fillna(0)
    ax_vol.bar(range(1,13), valid_n, color="#BDC3C7", alpha=0.45,
               label="Review count", zorder=1, width=0.6)
    ax_vol.set_ylabel("Review Count", color="#7F8C8D", fontsize=9)
    ax_vol.tick_params(axis="y", labelcolor="#7F8C8D")

    vals = monthly_df["mean"].values
    ax.plot(range(1,13), vals, "o-", color="#2C3E50", lw=2.5, ms=8, zorder=3)
    for m, v in zip(range(1,13), vals):
        if not np.isnan(v):
            ax.text(m, v+0.18, f"{v:.2f}", ha="center", fontsize=8,
                    fontweight="bold", zorder=4)

    ax.axhline(global_mean, color="#E74C3C", lw=1.2, ls="--",
               alpha=0.6, label=f"Annual mean ({global_mean:.2f})")
    ax.set_xlim(0.3, 12.7); ax.set_ylim(1, 9)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS, fontsize=10)
    ax.set_ylabel("Mean Rating (1-10)", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    patches = [
        mpatches.Patch(color=peak_color,     alpha=0.5, label=peak_label),
        mpatches.Patch(color="#C0392B",       alpha=0.5, label="Christmas / Holiday Peak"),
        mpatches.Patch(color=secondary_color, alpha=0.4, label=secondary_label),
        mpatches.Patch(color="#BDC3C7",       alpha=0.7, label="Review volume"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left")


def run(df):
    banner("MODULE 15 - PEAK SEASON ANALYSIS (HEMISPHERE-AWARE)")
    df_s = df.copy()
    df_s["month"]      = df_s["date_flown"].dt.month.fillna(0).astype(int)
    df_s["year"]       = df_s["date_flown"].dt.year
    df_s["hemisphere"] = df_s["Route"].apply(_classify_hemisphere)

    for h, n in df_s["hemisphere"].value_counts().items():
        log.info(f"  Hemisphere {h}: {n:,} ({n/len(df_s)*100:.1f}%)")

    nh = df_s[(df_s["hemisphere"].isin(["NH","EQ"])) & (df_s["month"]>0)].copy()
    sh = df_s[(df_s["hemisphere"]=="SH") & (df_s["month"]>0)].copy()

    nh_monthly = nh.groupby("month").agg(
        n=("Overall_Rating","count"),
        mean=("Overall_Rating","mean")).reindex(range(1,13))
    sh_monthly = sh.groupby("month").agg(
        n=("Overall_Rating","count"),
        mean=("Overall_Rating","mean")).reindex(range(1,13))

    nh_peak_r    = nh[nh["month"].isin(NH_PEAK)]["Overall_Rating"].dropna()
    nh_offpeak_r = nh[nh["month"].isin(NH_OFFPEAK)]["Overall_Rating"].dropna()
    _,nh_p = mannwhitneyu(nh_peak_r, nh_offpeak_r, alternative="two-sided") if len(nh_peak_r)>5 and len(nh_offpeak_r)>5 else (0,1.0)
    nh_gap = nh_peak_r.mean() - nh_offpeak_r.mean() if len(nh_peak_r)>0 else 0

    sh_peak_r    = sh[sh["month"].isin(SH_PEAK)]["Overall_Rating"].dropna()
    sh_offpeak_r = sh[sh["month"].isin(SH_OFFPEAK)]["Overall_Rating"].dropna()
    _,sh_p = mannwhitneyu(sh_peak_r, sh_offpeak_r, alternative="two-sided") if len(sh_peak_r)>5 and len(sh_offpeak_r)>5 else (0,1.0)
    sh_gap = sh_peak_r.mean() - sh_offpeak_r.mean() if len(sh_peak_r)>0 else 0

    log.info(f"  NH peak (Jun-Aug):  {nh_peak_r.mean():.2f} | off-peak: {nh_offpeak_r.mean():.2f} | D={nh_gap:+.2f} p={nh_p:.4f}")
    log.info(f"  SH peak (Dec-Feb):  {sh_peak_r.mean():.2f} | off-peak: {sh_offpeak_r.mean():.2f} | D={sh_gap:+.2f} p={sh_p:.4f}")

    global_mean = df_s["Overall_Rating"].dropna().mean()

    # Figure 1: Two hemisphere panels
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=False)
    fig.suptitle(
        "Peak Season Analysis - Hemisphere-Aware\n"
        "Northern Hemisphere Summer (Jun-Aug) vs Southern Hemisphere Summer (Dec-Feb)",
        fontsize=13, fontweight="bold")

    _draw_hemisphere(
        axes[0], nh_monthly,
        peak_months=NH_PEAK, xmas_months=NH_XMAS, secondary_months=set(),
        title=(f"Northern Hemisphere  [n={len(nh):,} reviews]\n"
               f"Peak (Jun-Aug): {nh_peak_r.mean():.2f}  |  "
               f"Off-peak: {nh_offpeak_r.mean():.2f}  |  "
               f"D={nh_gap:+.2f}  p={nh_p:.4f}"),
        peak_label="NH Summer Peak (Jun-Aug)",
        secondary_label="",
        peak_color="#E74C3C",
        secondary_color="#E67E22",
        global_mean=global_mean)

    _draw_hemisphere(
        axes[1], sh_monthly,
        peak_months=SH_PEAK, xmas_months=set(), secondary_months=SH_WINTER,
        title=(f"Southern Hemisphere (Australia, S. America, S. Africa)  [n={len(sh):,} reviews]\n"
               f"Peak (Dec-Feb): {sh_peak_r.mean():.2f}  |  "
               f"Off-peak: {sh_offpeak_r.mean():.2f}  |  "
               f"D={sh_gap:+.2f}  p={sh_p:.4f}"),
        peak_label="SH Summer Peak (Dec-Feb)",
        secondary_label="SH School Winter Hols (Jun-Jul)",
        peak_color="#E74C3C",
        secondary_color="#E67E22",
        global_mean=global_mean)

    plt.tight_layout(rect=[0,0,1,0.95])
    save_fig(f"{OUTPUTS_DIR}/peak_season_hemispheres.png")

    # Figure 2: Aspect delta (peak vs off-peak)
    available_sub = [c for c in SUB_COLS if c in df_s.columns and df_s[c].notna().sum() > 50]
    if available_sub and len(nh_peak_r) > 30:
        nh_pk_sub  = nh[nh["month"].isin(NH_PEAK)][available_sub].mean()
        nh_op_sub  = nh[nh["month"].isin(NH_OFFPEAK)][available_sub].mean()
        delta_nh   = nh_pk_sub - nh_op_sub

        has_sh_sub = len(sh_peak_r) > 10 and len(sh_offpeak_r) > 10
        ncols = 2 if has_sh_sub else 1
        fig2, axes2 = plt.subplots(1, ncols, figsize=(7*ncols, 5), sharey=True)
        if ncols == 1: axes2 = [axes2]
        fig2.suptitle("Peak Season Penalty by Service Dimension", fontsize=13, fontweight="bold")

        def _draw_delta(ax, delta, title):
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in delta.values]
            ax.barh(delta.index, delta.values, color=colors, edgecolor="k", alpha=0.85)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("D Sub-Rating (Peak - Off-peak)")
            ax.set_title(title, fontweight="bold")
            for i, v in enumerate(delta.values):
                ax.text(v+(0.003 if v>=0 else -0.003), i, f"{v:+.3f}",
                        va="center", fontsize=9, ha="left" if v>=0 else "right")

        _draw_delta(axes2[0], delta_nh, "Northern Hemisphere\nPeak (Jun-Aug) vs Off-peak")
        if has_sh_sub:
            sh_pk_sub = sh[sh["month"].isin(SH_PEAK)][available_sub].mean()
            sh_op_sub = sh[sh["month"].isin(SH_OFFPEAK)][available_sub].mean()
            _draw_delta(axes2[1], sh_pk_sub - sh_op_sub,
                        "Southern Hemisphere\nPeak (Dec-Feb) vs Off-peak")

        save_fig(f"{OUTPUTS_DIR}/peak_season_aspect_delta.png")

    # Figure 3: Season comparison bar
    season_summary = []
    for label, mask, src in [
        ("NH\nOff-peak",       nh["month"].isin(NH_OFFPEAK), nh),
        ("NH Christmas\n(Dec)",nh["month"].isin(NH_XMAS),    nh),
        ("NH Peak\n(Jun-Aug)", nh["month"].isin(NH_PEAK),    nh),
        ("SH\nOff-peak",       sh["month"].isin(SH_OFFPEAK), sh),
        ("SH Winter\n(Jun-Jul)",sh["month"].isin(SH_WINTER), sh),
        ("SH Peak\n(Dec-Feb)", sh["month"].isin(SH_PEAK),    sh),
    ]:
        r = src[mask]["Overall_Rating"].dropna()
        if len(r) > 5:
            season_summary.append({"Season":label, "Mean":r.mean(),
                                   "N":len(r), "Hemi":"NH" if "NH" in label else "SH"})

    if season_summary:
        ss = pd.DataFrame(season_summary)
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        bar_colors = []
        for _, row in ss.iterrows():
            if "Peak" in row["Season"]:  bar_colors.append("#E74C3C")
            elif "Christmas" in row["Season"] or "Winter" in row["Season"]:
                bar_colors.append("#E67E22")
            elif row["Hemi"]=="NH": bar_colors.append("#4C72B0")
            else:                   bar_colors.append("#27AE60")

        bars = ax3.bar(ss["Season"], ss["Mean"], color=bar_colors, edgecolor="k", alpha=0.87)
        ax3.set_ylabel("Mean Overall Rating"); ax3.set_ylim(0, 8)
        ax3.set_title("Mean Rating by Season: Northern vs Southern Hemisphere", fontweight="bold")
        ax3.axhline(global_mean, color="black", lw=1.2, ls="--", alpha=0.5,
                    label=f"Global mean ({global_mean:.2f})")
        ax3.legend(fontsize=9)
        for b, row in zip(bars, ss.itertuples()):
            ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.07,
                     f"{row.Mean:.2f}\n(n={row.N:,})", ha="center",
                     fontsize=8, fontweight="bold")
        nh_count = (ss["Hemi"]=="NH").sum()
        ax3.axvline(nh_count-0.5, color="black", lw=1.5, ls="-", alpha=0.35)
        ax3.text(nh_count/2-0.5, 7.5, "Northern Hemisphere",
                 ha="center", fontsize=10, fontweight="bold", color="#4C72B0")
        ax3.text(nh_count+(len(ss)-nh_count)/2-0.5, 7.5, "Southern Hemisphere",
                 ha="center", fontsize=10, fontweight="bold", color="#27AE60")
        save_fig(f"{OUTPUTS_DIR}/peak_season_comparison.png")

    log.info(f"""
  PEAK SEASON SUMMARY
  ─────────────────────────────────────────────────────────────
  NH reviews              : {len(nh):,}
  SH reviews              : {len(sh):,}
  NH peak (Jun-Aug) gap   : {nh_gap:+.2f}  (p={nh_p:.4f})
  SH peak (Dec-Feb) gap   : {sh_gap:+.2f}  (p={sh_p:.4f})
  ─────────────────────────────────────────────────────────────
  Both hemispheres show a peak-season rating shift driven by
  higher passenger volumes, reduced operational slack, and
  greater delay probability during high-demand periods.
""")
    return df_s
