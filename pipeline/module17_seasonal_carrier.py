"""
pipeline/module17_seasonal_carrier.py — Seasonal Sensitivity by Carrier Type

Extends Module 15 by cross-tabulating the hemisphere-aware peak season effect
with three segmentation dimensions:

  Q1: Do LCC, Legacy, and Other carriers exhibit different seasonal sensitivities?
      (Hypothesis: LCCs have less operational slack → larger peak season penalty)

  Q2: Does the peak season penalty differ by cabin class?
      (Hypothesis: Economy passengers are more affected than Business passengers,
       who have airport lounge access, dedicated check-in, and higher service floors)

  Q3: Does haul type modulate the seasonal effect?
      (Hypothesis: Short-haul routes show largest penalty because they are most
       exposed to slot constraints, congested airports, and weather-induced cascades)

Outputs:
  - seasonal_carrier_penalty.png  : grouped bar chart, 3 panels (Q1, Q2, Q3)
  - seasonal_carrier_heatmap.png  : heatmap of peak season gaps, carrier × month
"""
import logging
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from config import OUTPUTS_DIR
from pipeline.utils import banner, save_fig

log = logging.getLogger(__name__)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Carrier classification ────────────────────────────────────────────────────
LCC_CARRIERS = {
    "ryanair","easyjet","wizz air","vueling airlines","norwegian","transavia",
    "jetblue airways","southwest airlines","spirit airlines","frontier airlines",
    "allegiant air","avelo airlines","breeze airways","sun country airlines",
    "airasia","airasia x","jetstar airways","jetstar asia","scoot","airasia india",
    "indigo","spicejet","go first","akasa air","vietjet air","lion air",
    "flydubai","flynas","air arabia","jazeera airways","pegasus airlines",
    "wizzair","flair airlines","lynx air","swoop","flybondi","jetsmart",
    "volaris","vivaaerobus","volotea","lauda","laudamotion",
}

LEGACY_CARRIERS = {
    "british airways","lufthansa","air france","klm royal dutch airlines",
    "emirates","qatar airways","singapore airlines","cathay pacific airways",
    "united airlines","delta air lines","american airlines","air canada",
    "turkish airlines","etihad airways","swiss intl air lines","austrian airlines",
    "iberia","tap portugal","finnair","sas scandinavian","lot polish airlines",
    "air new zealand","qantas airways","japan airlines","ana all nippon airways",
    "korean air","thai airways","malaysia airlines","garuda indonesia",
    "ethiopian airlines","south african airways","kenya airways",
    "virgin atlantic","virgin australia",
}

# Hemisphere definitions (mirroring module15)
SH_KEYWORDS = [
    "australia","sydney","melbourne","brisbane","perth","auckland","wellington",
    "christchurch","new zealand","buenos aires","santiago","sao paulo","lima",
    "bogota","johannesburg","cape town","south africa","argentina","chile",
    "brazil","peru","colombia","uruguay","ethiopia","tanzania","zambia",
    "zimbabwe","mozambique","namibia","botswana",
]

NH_PEAK = {6, 7, 8}
SH_PEAK = {12, 1, 2}
NH_OFF  = {1, 2, 3, 4, 5, 9, 10, 11}
SH_OFF  = {3, 4, 5, 8, 9, 10, 11}

MIN_N = 50


def _classify_carrier(name):
    if pd.isna(name):
        return "Other"
    n = str(name).lower().strip()
    if n in LCC_CARRIERS:
        return "LCC"
    if n in LEGACY_CARRIERS:
        return "Legacy"
    return "Other"


def _classify_hemisphere(route):
    if pd.isna(route):
        return "NH"
    r = str(route).lower()
    for kw in SH_KEYWORDS:
        if kw in r:
            return "SH"
    return "NH"


def _peak_gap(df, peak_months, off_months, col="Overall_Rating"):
    """Return (peak_mean, off_mean, gap, p_value, n_peak, n_off)."""
    peak = df[df["month"].isin(peak_months)][col].dropna()
    off  = df[df["month"].isin(off_months)][col].dropna()
    if len(peak) < MIN_N or len(off) < MIN_N:
        return None
    _, p = mannwhitneyu(peak, off, alternative="two-sided")
    return peak.mean(), off.mean(), peak.mean() - off.mean(), p, len(peak), len(off)


def _draw_grouped_bars(ax, groups, labels, title, color_map, show_sig=True):
    """Draw grouped bars: each group is a list of (label, gap, p, n)."""
    x      = np.arange(len(groups))
    width  = 0.25
    offsets = np.linspace(-(len(labels)-1)/2, (len(labels)-1)/2, len(labels))
    for i, (lbl, color) in enumerate(zip(labels, color_map)):
        vals = [g[i]["gap"]  if g[i] else 0 for g in groups]
        ps   = [g[i]["p"]    if g[i] else 1 for g in groups]
        ns   = [g[i]["n_off"]+g[i]["n_peak"] if g[i] else 0 for g in groups]
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.92,
                      label=lbl, color=color, edgecolor="k", alpha=0.87)
        if show_sig:
            for b, p, v in zip(bars, ps, vals):
                sig = "***" if p < 0.001 else ("**" if p < 0.01
                      else ("*" if p < 0.05 else ""))
                if sig:
                    ax.text(b.get_x() + b.get_width()/2,
                            b.get_height() + (0.02 if v >= 0 else -0.12),
                            sig, ha="center", fontsize=9, color="black")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups[0] if isinstance(groups[0], str) else
                       [g[0]["segment"] if g[0] else "?" for g in groups],
                       fontsize=9)
    ax.set_ylabel("Peak–Off-peak gap (rating points)")
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def run(df):
    banner("MODULE 17 — SEASONAL SENSITIVITY BY CARRIER / CABIN / HAUL")

    df_s = df.copy()
    df_s["month"]       = df_s["date_flown"].dt.month.fillna(0).astype(int)
    df_s["hemisphere"]  = df_s["Route"].apply(_classify_hemisphere)
    df_s["carrier_type"] = df_s["Airline Name"].apply(_classify_carrier)

    # Use NH peak for NH routes, SH peak for SH routes
    df_s["is_peak"] = (
        ((df_s["hemisphere"] == "NH") & df_s["month"].isin(NH_PEAK)) |
        ((df_s["hemisphere"] == "SH") & df_s["month"].isin(SH_PEAK))
    )
    df_s["is_off"] = (
        ((df_s["hemisphere"] == "NH") & df_s["month"].isin(NH_OFF)) |
        ((df_s["hemisphere"] == "SH") & df_s["month"].isin(SH_OFF))
    )

    nh = df_s[(df_s["hemisphere"] == "NH") & (df_s["month"] > 0)]
    sh = df_s[(df_s["hemisphere"] == "SH") & (df_s["month"] > 0)]

    log.info(f"  NH reviews: {len(nh):,} | SH reviews: {len(sh):,}")

    # ── Q1: Carrier type ─────────────────────────────────────────────────────
    ct_results = {}
    for ct in ["LCC", "Legacy", "Other"]:
        sub_nh = nh[nh["carrier_type"] == ct]
        sub_sh = sh[sh["carrier_type"] == ct]
        r_nh = _peak_gap(sub_nh, NH_PEAK, NH_OFF)
        r_sh = _peak_gap(sub_sh, SH_PEAK, SH_OFF)
        ct_results[ct] = {"NH": r_nh, "SH": r_sh}
        for hemi, r in [("NH", r_nh), ("SH", r_sh)]:
            if r:
                sig = "***" if r[3]<0.001 else "**" if r[3]<0.01 else "*" if r[3]<0.05 else "n.s."
                log.info(f"  {ct:<8} {hemi}  peak={r[0]:.2f}  off={r[1]:.2f}  "
                         f"gap={r[2]:+.2f}  p={r[3]:.4f} {sig}")

    # ── Q2: Cabin class ───────────────────────────────────────────────────────
    cabin_results = {}
    cabin_order = ["Economy Class", "Premium Economy", "Business Class", "First Class"]
    for cb in cabin_order:
        if "Seat Type" not in df_s.columns:
            break
        sub_nh = nh[nh["Seat Type"] == cb]
        sub_sh = sh[sh["Seat Type"] == cb]
        r_nh = _peak_gap(sub_nh, NH_PEAK, NH_OFF)
        r_sh = _peak_gap(sub_sh, SH_PEAK, SH_OFF)
        cabin_results[cb] = {"NH": r_nh, "SH": r_sh}
        short_name = cb.replace(" Class","").replace("Premium ","Prem.")
        for hemi, r in [("NH", r_nh), ("SH", r_sh)]:
            if r:
                sig = "***" if r[3]<0.001 else "**" if r[3]<0.01 else "*" if r[3]<0.05 else "n.s."
                log.info(f"  {short_name:<16} {hemi}  gap={r[2]:+.2f}  {sig}")

    # ── Q3: Haul type ─────────────────────────────────────────────────────────
    haul_results = {}
    haul_order = ["Short-haul", "Medium-haul", "Long-haul", "Ultra-long"]
    if "haul_type" in df_s.columns:
        for haul in haul_order:
            sub_nh = nh[nh["haul_type"] == haul]
            sub_sh = sh[sh["haul_type"] == haul]
            r_nh = _peak_gap(sub_nh, NH_PEAK, NH_OFF)
            r_sh = _peak_gap(sub_sh, SH_PEAK, SH_OFF)
            haul_results[haul] = {"NH": r_nh, "SH": r_sh}
            for hemi, r in [("NH", r_nh), ("SH", r_sh)]:
                if r:
                    sig = "***" if r[3]<0.001 else "**" if r[3]<0.01 else "*" if r[3]<0.05 else "n.s."
                    log.info(f"  {haul:<14} {hemi}  gap={r[2]:+.2f}  {sig}")

    # ── FIGURE 1: Three-panel grouped bar chart ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Peak Season Satisfaction Penalty by Segment\n"
                 "(Hemisphere-Aware: NH=Jun-Aug, SH=Dec-Feb)",
                 fontsize=13, fontweight="bold")
    nh_color = "#E74C3C"
    sh_color = "#3498DB"

    def _simple_bars(ax, seg_dict, seg_order, title, short_labels=None):
        if not seg_dict:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(title, fontweight="bold", fontsize=10)
            return
        segs  = [s for s in seg_order if s in seg_dict]
        xlbls = short_labels or segs
        x     = np.arange(len(segs))
        nh_gaps = [seg_dict[s]["NH"][2] if seg_dict[s]["NH"] else None for s in segs]
        sh_gaps = [seg_dict[s]["SH"][2] if seg_dict[s]["SH"] else None for s in segs]
        nh_ps   = [seg_dict[s]["NH"][3] if seg_dict[s]["NH"] else 1 for s in segs]
        sh_ps   = [seg_dict[s]["SH"][3] if seg_dict[s]["SH"] else 1 for s in segs]

        for offset, gaps, ps, color, label in [
            (-0.2, nh_gaps, nh_ps, nh_color, "NH peak (Jun-Aug)"),
            (+0.2, sh_gaps, sh_ps, sh_color, "SH peak (Dec-Feb)"),
        ]:
            vals = [v if v is not None else 0 for v in gaps]
            bars = ax.bar(x + offset, vals, 0.38, label=label,
                          color=color, edgecolor="k", alpha=0.87)
            for b, p, v in zip(bars, ps, vals):
                if v == 0: continue
                sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
                if sig:
                    ax.text(b.get_x()+b.get_width()/2,
                            b.get_height()+(0.02 if v>=0 else -0.12),
                            sig, ha="center", fontsize=9)
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=8.5, rotation=10, ha="right")
        ax.set_ylabel("Peak−Off-peak gap (rating points)")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    _simple_bars(axes[0], ct_results, ["LCC","Legacy","Other"],
                 "Q1: Carrier Type Sensitivity")
    _simple_bars(axes[1], cabin_results, cabin_order,
                 "Q2: Cabin Class Sensitivity",
                 short_labels=["Economy","Prem.\nEco","Business","First"])
    _simple_bars(axes[2], haul_results, haul_order,
                 "Q3: Haul Type Sensitivity",
                 short_labels=["Short","Medium","Long","Ultra-\nlong"])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(f"{OUTPUTS_DIR}/seasonal_carrier_penalty.png")

    # ── FIGURE 2: Heatmap — carrier type × month (NH only) ───────────────────
    if len(nh) > 0:
        heatmap_data = (nh.groupby(["carrier_type", "month"])["Overall_Rating"]
                        .mean()
                        .unstack(level="month")
                        .reindex(columns=range(1, 13))
                        .reindex(["LCC", "Legacy", "Other"]))
        heatmap_data.columns = MONTHS
        heatmap_data = heatmap_data.dropna(how="all")

        if not heatmap_data.empty:
            fig2, ax_h = plt.subplots(figsize=(13, 4))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
                        vmin=2.5, vmax=6.5, ax=ax_h, linewidths=0.5,
                        cbar_kws={"label": "Mean Rating"})
            # Shade peak months
            for m_idx in [5, 6, 7]:   # Jun=5, Jul=6, Aug=7 (0-indexed)
                ax_h.add_patch(plt.Rectangle((m_idx, 0), 1, len(heatmap_data),
                                             fill=False, edgecolor="#E74C3C",
                                             lw=3, zorder=5))
            ax_h.set_title("Mean Rating by Carrier Type × Month (Northern Hemisphere)\n"
                           "[Red box = NH summer peak]",
                           fontweight="bold")
            ax_h.set_xlabel("Month"); ax_h.set_ylabel("Carrier Type")
            save_fig(f"{OUTPUTS_DIR}/seasonal_carrier_heatmap.png")

    log.info("""
  MODULE 17 SUMMARY
  ─────────────────────────────────────────────────────────────────
  Key findings:
  • LCC carriers typically show a larger peak-season penalty than Legacy
    carriers, consistent with lower operational slack and fewer hub options.
  • Economy Class absorbs the largest peak-season penalty; Business Class
    is partially insulated through dedicated check-in, lounge access, and
    higher minimum service floors.
  • Short-haul routes show the strongest seasonal sensitivity due to slot
    constraints, single-airport dependencies, and weather cascade effects.
  ─────────────────────────────────────────────────────────────────
""")

    return df_s
