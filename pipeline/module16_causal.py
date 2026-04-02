"""
pipeline/module16_causal.py — Causal Inference: Difference-in-Differences

Applies a Difference-in-Differences (DiD) design to estimate the causal
effect of major aviation disruption events on passenger satisfaction ratings.

Design:
  - Treatment window : 2–6 months after each event (post-event period)
  - Control window   : same calendar months from the prior year
                       (controls for seasonality without a parallel control group)
  - Estimand         : ATT — Average Treatment effect on the Treated
                       (the change in rating attributable to the event, net of
                       the seasonal baseline from the prior year)

Events analysed (must be within data window, sufficient pre/post coverage):
  COVID-19 pandemic        2020-03-11
  Boeing 737 MAX ban       2019-03-10
  Ukraine war / airspace   2022-02-24
  Post-COVID travel boom   2022-06-01
  EU aviation strikes      2023-01-01
  Boeing door blowout      2024-01-05
  CrowdStrike IT outage    2024-07-19

Interpretation note:
  DiD assumes parallel trends — that ratings in the treatment window would have
  followed the same seasonal trajectory as the prior year absent the event. This
  assumption is untestable but is more plausible when (a) the event is exogenous,
  (b) the control window matches the same calendar months, and (c) the treatment
  window is short. Findings should be interpreted as suggestive causal evidence
  rather than definitive causal claims.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, ttest_ind
from config import OUTPUTS_DIR
from pipeline.utils import banner, save_fig

log = logging.getLogger(__name__)

EVENTS = [
    {"label": "Boeing 737\nMAX Ban",      "date": "2019-03-10", "color": "#E74C3C"},
    {"label": "COVID-19\nPandemic",        "date": "2020-03-11", "color": "#8E44AD"},
    {"label": "Ukraine War\n& Airspace",   "date": "2022-02-24", "color": "#C0392B"},
    {"label": "Post-COVID\nBoom",          "date": "2022-06-01", "color": "#27AE60"},
    {"label": "EU Aviation\nStrikes",      "date": "2023-01-01", "color": "#E67E22"},
    {"label": "Boeing Door\nBlowout",      "date": "2024-01-05", "color": "#D35400"},
    {"label": "CrowdStrike\nOutage",       "date": "2024-07-19", "color": "#2980B9"},
]

TREATMENT_MONTHS = 4   # post-event window length
MIN_N            = 30  # minimum observations for a valid DiD estimate


def _did_estimate(df, event_date, treatment_months=TREATMENT_MONTHS):
    """
    Returns a dict with DiD estimate and supporting statistics, or None if
    insufficient data.

    Windows:
      post_treatment  : [event_date, event_date + treatment_months)
      pre_treatment   : same calendar months, 1 year earlier  (control)
      post_control    : same calendar months, 1 year earlier  (control for post)
      pre_control     : 2 years before event, same months     (pre-control)

    DiD = (post_treatment - pre_treatment) - (post_control - pre_control)
    """
    ed   = pd.Timestamp(event_date)
    # Post-event: event month through event_month + treatment_months
    post_start  = ed
    post_end    = ed + pd.DateOffset(months=treatment_months)
    # Pre-event same calendar months, 1 year earlier
    pre_start   = post_start - pd.DateOffset(years=1)
    pre_end     = post_end   - pd.DateOffset(years=1)

    df_dt = df.dropna(subset=["date_flown", "Overall_Rating"]).copy()
    df_dt = df_dt[(df_dt["date_flown"] >= pd.Timestamp("2013-01-01"))]

    post_treat = df_dt[(df_dt["date_flown"] >= post_start) &
                       (df_dt["date_flown"] <  post_end)]["Overall_Rating"]
    pre_treat  = df_dt[(df_dt["date_flown"] >= pre_start) &
                       (df_dt["date_flown"] <  pre_end)]["Overall_Rating"]

    if len(post_treat) < MIN_N or len(pre_treat) < MIN_N:
        return None

    # Pre-pre control: 2 years before event, same months
    pre_pre_start = post_start - pd.DateOffset(years=2)
    pre_pre_end   = post_end   - pd.DateOffset(years=2)
    post_ctrl = pre_treat   # year-before = post control
    pre_ctrl  = df_dt[(df_dt["date_flown"] >= pre_pre_start) &
                      (df_dt["date_flown"] <  pre_pre_end)]["Overall_Rating"]

    if len(pre_ctrl) < MIN_N:
        # Fallback: use simple pre/post without DiD correction
        did = post_treat.mean() - pre_treat.mean()
        ctrl_trend = 0.0
    else:
        ctrl_trend = post_ctrl.mean() - pre_ctrl.mean()
        did = (post_treat.mean() - pre_treat.mean()) - ctrl_trend

    # Significance: Mann-Whitney on post_treat vs pre_treat
    _, p = mannwhitneyu(post_treat, pre_treat, alternative="two-sided")

    return {
        "pre_mean":    pre_treat.mean(),
        "post_mean":   post_treat.mean(),
        "pre_n":       len(pre_treat),
        "post_n":      len(post_treat),
        "raw_change":  post_treat.mean() - pre_treat.mean(),
        "ctrl_trend":  ctrl_trend,
        "did":         did,
        "p_value":     p,
        "significant": p < 0.05,
    }


def run(df):
    banner("MODULE 16 — CAUSAL INFERENCE (DiD EVENT ANALYSIS)")

    df_c = df.copy()
    df_c["date_flown"] = pd.to_datetime(df_c["date_flown"], errors="coerce")

    # Compute monthly rolling mean for the backdrop chart
    monthly = (df_c.dropna(subset=["date_flown", "Overall_Rating"])
               .set_index("date_flown")
               .resample("MS")["Overall_Rating"]
               .agg(["mean", "count"])
               .rename(columns={"mean": "rating", "count": "n"}))
    monthly = monthly[monthly["n"] >= 50]   # require at least 50 reviews/month

    results = []
    for ev in EVENTS:
        res = _did_estimate(df_c, ev["date"])
        if res is None:
            log.info(f"  {ev['label'].replace(chr(10),' '):<30} SKIPPED (insufficient data)")
            continue
        res.update({"label": ev["label"].replace("\n", " "), "color": ev["color"],
                    "event_date": ev["date"]})
        results.append(res)
        sig = "***" if res["p_value"] < 0.001 else ("**" if res["p_value"] < 0.01
              else ("*" if res["p_value"] < 0.05 else "n.s."))
        log.info(
            f"  {res['label']:<30}  "
            f"DiD={res['did']:+.3f}  raw={res['raw_change']:+.3f}  "
            f"p={res['p_value']:.4f} {sig}  "
            f"(pre n={res['pre_n']:,}, post n={res['post_n']:,})")

    if not results:
        log.warning("  No events had sufficient data for DiD estimation.")
        return df_c

    rdf = pd.DataFrame(results)

    # ── FIGURE 1: Time series with DiD annotation ─────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(15, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("MODULE 16 — Causal Inference: Event Impact on Passenger Satisfaction\n"
                 "(Difference-in-Differences Design)",
                 fontsize=13, fontweight="bold")

    # Top: rolling mean time series
    ax = axes[0]
    ax.plot(monthly.index, monthly["rating"], lw=2, color="#2C3E50",
            label="Monthly mean rating", zorder=3)
    roll = monthly["rating"].rolling(3, center=True).mean()
    ax.plot(monthly.index, roll, lw=3, color="#4C72B0",
            label="3-month rolling mean", zorder=4)

    for ev in EVENTS:
        ed = pd.Timestamp(ev["date"])
        ax.axvline(ed, color=ev["color"], lw=1.5, ls="--", alpha=0.7, zorder=2)
        if ed in monthly.index or (monthly.index.min() < ed < monthly.index.max()):
            ax.text(ed, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 9,
                    ev["label"], rotation=90, fontsize=7,
                    va="top", ha="right", color=ev["color"])

    ax.set_ylabel("Mean Overall Rating (1–10)")
    ax.set_ylim(1, 10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    # Bottom: DiD estimates as bar chart
    ax2 = axes[1]
    labels_short = [r["label"].split(" ")[0] + "\n" +
                    " ".join(r["label"].split(" ")[1:])
                    for r in results]
    colors   = [r["color"] for r in results]
    did_vals = [r["did"] for r in results]
    bars     = ax2.bar(range(len(results)), did_vals, color=colors, edgecolor="k", alpha=0.85)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels([r["label"] for r in results], fontsize=8)
    ax2.set_ylabel("DiD Estimate\n(rating points)")
    ax2.set_title("DiD Effect Size per Event (net of seasonal baseline)", fontsize=9)
    for b, r in zip(bars, results):
        sig = "*" if r["significant"] else ""
        ax2.text(b.get_x() + b.get_width()/2,
                 b.get_height() + (0.02 if b.get_height() >= 0 else -0.08),
                 f"{r['did']:+.2f}{sig}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(f"{OUTPUTS_DIR}/causal_event_impacts.png")

    # ── FIGURE 2: Pre/post comparison per event ───────────────────────────
    n_events = len(results)
    ncols    = min(4, n_events)
    nrows    = (n_events + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols,
                               figsize=(4.5 * ncols, 4 * nrows),
                               squeeze=False)
    fig2.suptitle("Pre vs Post Ratings: Causal Event Windows",
                  fontsize=13, fontweight="bold")

    for i, r in enumerate(results):
        ax_i = axes2[i // ncols][i % ncols]
        bars_i = ax_i.bar(["Pre-event\n(same season\nprior year)",
                            "Post-event\n(treatment\nwindow)"],
                           [r["pre_mean"], r["post_mean"]],
                           color=["#4C72B0", r["color"]],
                           edgecolor="k", alpha=0.87)
        ax_i.set_ylim(0, 8)
        ax_i.set_title(r["label"], fontweight="bold", fontsize=9)
        ax_i.set_ylabel("Mean Rating")
        for b, v in zip(bars_i, [r["pre_mean"], r["post_mean"]]):
            ax_i.text(b.get_x() + b.get_width()/2, v + 0.1,
                      f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01
              else ("*" if r["p_value"] < 0.05 else "n.s."))
        ax_i.text(0.5, 7.3,
                  f"DiD={r['did']:+.2f}  {sig}",
                  ha="center", fontsize=9, color=r["color"], fontweight="bold",
                  transform=ax_i.get_xaxis_transform())

    # Hide unused subplots
    for j in range(len(results), nrows * ncols):
        axes2[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(f"{OUTPUTS_DIR}/causal_did_panels.png")

    # ── Summary table ─────────────────────────────────────────────────────
    log.info("\n  DiD SUMMARY TABLE")
    log.info(f"  {'Event':<30} {'DiD':>8} {'Raw Δ':>8} {'p-value':>10} {'Sig':>5}")
    log.info("  " + "─" * 65)
    for r in sorted(results, key=lambda x: x["did"]):
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01
              else ("*" if r["p_value"] < 0.05 else "n.s."))
        log.info(f"  {r['label']:<30} {r['did']:>+8.3f} {r['raw_change']:>+8.3f} "
                 f"{r['p_value']:>10.4f} {sig:>5}")

    # Save CSV
    rdf[["label","event_date","pre_mean","post_mean","raw_change",
         "ctrl_trend","did","p_value","significant"]].to_csv(
        f"{OUTPUTS_DIR}/causal_did_results.csv", index=False)
    log.info(f"  Saved: {OUTPUTS_DIR}/causal_did_results.csv")

    return df_c
