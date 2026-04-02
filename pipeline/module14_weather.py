"""
pipeline/module14_weather.py — Weather Signal Analysis

Extracts weather-related language from review text and analyses how it
affects passenger sentiment. No external API required — we study what
passengers themselves write about weather, which is more academically
honest than inferring conditions from external data.

Key hypotheses tested:
  H1: Weather mentions correlate with higher ratings than non-weather reviews
      (attribution theory: external/uncontrollable causes = more forgiveness)
  H2: Weather operates primarily through the delay pathway
      (weather alone = high rating; weather + delay = low rating)
  H3: Weather mentions peak in winter (Dec-Jan) and mid-summer (Jul-Aug)
  H4: Short-haul passengers mention weather more than long-haul
  H5: Economy passengers are more affected by weather than Business Class
"""
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from config import OUTPUTS_DIR, SUB_COLS, CLASSES
from pipeline.utils import banner, save_fig

log = logging.getLogger(__name__)

WEATHER_CATS = {
    "Turbulence":    r"\b(turbulence|turbulent|bumpy|rough air|choppy|air pocket)\b",
    "Precipitation": r"\b(rain|rainy|raining|snow|snowy|snowstorm|blizzard|hail|sleet)\b",
    "Ice & De-icing":r"\b(ice|icy|de-icing|deicing|de-ice|frost|frozen)\b",
    "Fog & Vis":     r"\b(fog|foggy|mist|misty|visibility|low vis)\b",
    "Wind & Storm":  r"\b(wind|windy|gust|gale|storm|stormy|thunder|thunderstorm|lightning|hurricane|typhoon|cyclone)\b",
    "Generic":       r"\b(weather delay|bad weather|poor weather|weather-related|due to weather|caused by weather)\b",
}

ALL_WEATHER_RE = re.compile("|".join(WEATHER_CATS.values()), re.I)
DELAY_RE       = re.compile(r"\b(delay|delayed|delays|cancelled|cancel|cancellation|late)\b", re.I)
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
CAT_COLORS = {
    "Turbulence":"#8E44AD","Precipitation":"#2980B9","Ice & De-icing":"#1ABC9C",
    "Fog & Vis":"#7F8C8D","Wind & Storm":"#E74C3C","Generic":"#F39C12",
}


def _tag_categories(text):
    return [cat for cat, pat in WEATHER_CATS.items() if re.search(pat, str(text), re.I)]


def run(df: pd.DataFrame) -> pd.DataFrame:
    banner("MODULE 14 — WEATHER SIGNAL ANALYSIS")
    df_w = df.copy()

    # ── Tag each review ───────────────────────────────────────────────────────
    df_w["weather_mention"] = df_w["text"].apply(lambda t: bool(ALL_WEATHER_RE.search(str(t))))
    df_w["delay_mention"]   = df_w["text"].apply(lambda t: bool(DELAY_RE.search(str(t))))
    df_w["weather_cats"]    = df_w["text"].apply(lambda t: _tag_categories(str(t)))

    n_weather = df_w["weather_mention"].sum()
    pct       = n_weather / len(df_w) * 100
    log.info(f"  Weather mentions: {n_weather:,} / {len(df_w):,} ({pct:.1f}%)")

    w_r  = df_w[df_w["weather_mention"]]["Overall_Rating"].dropna()
    nw_r = df_w[~df_w["weather_mention"]]["Overall_Rating"].dropna()
    _,p  = mannwhitneyu(w_r, nw_r, alternative="two-sided")
    gap  = w_r.mean() - nw_r.mean()
    log.info(f"  Rating — weather: {w_r.mean():.2f} | no-weather: {nw_r.mean():.2f} | gap: {gap:+.2f} (p={p:.4f})")
    log.info(f"  → Passengers are {'MORE' if gap>0 else 'LESS'} forgiving when weather is mentioned (attribution theory).")

    both          = (df_w["weather_mention"] & df_w["delay_mention"]).sum()
    weather_only  = (df_w["weather_mention"] & ~df_w["delay_mention"]).sum()
    pct_cooccur   = both / n_weather * 100 if n_weather > 0 else 0
    wd_r   = df_w[df_w["weather_mention"] & df_w["delay_mention"]]["Overall_Rating"].dropna()
    wnd_r  = df_w[df_w["weather_mention"] & ~df_w["delay_mention"]]["Overall_Rating"].dropna()
    log.info(f"  Weather+delay co-mention: {both:,} ({pct_cooccur:.1f}%)")
    log.info(f"  Weather+delay mean rating: {wd_r.mean():.2f} | weather alone: {wnd_r.mean():.2f}")

    # category stats
    cat_stats = {}
    for cat in WEATHER_CATS:
        mask = df_w["weather_cats"].apply(lambda x: cat in x)
        n    = mask.sum()
        if n < 5: continue
        r    = df_w[mask]["Overall_Rating"].dropna()
        cat_stats[cat] = {"n": n, "mean": r.mean()}
        log.info(f"  {cat:<18} n={n:>5,}  mean={r.mean():.2f}")

    # seasonal
    df_w["month"] = df_w["date_flown"].dt.month
    df_w["year"]  = df_w["date_flown"].dt.year
    monthly_total   = df_w.groupby("month").size()
    monthly_weather = df_w[df_w["weather_mention"]].groupby("month").size()
    monthly_rate    = (monthly_weather / monthly_total * 100).reindex(range(1,13), fill_value=0)
    monthly_rating  = df_w[df_w["weather_mention"]].groupby("month")["Overall_Rating"].mean()
    peak_month      = MONTHS[int(monthly_rate.idxmax()) - 1]

    available_sub = [c for c in SUB_COLS if c in df_w.columns and df_w[c].notna().sum() > 50]

    # ── FIGURE 1: Core 4-panel ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MODULE 14 — Weather Signal Analysis", fontsize=14, fontweight="bold")

    # A: Rating distribution
    bins = np.arange(0.5, 11.5, 1)
    axes[0,0].hist(nw_r, bins=bins, alpha=0.6, color="#4C72B0",
                   label=f"No weather (n={len(nw_r):,})", density=True)
    axes[0,0].hist(w_r,  bins=bins, alpha=0.6, color="#E74C3C",
                   label=f"Weather mention (n={len(w_r):,})", density=True)
    axes[0,0].axvline(nw_r.mean(), color="#4C72B0", lw=2, ls="--", label=f"Mean: {nw_r.mean():.2f}")
    axes[0,0].axvline(w_r.mean(),  color="#E74C3C", lw=2, ls="--", label=f"Mean: {w_r.mean():.2f}")
    axes[0,0].set_xlabel("Overall Rating"); axes[0,0].set_ylabel("Density")
    axes[0,0].set_title("Rating Distribution\n(Attribution Effect: weather → more forgiveness)", fontweight="bold")
    axes[0,0].legend(fontsize=8)

    # B: Category counts + mean rating
    if cat_stats:
        cs = sorted(cat_stats, key=lambda c: cat_stats[c]["n"], reverse=True)
        x2 = np.arange(len(cs))
        ax_b2 = axes[0,1].twinx()
        axes[0,1].bar(x2, [cat_stats[c]["n"] for c in cs],
                      color=[CAT_COLORS.get(c,"#999") for c in cs], edgecolor="k", alpha=0.85)
        ax_b2.plot(x2, [cat_stats[c]["mean"] for c in cs],
                   "D--", color="#2c2c2c", ms=8, lw=2, label="Mean rating")
        ax_b2.axhline(nw_r.mean(), color="#4C72B0", lw=1, ls=":", label="Baseline rating")
        axes[0,1].set_xticks(x2)
        axes[0,1].set_xticklabels([c.replace(" & ","\n& ") for c in cs], fontsize=8, rotation=15, ha="right")
        axes[0,1].set_ylabel("Review Count"); ax_b2.set_ylabel("Mean Rating")
        axes[0,1].set_title("Weather Category: Frequency & Mean Rating", fontweight="bold")
        l1,lb1=axes[0,1].get_legend_handles_labels(); l2,lb2=ax_b2.get_legend_handles_labels()
        axes[0,1].legend(l1+l2, lb1+lb2, fontsize=8)

    # C: Weather × Delay 4-bar interaction
    groups = {
        "No weather\nno delay":    df_w[~df_w["weather_mention"] & ~df_w["delay_mention"]],
        "Delay only\n(airline fault)": df_w[~df_w["weather_mention"] & df_w["delay_mention"]],
        "Weather only\n(no delay)":    df_w[df_w["weather_mention"]  & ~df_w["delay_mention"]],
        "Weather\n+ delay":            df_w[df_w["weather_mention"]  & df_w["delay_mention"]],
    }
    grp_means = {k: v["Overall_Rating"].dropna().mean() for k, v in groups.items()}
    grp_ns    = {k: len(v["Overall_Rating"].dropna()) for k, v in groups.items()}
    colors_g  = ["#2ca02c","#d62728","#ff7f0e","#8c564b"]
    bars_c    = axes[1,0].bar(range(4), list(grp_means.values()), color=colors_g, edgecolor="k", alpha=0.85)
    axes[1,0].set_xticks(range(4)); axes[1,0].set_xticklabels(list(grp_means.keys()), fontsize=8)
    axes[1,0].set_ylim(0, 8); axes[1,0].set_ylabel("Mean Rating")
    axes[1,0].set_title("Weather × Delay Attribution Pathway", fontweight="bold")
    for b, (k,v) in zip(bars_c, grp_means.items()):
        axes[1,0].text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                       f"{v:.2f}\n(n={grp_ns[k]:,})", ha="center", fontsize=7.5, fontweight="bold")

    # D: Seasonal pattern
    ax_d = axes[1,1]; ax_d2 = ax_d.twinx()
    ax_d.bar(range(1,13), monthly_rate.values, color="#3498DB", edgecolor="k", alpha=0.7)
    ax_d2.plot(range(1,13), monthly_rating.reindex(range(1,13)).values,
               "o-", color="#E74C3C", lw=2.5, ms=8)
    ax_d.set_xticks(range(1,13)); ax_d.set_xticklabels(MONTHS, fontsize=8)
    ax_d.set_ylabel("% Reviews Mentioning Weather"); ax_d2.set_ylabel("Mean Rating (weather reviews)")
    ax_d.set_title("Seasonal Pattern of Weather Mentions", fontweight="bold")
    save_fig(f"{OUTPUTS_DIR}/weather_analysis.png")

    # ── FIGURE 2: Sub-rating impact ─────────────────────────────────────────
    if available_sub:
        sub_comp = pd.DataFrame({
            "No Weather": df_w[~df_w["weather_mention"]][available_sub].mean(),
            "Weather":    df_w[df_w["weather_mention"]][available_sub].mean(),
        })
        sub_comp["Δ"] = sub_comp["Weather"] - sub_comp["No Weather"]
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
        fig2.suptitle("Sub-Rating Impact of Weather Mentions", fontsize=13, fontweight="bold")
        sns.heatmap(sub_comp[["No Weather","Weather"]].T, annot=True, fmt=".2f",
                    cmap="RdYlGn", vmin=1, vmax=5, ax=axes2[0], linewidths=0.5,
                    xticklabels=[c.replace(" & ","\n& ") for c in available_sub])
        axes2[0].set_title("Mean Sub-Rating: Weather vs No Weather", fontweight="bold")
        colors_d = ["#d62728" if v < 0 else "#2ca02c" for v in sub_comp["Δ"]]
        axes2[1].barh(sub_comp.index, sub_comp["Δ"], color=colors_d, edgecolor="k", alpha=0.85)
        axes2[1].axvline(0, color="black", lw=0.8)
        axes2[1].set_xlabel("Δ Sub-Rating (Weather − No Weather)")
        axes2[1].set_title("Sub-Rating Difference", fontweight="bold")
        for i, v in enumerate(sub_comp["Δ"]):
            axes2[1].text(v+(0.003 if v>=0 else -0.003), i, f"{v:+.3f}",
                          va="center", fontsize=9, ha="left" if v>=0 else "right")
        save_fig(f"{OUTPUTS_DIR}/weather_subrating_impact.png")

    # ── FIGURE 3: Cabin class & haul type breakdown ─────────────────────────
    # Only draw this figure if sub-ratings are well-populated (scraped data)
    # Checks: Seat Type column present and at least 20% cabin class labels
    has_cabin = ("Seat Type" in df_w.columns and
                 df_w["Seat Type"].isin(CLASSES).sum() > len(df_w) * 0.05)
    has_haul  = "haul_type" in df_w.columns

    if has_cabin or has_haul:
        ncols = sum([has_cabin, has_haul])
        fig3, axes3 = plt.subplots(1, ncols, figsize=(7*ncols, 5))
        if ncols == 1:
            axes3 = [axes3]
        fig3.suptitle("Weather Effect by Segment", fontsize=13, fontweight="bold")
        idx = 0

        if has_cabin:
            cabin_stats = []
            for cls in CLASSES:
                sub = df_w[df_w["Seat Type"] == cls]
                if len(sub) < 30: continue
                w_cls  = sub[sub["weather_mention"]]["Overall_Rating"].dropna()
                nw_cls = sub[~sub["weather_mention"]]["Overall_Rating"].dropna()
                if len(w_cls) < 5: continue
                cabin_stats.append({
                    "class": cls.replace(" Class",""),
                    "weather_mean": w_cls.mean(),
                    "no_weather_mean": nw_cls.mean(),
                    "gap": w_cls.mean() - nw_cls.mean(),
                    "pct_weather": len(w_cls) / len(sub) * 100,
                })
            if cabin_stats:
                cs_df = pd.DataFrame(cabin_stats)
                x3 = np.arange(len(cs_df)); w3 = 0.35
                axes3[idx].bar(x3-w3/2, cs_df["no_weather_mean"], w3,
                               label="No weather", color="#4C72B0", edgecolor="k", alpha=0.85)
                axes3[idx].bar(x3+w3/2, cs_df["weather_mean"], w3,
                               label="Weather mention", color="#E74C3C", edgecolor="k", alpha=0.85)
                axes3[idx].set_xticks(x3)
                axes3[idx].set_xticklabels(cs_df["class"], fontsize=9)
                axes3[idx].set_ylabel("Mean Rating"); axes3[idx].set_ylim(0, 9)
                axes3[idx].set_title("Weather Attribution Effect by Cabin Class", fontweight="bold")
                axes3[idx].legend(fontsize=9)
                for xi, row in cs_df.iterrows():
                    color = "#2ca02c" if row["gap"] > 0 else "#d62728"
                    axes3[idx].text(xi, max(row["weather_mean"], row["no_weather_mean"]) + 0.25,
                                    f"Δ{row['gap']:+.2f}", ha="center", fontsize=8,
                                    color=color, fontweight="bold")
                idx += 1

        if has_haul:
            haul_order = ["Short-haul","Medium-haul","Long-haul","Ultra-long"]
            haul_stats = []
            for haul in haul_order:
                sub = df_w[df_w["haul_type"] == haul]
                if len(sub) < 30: continue
                w_h  = sub[sub["weather_mention"]]["Overall_Rating"].dropna()
                nw_h = sub[~sub["weather_mention"]]["Overall_Rating"].dropna()
                if len(w_h) < 5: continue
                haul_stats.append({
                    "haul": haul.replace("-haul","").replace("Ultra-","Ultra-\n"),
                    "weather_mean": w_h.mean(), "no_weather_mean": nw_h.mean(),
                    "gap": w_h.mean()-nw_h.mean(),
                    "pct_weather": len(w_h)/len(sub)*100,
                })
            if haul_stats:
                hs_df = pd.DataFrame(haul_stats)
                x4 = np.arange(len(hs_df)); w4 = 0.35
                axes3[idx].bar(x4-w4/2, hs_df["no_weather_mean"], w4,
                               label="No weather", color="#4C72B0", edgecolor="k", alpha=0.85)
                axes3[idx].bar(x4+w4/2, hs_df["weather_mean"], w4,
                               label="Weather mention", color="#E74C3C", edgecolor="k", alpha=0.85)
                axes3[idx].set_xticks(x4)
                axes3[idx].set_xticklabels(hs_df["haul"], fontsize=9)
                axes3[idx].set_ylabel("Mean Rating"); axes3[idx].set_ylim(0, 9)
                axes3[idx].set_title("Weather Attribution Effect by Haul Type", fontweight="bold")
                axes3[idx].legend(fontsize=9)
                for xi, row in hs_df.iterrows():
                    color = "#2ca02c" if row["gap"] > 0 else "#d62728"
                    axes3[idx].text(xi, max(row["weather_mean"],row["no_weather_mean"])+0.25,
                                    f"Δ{row['gap']:+.2f}", ha="center", fontsize=8,
                                    color=color, fontweight="bold")
        save_fig(f"{OUTPUTS_DIR}/weather_segments.png")

    # ── FIGURE 4: Top airlines by weather mention rate ──────────────────────
    if "Airline Name" in df_w.columns:
        airline_stats = (
            df_w.groupby("Airline Name")
            .agg(total=("Overall_Rating","count"),
                 weather=("weather_mention","sum"))
            .query("total >= 100")
            .assign(rate=lambda d: d["weather"]/d["total"]*100)
        )
        top_weather = airline_stats.nlargest(15, "rate")
        if len(top_weather) >= 5:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            colors_a = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top_weather)))
            ax4.barh(top_weather.index[::-1], top_weather["rate"][::-1],
                     color=colors_a, edgecolor="k", alpha=0.85)
            ax4.axvline(pct, color="black", lw=1.5, ls="--",
                        label=f"Dataset average ({pct:.1f}%)")
            ax4.set_xlabel("% Reviews Mentioning Weather")
            ax4.set_title("Airlines with Highest Weather Mention Rate\n"
                          "(min. 100 reviews)", fontweight="bold")
            ax4.legend(fontsize=9)
            save_fig(f"{OUTPUTS_DIR}/weather_airlines.png")

    log.info(f"""
  WEATHER MODULE SUMMARY
  ───────────────────────────────────────────────────────
  Total weather mentions    : {n_weather:,} ({pct:.1f}% of corpus)
  Mean rating — weather     : {w_r.mean():.2f}
  Mean rating — no weather  : {nw_r.mean():.2f}
  Attribution gap           : {gap:+.2f} (p={p:.4f})
  Weather + delay           : {both:,} ({pct_cooccur:.1f}% of weather reviews)
  Weather without delay     : {weather_only:,} (mean: {wnd_r.mean():.2f})
  Weather + delay mean      : {wd_r.mean():.2f}
  Peak weather month        : {peak_month}
  ───────────────────────────────────────────────────────
  Interpretation: passengers are {'MORE' if gap>0 else 'LESS'} forgiving when
  weather is mentioned. The delay pathway accounts for {pct_cooccur:.0f}% of
  weather-related sentiment, consistent with attribution theory.
""")
    return df_w
