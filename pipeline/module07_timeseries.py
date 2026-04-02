"""
pipeline/module07_timeseries.py — Time series, global events, seasonality.

Events sourced from: IATA, FlightGlobal, EASA, FAA, aviation operational bulletins.
Timeline covers 2019 through March 2026.
Year range is fully dynamic — uses whatever date range exists in the data.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUTS_DIR
from pipeline.utils import banner, save_fig

log = logging.getLogger(__name__)

# ── Global aviation events timeline ──────────────────────────────────────────
# (date, label, colour, sign)
# sign +1 = positive / recovery event
# sign -1 = negative / disruptive event
# All events have documented direct operational impact on airlines.
EVENTS = [

    # ── 2019 ──────────────────────────────────────────────────────────────────
    ("2019-03-10", "Boeing 737\nMAX Ban",              "#E76F51",  1),

    # ── 2020–2021: COVID ──────────────────────────────────────────────────────
    ("2020-03-11", "COVID-19\nPandemic",               "#6D2B7F", -1),
    ("2021-06-01", "EU Green\nPass",                   "#2A9D8F",  1),

    # ── 2022 ──────────────────────────────────────────────────────────────────
    ("2022-02-24", "Ukraine War\n& Airspace Closure",  "#E63946", -1),
    ("2022-06-01", "Post-COVID\nBoom",                 "#F4A261",  1),

    # ── 2023 ──────────────────────────────────────────────────────────────────
    ("2023-01-01", "EU Strikes &\nCancellations",      "#E76F51", -1),
    ("2023-10-07", "Israel-Gaza\nWar Begins",          "#C0392B", -1),
    # Airlines suspend Tel Aviv, Beirut routes; Middle East airspace disruption

    # ── 2024 ──────────────────────────────────────────────────────────────────
    ("2024-01-05", "Boeing 737\nDoor Blowout",         "#922B21", -1),
    # Alaska Airlines mid-air panel loss; FAA production cap on Boeing

    ("2024-04-13", "Iran Attacks\nIsrael (1st)",       "#E74C3C", -1),
    # 300+ drones & missiles; Jordan, Iraq, Gulf airspace closed for hours

    ("2024-07-19", "CrowdStrike\nIT Outage",           "#8E44AD", -1),
    # Global IT failure grounds ~8,500 flights in a single day

    ("2024-10-01", "Iran Attacks\nIsrael (2nd)",       "#C0392B", -1),
    # 2nd ballistic missile barrage; repeated airspace closures

    # ── 2025 ──────────────────────────────────────────────────────────────────
    ("2025-01-29", "DCA Mid-Air\nCollision",           "#7B241C", -1),
    # Washington Reagan Airport; renewed ATC staffing scrutiny

    ("2025-10-01", "GNSS Spoofing\nCrisis",            "#884EA0", -1),
    # EASA/IATA report skyrocketing GPS spoofing events (+220%);
    # Middle East & Eastern Europe electronic warfare forces rerouting

    # ── 2026 ──────────────────────────────────────────────────────────────────
    ("2026-02-28", "Hormuz Closure\n& Fuel Surge",     "#E74C3C", -1),
    # Strait of Hormuz effectively suspended; global jet fuel prices spike immediately
    # Source: IATA operational bulletin, Feb 2026

    ("2026-03-10", "Iran/Iraq/Kuwait\nAirspace Closed", "#C0392B", -1),
    # Official NOTAMs close airspace over Iran, Iraq, Kuwait, Syria;
    # Europe–Asia corridors severely disrupted
    # Source: FlightGlobal / NOTAM records, March 2026

    ("2026-03-19", "FAA Radar\nSeparation Mandate",    "#1A5276", -1),
    # FAA suspends visual separation for helicopters & fixed-wing in high-traffic airspace
    # Source: FAA policy directive, March 19 2026

    ("2026-03-25", "Jordan Transit\nHub Targeted",      "#922B21", -1),
    # Strikes hit Muwaffaq Salti Air Base, Azraq; Amman secondary hub compromised;
    # risk profile of Jordanian corridor critically elevated
    # Source: Aviation operational bulletins, March 25-26 2026

    ("2026-03-29", "Gulf Capacity\nCollapse",           "#C0392B", -1),
    # Riyadh & Jeddah hubs in chaos; UAE/Saudi/Oman/Qatar corridors restricted;
    # record 171M spring passengers vs sudden capacity cuts = Perfect Storm
    # Source: IATA, Accenture industry reports, March 2026

]


def run(df):
    banner("MODULE 7 — TIME SERIES & GLOBAL EVENTS")

    df_ts = df[df["date_flown"].notna()].copy()
    df_ts["ym"] = df_ts["date_flown"].dt.to_period("M")

    monthly = df_ts.groupby("ym").agg(
        n_reviews   = ("Overall_Rating", "count"),
        mean_rating = ("Overall_Rating", "mean"),
        pct_neg     = ("sentiment", lambda x: (x == "negative").mean() * 100),
    ).reset_index()
    monthly["date"] = monthly["ym"].dt.to_timestamp()
    monthly = monthly[monthly["n_reviews"] >= 5]

    data_start = monthly["date"].min()
    data_end   = monthly["date"].max()

    # ── Three-panel time series ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(20, 13), sharex=True)
    fig.suptitle("Airline Review Sentiment Over Time — Global Context",
                 fontsize=13, fontweight="bold")

    axes[0].plot(monthly["date"], monthly["mean_rating"],
                 color="#4C72B0", alpha=0.4, lw=1.2)
    roll = monthly.set_index("date")["mean_rating"].rolling(3, center=True).mean()
    axes[0].plot(roll.index, roll.values, color="#4C72B0", lw=2.5,
                 label="3-month rolling average")
    axes[0].set_ylim(1, 10)
    axes[0].set_ylabel("Mean Rating")
    axes[0].legend(fontsize=9)

    axes[1].fill_between(monthly["date"], monthly["pct_neg"],
                         alpha=0.35, color="#d62728")
    axes[1].plot(monthly["date"], monthly["pct_neg"], color="#d62728", lw=1.5)
    axes[1].set_ylabel("% Negative Reviews")
    axes[1].set_ylim(0, 100)

    axes[2].bar(monthly["date"], monthly["n_reviews"],
                width=25, color="#55A868", alpha=0.7)
    axes[2].set_ylabel("Review Count")
    axes[2].set_xlabel("Date of Flight")

    # Annotate events within the data window
    # Use more height levels to handle clustered 2026 events without overlap
    pos_heights = [8.5, 7.8, 8.2, 7.5]
    neg_heights = [2.5, 1.6, 3.1, 2.0, 1.2, 3.5]
    pos_idx = neg_idx = 0
    annotated = 0

    for ev_date, label, color, sign in EVENTS:
        ev_dt = pd.Timestamp(ev_date)
        if ev_dt < data_start or ev_dt > data_end:
            continue
        for ax in axes:
            ax.axvline(ev_dt, color=color, lw=1.5, ls="--", alpha=0.8)
        if sign > 0:
            y_pos = pos_heights[pos_idx % len(pos_heights)]; pos_idx += 1
        else:
            y_pos = neg_heights[neg_idx % len(neg_heights)]; neg_idx += 1
        axes[0].annotate(
            label, xy=(ev_dt, y_pos), fontsize=6.0, color=color,
            fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.80),
        )
        annotated += 1

    # Dynamic seasonal highlights for every year in the data
    min_yr = int(monthly["date"].dt.year.min())
    max_yr = int(monthly["date"].dt.year.max()) + 1
    for yr in range(min_yr, max_yr):
        for ax in axes:
            ax.axvspan(pd.Timestamp(f"{yr}-07-01"), pd.Timestamp(f"{yr}-08-28"),
                       alpha=0.10, color="#FFE8A3", zorder=0)
            ax.axvspan(pd.Timestamp(f"{yr}-12-01"), pd.Timestamp(f"{yr}-12-28"),
                       alpha=0.10, color="#D4E6F1", zorder=0)

    save_fig(f"{OUTPUTS_DIR}/timeseries_global_events.png")

    # ── Seasonality heatmap — dynamic year range ──────────────────────────────
    df_ts["year"]  = df_ts["date_flown"].dt.year
    df_ts["month"] = df_ts["date_flown"].dt.month
    pivot = df_ts.groupby(["year", "month"])["Overall_Rating"].mean().unstack()
    pivot = pivot[pivot.notna().sum(axis=1) >= 6]

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_h = max(5, len(pivot) * 0.75)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=2, vmax=8, ax=ax, linewidths=0.5,
                xticklabels=month_names)
    ax.set_title(
        f"Seasonality Heatmap — Mean Rating by Month × Year "
        f"({int(pivot.index.min())}–{int(pivot.index.max())})",
        fontweight="bold",
    )
    save_fig(f"{OUTPUTS_DIR}/seasonality_heatmap.png")

    log.info(
        f"Time series complete | "
        f"Data: {min_yr}–{max_yr - 1} | "
        f"Events annotated: {annotated}/{len(EVENTS)}"
    )
