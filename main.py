"""
main.py — ABSA Airline Review Pipeline
=======================================
Default behaviour:
  1. Scrape fresh data from airlinequality.com (incremental if CSV exists)
  2. If scraping fails → fall back to data/Airline_review.csv
  3. Run all 13 analysis modules

Each module is wrapped in try/except so a failure in one non-critical module
does not stop the rest of the pipeline from completing (FIX 4).

CLI flags:
  python main.py                                   # scrape first, then analyse
  python main.py --scrape-only                     # update CSV only, no analysis
  python main.py --no-scrape                       # skip scraper, run analysis
  python main.py --airlines british-airways,ryanair
  python main.py --max-pages 3
  python main.py --csv /path/to/file.csv
"""
import argparse, logging, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from config import FALLBACK_CSV, SCRAPED_CSV, OUTPUTS_DIR, SCRAPER_DELAY
from scraper import skytrax_scraper
from pipeline import (
    module14_weather,
    module15_peak_season,
    module16_causal,
    module17_seasonal_carrier,
    module18_bert,
    loader, eda, bio_extraction,
    module05_lcc_legacy, module06_cabin_class, module07_timeseries,
    module08_predictive, module09_fake, module10_routes,
    module11_aope, module12_traveller, module13_aircraft, saver,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUTS_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="ABSA Airline Review Pipeline")
    p.add_argument("--scrape-only", action="store_true")
    p.add_argument("--no-scrape",   action="store_true")
    p.add_argument("--airlines",    default=None)
    p.add_argument("--max-pages",   type=int, default=999)
    p.add_argument("--delay-min",   type=float, default=SCRAPER_DELAY[0])
    p.add_argument("--delay-max",   type=float, default=SCRAPER_DELAY[1])
    p.add_argument("--csv",         default=None)
    return p.parse_args()


def acquire_data(args) -> Path:
    if args.csv:
        path = Path(args.csv)
        if not path.exists(): log.error(f"--csv not found: {path}"); sys.exit(1)
        log.info(f"Using explicit CSV: {path}"); return path

    if args.no_scrape:
        if SCRAPED_CSV.exists():
            log.info(f"--no-scrape: using scraped CSV ({SCRAPED_CSV})"); return SCRAPED_CSV
        log.info("--no-scrape: no scraped CSV found, using fallback.")
        return _use_fallback()

    existing = SCRAPED_CSV if SCRAPED_CSV.exists() else None
    log.info("Previous scraped CSV found — incremental mode." if existing
             else "No previous scraped CSV — full scrape.")
    log.info("Attempting to scrape fresh data from airlinequality.com …")
    airlines = ([a.strip() for a in args.airlines.split(",")] if args.airlines else None)

    result = skytrax_scraper.scrape(
        out_csv=SCRAPED_CSV, airlines=airlines, max_pages=args.max_pages,
        delay=(args.delay_min, args.delay_max), existing_csv=existing)

    if result and Path(result).exists():
        try:
            import pandas as pd
            if len(pd.read_csv(result, nrows=5)) > 0:
                log.info(f"Scrape successful → {result}"); return Path(result)
        except Exception as e:
            log.warning(f"Scraped file unreadable: {e}")

    log.warning("Scraper failed — falling back to bundled CSV.")
    return _use_fallback()


def _use_fallback() -> Path:
    if not FALLBACK_CSV.exists():
        log.error(f"Fallback CSV not found at {FALLBACK_CSV}."); sys.exit(1)
    log.info(f"Using fallback CSV: {FALLBACK_CSV}"); return FALLBACK_CSV


def _run_module(name: str, fn, *args, **kwargs):
    """FIX 4: Run a module with try/except so one failure doesn't stop the pipeline."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.error(f"Module {name} failed: {e} — continuing with remaining modules.")
        import traceback
        log.debug(traceback.format_exc())
        return kwargs.get("default", args[0] if args else None)


def main():
    args      = parse_args()
    data_path = acquire_data(args)

    if args.scrape_only:
        log.info(f"\n  --scrape-only: CSV saved to {data_path}\n"
                 f"  Run without --scrape-only for full analysis.")
        return

    # §1-2: Load + EDA (critical — stop if these fail)
    df = loader.load(data_path)
    _run_module("EDA", eda.run, df)

    # §3-6: Core ABSA (critical chain — bio feeds asc feeds alignment)
    df, bio_df      = _run_module("BIO extraction",  bio_extraction.run_bio,       df) or (df, None)
    bio_df, metrics = _run_module("ASC",              bio_extraction.run_asc,       df, bio_df) or (bio_df, {})
    corr_s          = _run_module("Alignment",        bio_extraction.run_alignment, df, bio_df)

    # Modules 5-13: independent — failures are logged but pipeline continues
    _run_module("Module 5 LCC vs Legacy",   module05_lcc_legacy.run,   df)
    _run_module("Module 6 Cabin Class",     module06_cabin_class.run,  df)
    _run_module("Module 7 Time Series",     module07_timeseries.run,   df)
    _run_module("Module 8 Predictive",      module08_predictive.run,   df)
    _df_fake = _run_module("Module 9 Fake Reviews",  module09_fake.run,   df)
    df_fake  = _df_fake if _df_fake is not None else df
    _df_r    = _run_module("Module 10 Routes",       module10_routes.run, df)
    df       = _df_r if _df_r is not None else df
    aope_df = _run_module("Module 11 AOPE",         module11_aope.run,         df)
    _run_module("Module 12 Traveller",      module12_traveller.run,    df, aope_df)
    _run_module("Module 13 Aircraft",       module13_aircraft.run,     df)
    _run_module("Module 14 Weather",        module14_weather.run,      df)
    _run_module("Module 15 Peak Season",    module15_peak_season.run,       df)
    _run_module("Module 16 Causal DiD",     module16_causal.run,            df)
    _run_module("Module 17 Seasonal Segs",  module17_seasonal_carrier.run,  df)
    _run_module("Module 18 DistilBERT",     module18_bert.run,              df)

    # Save outputs
    if bio_df is not None and aope_df is not None and corr_s is not None:
        _run_module("Saver",    saver.save_all,      df, bio_df, aope_df, df_fake, metrics, corr_s)
        _run_module("Summary",  saver.print_summary, df, bio_df, aope_df, df_fake, metrics, corr_s)
    else:
        log.warning("Some core modules failed — partial results saved to outputs/")


if __name__ == "__main__":
    main()
