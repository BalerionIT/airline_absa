# ABSA Airline Review Pipeline

Aspect-Based Sentiment Analysis of airline reviews — full 13-module thesis pipeline.

---

## Folder structure

```
airline_absa/
├── main.py                    ← entry point — run this
├── config.py                  ← all constants, paths, lexicons
├── requirements.txt
│
├── data/
│   └── Airline_review.csv     ← fallback dataset (Kaggle, 22 k reviews)
│
├── scraper/
│   └── skytrax_scraper.py     ← incremental SkyTrax downloader
│
├── pipeline/
│   ├── utils.py               ← shared helpers
│   ├── loader.py              ← §1  data loading & cleaning
│   ├── eda.py                 ← §2  exploratory analysis
│   ├── bio_extraction.py      ← §3-6 BIO tagging · ASC · alignment
│   ├── module05_lcc_legacy.py
│   ├── module06_cabin_class.py
│   ├── module07_timeseries.py
│   ├── module08_predictive.py
│   ├── module09_fake.py
│   ├── module10_routes.py
│   ├── module11_aope.py
│   ├── module12_traveller.py
│   └── module13_aircraft.py
│
├── outputs/                   ← all charts + CSVs written here (auto-created)
│
└── .vscode/
    └── launch.json            ← VS Code run configurations
```

---

## Setup

```bash
pip install -r requirements.txt
python main.py
```

---

## Data acquisition logic

```
python main.py   (default)
  │
  ├─ outputs/skytrax_fresh.csv exists?
  │     YES → incremental scrape (only new reviews since last run)
  │     NO  → full scrape of all airlines
  │
  ├─ Scrape succeeded?
  │     YES → run analysis on scraped data
  │     NO  → fall back to data/Airline_review.csv
  │
  └─ Run all 13 analysis modules
```

The scraper always serves newest reviews first and stops per-airline the
moment it hits reviews older than the newest date in your existing CSV.
Re-running daily or weekly only downloads what is genuinely new.

---

## CLI options

| Flag | What it does |
|------|-------------|
| *(none)* | Scrape (incremental if possible) → fallback → full analysis |
| `--scrape-only` | Update / create the scraped CSV, then stop |
| `--no-scrape` | Skip scraper, use `data/Airline_review.csv` directly |
| `--airlines british-airways,ryanair` | Scrape specific airlines only |
| `--max-pages 3` | Limit pages per airline (good for testing) |
| `--delay-min 3 --delay-max 6` | Slower, more polite scraping |
| `--csv /path/to/file.csv` | Use a specific file, skip scraper entirely |

---

## VS Code

Open the project folder in VS Code.
Go to **Run & Debug** (Ctrl+Shift+D / Cmd+Shift+D) and choose a profile:

| Profile | Description |
|---------|-------------|
| `▶  Run Pipeline` | Default — scrape first, fall back to CSV, then analyse |
| `🔄  Scrape only` | Update the scraped CSV without running analysis |
| `📂  No scrape` | Skip scraper, run analysis on the local CSV |
| `🧪  Quick test` | 3 airlines · 2 pages · full analysis (fast) |
| `🧪  Quick scrape only` | 3 airlines · 2 pages · stop after scrape |

---

## Outputs

Everything is written to `outputs/`:

| File | Description |
|------|-------------|
| `annotated_reviews_full.csv` | Full dataset with all derived fields |
| `aspect_span_predictions.csv` | BIO-tagged spans with ASC labels |
| `aope_pairs.csv` | Aspect-opinion pairs (Module 11) |
| `flagged_fake_reviews.csv` | Suspected fake reviews (Module 9) |
| `evaluation_summary.csv` | P / R / F1 per aspect and overall |
| `subrating_correlation.csv` | Pearson r of sub-ratings vs overall |
| `skytrax_fresh.csv` | Scraped data (only present after a scrape run) |
| `pipeline.log` | Full timestamped execution log |
| `*.png` | 21 charts covering all modules |
