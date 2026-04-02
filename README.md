# Aspect-Based Sentiment Analysis of Airline Reviews
### A Domain-Aware NLP Study — Master's Thesis, Università degli Studi di Milano

**Author:** Antonella Convertini  
**Supervisor:** Prof. Alfio Ferrara
**Academic Year:** 2025–2026

---

## Overview

A modular end-to-end pipeline for Aspect-Based Sentiment Analysis (ABSA) of airline
passenger reviews scraped from [airlinequality.com](https://www.airlinequality.com) (SkyTrax).

The pipeline covers:
- Custom Selenium scraper with incremental update mode
- BIO sequence tagging for aspect extraction
- TF-IDF + Logistic Regression for aspect sentiment classification
- Fine-tuned DistilBERT (transformer-based ASC)
- Rating–aspect correlation and predictive modelling
- 10 thematic analysis modules (LCC vs Legacy, cabin class, time series,
  anomaly detection, route analysis, AOPE, traveller type, aircraft type,
  weather attribution, peak season)
- Causal inference (Difference-in-Differences) for event impact estimation
- Seasonal sensitivity by carrier type, cabin class, and haul type

**Dataset:** 144,934 reviews · 578 airlines · April 2012 – March 2026

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Reviews | 144,934 |
| Airlines | 578 |
| BIO spans extracted | 717,904 (95.8% coverage) |
| AOPE pairs | 419,153 |
| Lexicon baseline F1 | 0.462 |
| TF-IDF + LogReg F1 | 0.608 (+14.6pp) |
| DistilBERT F1 | 0.574 |
| Predictive model R² | 0.874 |
| Dominant satisfaction driver | Value For Money (r = 0.899) |
| NH peak-season penalty | −0.47 (p < 0.001) |
| Weather attribution gap | +0.20 (p < 0.001) |

---

## Pipeline Architecture

```
main.py
│
├── §1  Data Loading & Preprocessing
├── §2  Exploratory Data Analysis
├── §3–4  BIO Aspect Extraction
├── §5  Aspect Sentiment Classification (TF-IDF + LogReg)
├── §6  Rating–Aspect Alignment (Pearson correlation)
│
├── Module 5   LCC vs Legacy Comparison
├── Module 6   Cabin Class Comparison
├── Module 7   Time Series & Global Events (18 events annotated)
├── Module 8   Predictive Model + Delay Threshold (Ridge regression)
├── Module 9   Anomalous Review Detection (Isolation Forest)
├── Module 10  Route & Haul Analysis
├── Module 11  Aspect-Opinion Pair Extraction (AOPE)
├── Module 12  Traveller Type × Aspect Interaction
├── Module 13  Aircraft Type Analysis
├── Module 14  Weather Signal Analysis (attribution theory)
├── Module 15  Peak Season Analysis (hemisphere-aware)
├── Module 16  Causal Inference — Difference-in-Differences
├── Module 17  Seasonal Sensitivity by Carrier / Cabin / Haul
└── Module 18  Fine-Tuned DistilBERT ASC
```

---

## Installation

```bash
# Core dependencies
pip install requests beautifulsoup4 lxml pandas numpy matplotlib seaborn \
            scikit-learn scipy selenium webdriver-manager

# Module 18 (DistilBERT) — optional, large download ~350 MB
pip install torch transformers datasets
```

Python 3.10+ required. Chrome browser required for the scraper.

---

## Usage

```bash
# Run all modules on existing scraped CSV (recommended)
python main.py --no-scrape

# Scrape fresh data only (takes 4–5 hours for full corpus)
python main.py --scrape-only

# Scrape and run all modules
python main.py
```

### First run

1. Copy `skytrax_fresh.csv` into the `outputs/` folder (or run `--scrape-only` first)
2. Run `python main.py --no-scrape`
3. All charts and CSVs are saved to `outputs/`

### Module 18 (DistilBERT)

The first run trains and saves checkpoints to `outputs/bert_checkpoints/<aspect>/`.
Subsequent runs load from checkpoints automatically (~5 min vs ~3–4 hours for training).

Expected training time on CPU: ~3–4 hours total for all five aspects.  
Expected training time on GPU: ~20–30 minutes.

To force retraining, delete `outputs/bert_checkpoints/`.

---

## Project Structure

```
airline_absa/
├── main.py                          # Orchestrator — runs all 18 modules
├── config.py                        # Constants, paths, lexicons, aspect schema
├── requirements.txt                 # Package versions
├── README.md                        # This file
│
├── scraper/
│   └── skytrax_scraper.py           # Selenium scraper (incremental mode)
│
├── pipeline/
│   ├── utils.py                     # Shared utilities (banner, save_fig)
│   ├── loader.py                    # Data loading and normalisation
│   ├── eda.py                       # Exploratory data analysis
│   ├── bio_extraction.py            # BIO sequence tagging
│   ├── saver.py                     # Output saving
│   ├── module05_lcc_legacy.py       # LCC vs Legacy comparison
│   ├── module06_cabin_class.py      # Cabin class comparison
│   ├── module07_timeseries.py       # Time series + global events
│   ├── module08_predictive.py       # Ridge regression + delay threshold
│   ├── module09_fake.py             # Anomalous review detection
│   ├── module10_routes.py           # Route and haul analysis
│   ├── module11_aope.py             # Aspect-opinion pair extraction
│   ├── module12_traveller.py        # Traveller type × aspect
│   ├── module13_aircraft.py         # Aircraft type analysis
│   ├── module14_weather.py          # Weather signal analysis
│   ├── module15_peak_season.py      # Hemisphere-aware peak season
│   ├── module16_causal.py           # Difference-in-Differences causal inference
│   ├── module17_seasonal_carrier.py # Seasonal sensitivity by segment
│   └── module18_bert.py             # Fine-tuned DistilBERT ASC
│
├── outputs/                         # All charts, CSVs, checkpoints (gitignored)
│   ├── bert_checkpoints/            # DistilBERT model checkpoints (gitignored)
│   └── skytrax_fresh.csv            # Scraped data (gitignored)
│
└── .vscode/
    └── launch.json                  # VS Code run configurations
```

---

## Aspect Schema

| Aspect | Description | Key triggers |
|--------|-------------|--------------|
| Punctuality | Operational reliability | delay, cancelled, on time, missed connection |
| Cabin Comfort | Physical comfort | seat, legroom, recline, cramped, overhead |
| Cabin Crew Service | Staff behaviour | crew, staff, friendly, rude, helpful |
| Food & Beverages | Catering | food, meal, drink, catering, menu |
| Value For Money | Price perception | value, price, expensive, fare, ticket |

---

## Key Findings

### Model Comparison

| Aspect | Lexicon | TF-IDF | DistilBERT | BERT vs TF-IDF |
|--------|---------|--------|------------|----------------|
| Punctuality | 0.462 | 0.622 | 0.551 | −0.071 |
| Cabin Comfort | 0.462 | 0.608 | 0.593 | −0.015 |
| Cabin Crew Service | 0.462 | 0.603 | 0.590 | −0.013 |
| Food & Beverages | 0.462 | 0.618 | 0.619 | +0.001 |
| Value For Money | 0.462 | 0.584 | 0.520 | −0.064 |
| **Overall** | **0.462** | **0.608** | **0.574** | **−0.034** |

DistilBERT underperforms TF-IDF on 4 of 5 aspects. The performance gap correlates
with aspect ambiguity (largest for Value For Money, negligible for Food & Beverages),
providing evidence that **silver label noise is the binding constraint**, not model
capacity.

### Weather Attribution (Module 14)

| Condition | Mean rating | n |
|-----------|-------------|---|
| No weather, no delay | 4.57 | ~110,000 |
| Delay only | 2.64 | ~28,000 |
| Weather only (no delay) | **5.71** | 3,094 |
| Weather + delay | 3.77 | 4,364 |

Consistent with Weiner's (1985) attribution theory: passengers are more forgiving
when disruption is attributed to external uncontrollable causes.

### Causal Inference — DiD (Module 16)

| Event | DiD | Sig | Note |
|-------|-----|-----|------|
| EU Aviation Strikes 2023 | +1.329 | *** | Selection effect |
| Boeing 737 MAX Ban 2019 | +0.803 | *** | Capacity reduction |
| Post-COVID Boom 2022 | +0.730 | *** | Pent-up demand |
| Boeing Door Blowout 2024 | −0.301 | *** | Raw Δ = +0.32 (sign reversal!) |
| COVID-19 Pandemic 2020 | −0.893 | *** | Largest negative shock |

The Boeing Door Blowout illustrates why DiD is necessary: the raw change is positive
(+0.32) but the seasonal-corrected DiD is negative (−0.301). Without parallel-trends
correction the effect has the wrong sign.

---

## Methodological Notes

**Silver labels:** Aspect sentiment labels are derived automatically from SkyTrax
sub-ratings (≤2.0 = negative, ≤3.0 = neutral, >3.0 = positive). This enables
large-scale supervised training without manual annotation but introduces systematic
noise that disproportionately affects high-capacity models.

**Time-based split:** All classifiers use a temporal split (train: before Feb 2023,
test: after) to prevent data leakage. Random splits inflate R² by ~0.05 in the
predictive model.

**Anomaly detection:** Module 9 uses content-only signals (brevity, lexical
diversity, vagueness, Isolation Forest). Rating-based signals were explicitly
excluded after analysis showed they systematically flagged legitimate negative reviews.
Results should be interpreted as anomaly detection, not confirmed fake review counts.

**Hemisphere-aware seasons:** Module 15 defines NH peak = June–August and
SH peak = December–February. Applying a single June–August global definition
misclassifies ~10% of routes.

---

## Future Work

- Manual annotation of a gold-standard subset to properly evaluate silver label quality
- BERT-CRF architecture for aspect extraction (higher recall on paraphrastic references)
- Multilingual corpora for broader geographic representativeness
- Integration of external weather data (NOAA) and flight data (FlightAware)
- Synthetic control methods for causal estimation with heterogeneous pre-event trends
- Seasonal sensitivity analysis across finer-grained carrier business models

---

## Citation

```
Convertini, A. (2026). Aspect-Based Sentiment Analysis of Airline Reviews:
A Domain-Aware NLP Study. Master's Thesis, Università degli Studi di Milano.
```

---

## License

This project is released for academic purposes. The scraped review data belongs
to airlinequality.com (SkyTrax) and is not redistributed in this repository.
