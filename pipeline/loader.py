"""pipeline/loader.py — §1: Load, normalise columns, clean and enrich."""
import logging
import pandas as pd
from pathlib import Path
from config import SUB_COLS
from pipeline.utils import clean_text, rating_to_label, lexicon_score, banner

log = logging.getLogger(__name__)

RENAME_MAP = {
    "airline_name":"Airline Name","airline name":"Airline Name",
    "overall_rating":"Overall_Rating","overall rating":"Overall_Rating",
    "review_title":"Review_Title","review title":"Review_Title",
    "review_date":"Review Date","review date":"Review Date",
    "review":"Review","aircraft":"Aircraft",
    "type_of_traveller":"Type Of Traveller","type of traveller":"Type Of Traveller",
    "seat_type":"Seat Type","seat type":"Seat Type",
    "route":"Route","date_flown":"Date Flown","date flown":"Date Flown",
    "seat_comfort":"Seat Comfort","cabin_staff_service":"Cabin Staff Service",
    "food_beverages":"Food & Beverages","food & beverages":"Food & Beverages",
    "ground_service":"Ground Service","inflight_entertainment":"Inflight Entertainment",
    "wifi_connectivity":"Wifi & Connectivity","wifi & connectivity":"Wifi & Connectivity",
    "value_for_money":"Value For Money","recommended":"Recommended","verified":"Verified",
}

def load(csv_path: Path) -> pd.DataFrame:
    banner("§1  DATA LOADING & PREPROCESSING")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    log.info(f"Raw columns: {list(df.columns)}")
    df.rename(columns={c:RENAME_MAP[c.lower()] for c in df.columns if c.lower() in RENAME_MAP},inplace=True)
    for col in ["Airline Name","Review","Overall_Rating","Review_Title","Review Date",
                "Aircraft","Type Of Traveller","Seat Type","Route","Date Flown",
                "Recommended","Verified"]+SUB_COLS:
        if col not in df.columns:
            df[col]=None
            log.warning(f"  Column '{col}' missing — filled with None")
    df["Overall_Rating"]=pd.to_numeric(df["Overall_Rating"],errors="coerce")
    df=df.dropna(subset=["Overall_Rating","Review"]).copy()
    df["Overall_Rating"]=df["Overall_Rating"].astype(int)
    for c in SUB_COLS:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["text"]           =df["Review"].apply(clean_text)
    df["Recommended_bin"]=(df["Recommended"].astype(str).str.strip().str.lower()=="yes").astype(int)
    df["date_flown"]     =pd.to_datetime(df["Date Flown"],format="%B %Y",errors="coerce")
    df["review_date"]    =pd.to_datetime(df["Review Date"],errors="coerce")
    df["word_count"]     =df["text"].str.split().str.len()
    df["Review_Title"]   =df["Review_Title"].fillna("")
    df["title_clean"]    =df["Review_Title"].astype(str).str.strip('"').str.strip()
    df["sentiment"]      =df["Overall_Rating"].apply(rating_to_label)
    df["lex_sentiment"]  =df["text"].apply(lexicon_score)
    df["disc_flag"]=(
        ((df["lex_sentiment"]=="positive")&(df["Overall_Rating"]<=3))|
        ((df["lex_sentiment"]=="negative")&(df["Overall_Rating"]>=7)))
    log.info(f"Loaded {len(df):,} reviews | {df['Airline Name'].nunique()} airlines | "
             f"date range: {df['date_flown'].min()} – {df['date_flown'].max()}")
    return df
