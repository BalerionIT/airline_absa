"""
pipeline/module08_predictive.py — Predictive model + delay threshold analysis.

FIX: Uses a time-based train/test split (train on earlier years, test on later)
instead of random split, to prevent temporal data leakage. Reviews are sorted
by date_flown; the last 20% chronologically form the test set.
"""
import re, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import SUB_COLS, CLASSES, POS_LEX, NEG_LEX, OUTPUTS_DIR
from pipeline.utils import banner, save_fig

log = logging.getLogger(__name__)

DELAY_RE = r"\b(delay|delayed|late|cancelled|tarmac|cancel|missed|hours wait|waited)\b"
ANGER_RE = (r"\b(disgrace|disgusting|outrageous|unacceptable|furious|appalling|shocking|"
            r"catastrophic|awful|terrible|horrible|worst|never again|never fly|avoid|"
            r"pathetic|shameful|incompetent)\b")

def _extract_delay_hours(text):
    t=text.lower()
    if not re.search(DELAY_RE,t): return np.nan
    for pat in [r"(\d+)\s*hour",r"(\d+)h\b",r"over\s+(\d+)\s*hour",
                r"(\d+)\s*hrs?",r"more than\s+(\d+)",r"waited\s+(\d+)"]:
        m=re.search(pat,t)
        if m:
            h=float(m.group(1))
            if 0<h<=24: return h
    m2=re.search(r"(\d+)\s*min",t)
    if m2: return float(m2.group(1))/60
    return 0.5

def run(df: pd.DataFrame) -> pd.DataFrame:
    banner("MODULE 8 — PREDICTIVE MODEL + DELAY THRESHOLD")
    df_m = df[df["Seat Type"].isin(CLASSES)].copy()
    if len(df_m)<50:
        log.warning(f"Module 8 skipped — only {len(df_m)} rows match cabin classes.")
        return df

    df_m["seat_enc"]      =LabelEncoder().fit_transform(df_m["Seat Type"].fillna("Economy Class"))
    df_m["traveller_enc"] =LabelEncoder().fit_transform(df_m["Type Of Traveller"].fillna("Solo Leisure"))
    df_m["month_sin"]     =np.sin(2*np.pi*df_m["date_flown"].dt.month.fillna(1)/12)
    df_m["month_cos"]     =np.cos(2*np.pi*df_m["date_flown"].dt.month.fillna(1)/12)
    df_m["is_summer"]     =df_m["date_flown"].dt.month.isin([7,8]).astype(int)
    df_m["is_christmas"]  =df_m["date_flown"].dt.month.isin([12]).astype(int)
    df_m["is_covid"]      =((df_m["date_flown"]>="2020-03-01")&(df_m["date_flown"]<="2021-06-01")).astype(int)
    df_m["post_covid"]    =(df_m["date_flown"]>"2021-06-01").astype(int)
    df_m["delay_mention"] =df_m["text"].apply(lambda t: 1 if re.search(DELAY_RE,str(t).lower()) else 0)
    df_m["anger"]         =df_m["text"].apply(lambda t: 1 if re.search(ANGER_RE,str(t).lower()) else 0)
    df_m["lex_norm"]      =df_m["text"].apply(
        lambda t: sum(1 if w in POS_LEX else -1 if w in NEG_LEX else 0
                      for w in re.findall(r"\b\w+\b",t.lower()))/(len(t.split())+1))

    available_sub=[c for c in SUB_COLS if c in df_m.columns and df_m[c].notna().sum()>50]
    if available_sub:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            for c in available_sub:
                df_m[c]=df_m[c].fillna(df_m[c].median())
    else:
        log.warning("  No sub-rating columns have data — model runs on text/context features only.")

    FEAT=available_sub+["seat_enc","traveller_enc","month_sin","month_cos",
                        "is_summer","is_christmas","is_covid","post_covid",
                        "delay_mention","anger","lex_norm","word_count"]

    data=df_m[FEAT+["Overall_Rating","date_flown"]].dropna()
    if len(data)<50:
        log.warning(f"Module 8 skipped — only {len(data)} complete rows after dropna.")
        return df_m

    # ── FIX 3: Time-based split instead of random split ──────────────────────
    # Sort by flight date, use last 20% chronologically as test set.
    # This prevents the model from using future reviews to predict past ones.
    data=data.sort_values("date_flown").reset_index(drop=True)
    split_idx=int(len(data)*0.80)
    train_data=data.iloc[:split_idx]
    test_data =data.iloc[split_idx:]

    cutoff_date=data["date_flown"].iloc[split_idx]
    log.info(f"  Time-based split: train up to {cutoff_date.strftime('%Y-%m')}, "
             f"test after ({len(train_data):,} train / {len(test_data):,} test)")

    X_tr=np.nan_to_num(train_data[FEAT].values,nan=0.0,posinf=0.0,neginf=0.0)
    X_te=np.nan_to_num(test_data[FEAT].values, nan=0.0,posinf=0.0,neginf=0.0)
    X_tr=np.clip(X_tr,-1e6,1e6); X_te=np.clip(X_te,-1e6,1e6)
    y_tr=train_data["Overall_Rating"].values
    y_te=test_data["Overall_Rating"].values

    sc=StandardScaler(); ridge=Ridge(alpha=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        ridge.fit(sc.fit_transform(X_tr),y_tr)
        y_pr=ridge.predict(sc.transform(X_te))

    rmse=np.sqrt(mean_squared_error(y_te,y_pr))
    r2  =r2_score(y_te,y_pr)
    log.info(f"  Ridge RMSE={rmse:.3f} R²={r2:.3f} (time-based split, no leakage)")

    fig,ax=plt.subplots(figsize=(9,6))
    coef_df=pd.DataFrame({"feature":FEAT,"coef":ridge.coef_}).sort_values("coef",ascending=False)
    ax.barh(coef_df["feature"],coef_df["coef"],
            color=["#2ca02c" if v>0 else "#d62728" for v in coef_df["coef"]],edgecolor="k")
    ax.axvline(0,color="black",lw=0.8)
    ax.set_title("Feature Importance — Rating Prediction (Ridge, time-based split)",fontweight="bold")
    save_fig(f"{OUTPUTS_DIR}/predictive_feature_importance.png")

    # Delay threshold analysis
    df_d=df_m[df_m["delay_mention"]==1].copy()
    df_d["delay_hours"]=df_d["text"].apply(_extract_delay_hours).fillna(0.5)
    bins=[0,.5,1,2,3,5,8,24]; labels=["<30min","30m-1h","1-2h","2-3h","3-5h","5-8h","8h+"]
    df_d["delay_bin"]=pd.cut(df_d["delay_hours"],bins=bins,labels=labels)
    profile=df_d.groupby("delay_bin",observed=True).agg(
        n=("Overall_Rating","count"),mean_rating=("Overall_Rating","mean"),
        pct_neg=("sentiment",lambda x:(x=="negative").mean()*100),
        pct_angry=("anger","mean")).reset_index().dropna()

    if len(profile)>0:
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        colors_d=plt.cm.RdYlGn_r(np.linspace(0.1,0.9,len(profile)))
        bars=axes[0].bar(profile["delay_bin"],profile["mean_rating"],color=colors_d,edgecolor="k")
        axes[0].set_ylabel("Mean Rating"); axes[0].set_ylim(0,10)
        axes[0].set_title("Rating by Delay Duration",fontweight="bold")
        for b,v in zip(bars,profile["mean_rating"]):
            axes[0].text(b.get_x()+b.get_width()/2,b.get_height()+0.1,
                         f"{v:.1f}",ha="center",fontsize=9,fontweight="bold")
        axes[1].plot(range(len(profile)),profile["pct_neg"],"o-",color="#d62728",lw=2.5,ms=8,label="% Negative")
        axes[1].plot(range(len(profile)),profile["pct_angry"]*100,"s--",color="#8B0000",lw=2,ms=7,label="% Angry")
        axes[1].set_xticks(range(len(profile))); axes[1].set_xticklabels(profile["delay_bin"],rotation=20)
        axes[1].set_ylabel("% Reviews"); axes[1].set_ylim(0,105)
        axes[1].set_title("Frustration → Anger Switch",fontweight="bold"); axes[1].legend(fontsize=9)
        axes[2].bar(profile["delay_bin"],profile["n"],color="#4C72B0",edgecolor="k",alpha=0.8)
        axes[2].set_ylabel("Review Count"); axes[2].set_title("Volume of Delay Reviews",fontweight="bold")
        plt.suptitle("Delay Duration as a Sentiment Switch",fontsize=13,fontweight="bold",y=1.01)
        save_fig(f"{OUTPUTS_DIR}/delay_threshold_analysis.png")
    return df_m
