"""Module 9: Fake review detection (content-based signals only)."""
import re,logging
import numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from config import OUTPUTS_DIR
from pipeline.utils import banner,save_fig
log=logging.getLogger(__name__)
SPEC_RE=r"\b([A-Z][a-z]+\s+[A-Z][a-z]+|[A-Z]{2,3}\d{3,4}|seat \d+|row \d+|terminal \d+|gate \d+)\b"
SIG_COLS=["s_extreme_length","s_low_diversity","s_title_overlap","s_text_outlier","s_vague","s_short_unverified"]
SIG_WEIGHTS=[0.15,0.20,0.15,0.25,0.15,0.10]
THRESHOLD=0.70; SHORT_THRESHOLD=15
def _ttr(text):
    toks=re.findall(r"\b[a-z]{3,}\b",text.lower())
    return len(set(toks))/len(toks) if len(toks)>=5 else 0.0
def _title_overlap(row):
    t=str(row["title_clean"]).lower().split(); r2=str(row["text"]).lower().split()[:20]
    if not t or len(t)<2: return 0
    ov=sum(1 for w in t if w in r2 and len(w)>3)
    return 1 if ov/max(len(t),1)>0.6 else 0
def run(df):
    banner("MODULE 9 — FAKE REVIEW DETECTION")
    df_f=df.copy()

    # Only flag very SHORT reviews — long detailed reviews are genuine, not fake
    df_f["s_extreme_length"]=(df_f["word_count"]<SHORT_THRESHOLD).astype(int)
    df_f["ttr"]=df_f["text"].apply(_ttr)
    df_f["s_low_diversity"]=(df_f["ttr"]<df_f["ttr"].quantile(0.08)).astype(int)
    df_f["s_title_overlap"]=df_f.apply(_title_overlap,axis=1)
    log.info("  Training Isolation Forest …")
    tfidf=TfidfVectorizer(max_features=500,ngram_range=(1,1),min_df=3,sublinear_tf=True)
    X_if=tfidf.fit_transform(df_f["text"]); iso=IsolationForest(contamination=0.05,random_state=42,n_jobs=-1)
    iso.fit(X_if); df_f["s_text_outlier"]=(iso.predict(X_if)==-1).astype(int); df_f["iso_score"]=iso.score_samples(X_if)
    df_f["s_vague"]=df_f["text"].apply(lambda t:0 if re.search(SPEC_RE,str(t)) else 1)
    df_f["verified_num"]=df_f["Verified"].astype(bool).astype(int)
    df_f["s_short_unverified"]=((df_f["word_count"]<SHORT_THRESHOLD)&(~df_f["Verified"].astype(bool))).astype(int)
    df_f["suspicion_score"]=sum(df_f[c]*w for c,w in zip(SIG_COLS,SIG_WEIGHTS))
    iso_norm=((df_f["iso_score"]-df_f["iso_score"].min())/(df_f["iso_score"].max()-df_f["iso_score"].min()))
    df_f["suspicion_score"]+=(1-iso_norm)*0.10; df_f["suspicion_score"]=df_f["suspicion_score"].clip(0,1)
    df_f["flagged_fake"]=(df_f["suspicion_score"]>=THRESHOLD).astype(int)
    n_flagged=df_f["flagged_fake"].sum()
    log.info(f"  Flagged: {n_flagged:,} / {len(df_f):,} ({n_flagged/len(df_f)*100:.1f}%)")
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    axes[0].hist(df_f[df_f["flagged_fake"]==0]["suspicion_score"],bins=40,alpha=0.6,color="#2ca02c",label="Genuine",density=True)
    axes[0].hist(df_f[df_f["flagged_fake"]==1]["suspicion_score"],bins=40,alpha=0.6,color="#d62728",label="Flagged",density=True)
    axes[0].axvline(THRESHOLD,color="black",lw=2,ls="--",label=f"Threshold={THRESHOLD}")
    axes[0].set_title("Suspicion Score Distribution",fontweight="bold"); axes[0].legend(fontsize=9)
    sig_means=pd.DataFrame({"Genuine":df_f[df_f["flagged_fake"]==0][SIG_COLS].mean(),"Flagged":df_f[df_f["flagged_fake"]==1][SIG_COLS].mean()})
    sig_means.index=[c.replace("s_","").replace("_"," ").title() for c in SIG_COLS]
    sns.heatmap(sig_means,annot=True,fmt=".2f",cmap="YlOrRd",vmin=0,vmax=1,ax=axes[1],linewidths=0.5)
    axes[1].set_title("Detection Signals (Content-Based Only)",fontweight="bold")
    for label,color,mask in [("Genuine","#2ca02c",df_f["flagged_fake"]==0),("Flagged","#d62728",df_f["flagged_fake"]==1)]:
        vc=df_f[mask]["Overall_Rating"].value_counts().sort_index()
        axes[2].plot(vc.index,vc.values/vc.sum()*100,"o-",color=color,lw=2,ms=7,label=label)
    axes[2].set_xlabel("Overall Rating"); axes[2].set_ylabel("% of Group")
    axes[2].set_title("Rating Distribution: Genuine vs Flagged",fontweight="bold"); axes[2].legend()
    plt.suptitle("MODULE 9 — Fake Review Detection (Content Signals Only)",fontsize=13,fontweight="bold",y=1.01)
    save_fig(f"{OUTPUTS_DIR}/fake_review_detection.png")
    return df_f
