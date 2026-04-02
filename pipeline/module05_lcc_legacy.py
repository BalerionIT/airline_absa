"""Module 5: LCC vs Legacy."""
import logging, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from config import SUB_COLS,OUTPUTS_DIR
from pipeline.utils import banner,save_fig
log=logging.getLogger(__name__)
LCC_SET={"Ryanair","easyJet","Wizz Air","Norwegian","Vueling Airlines","Transavia","Germanwings","Eurowings","Volotea","flybe","Blue Air","Laudamotion","Monarch Airlines","Air Berlin","Jet2.com","Primera Air","NIKI","Pobeda Airlines","Iberia Express","BA CityFlyer"}
LEGACY_SET={"Lufthansa","Air France","British Airways","KLM Royal Dutch Airlines","Iberia","Alitalia","ITA Airways","Swiss International Air Lines","Austrian Airlines","Finnair","TAP Portugal","Brussels Airlines","LOT Polish Airlines","CSA Czech Airlines","Aegean Airlines","Croatia Airlines","airBaltic","Air Malta","Air Serbia","Adria Airways","Bulgaria Air","Belavia","Aeroflot Russian Airlines","Turkish Airlines","Olympic Air","Air Moldova","Montenegro Airlines","SAS Scandinavian Airlines"}
def run(df):
    banner("MODULE 5 — LCC vs LEGACY")
    df_eu=df.copy()
    df_eu["carrier_type"]=df_eu["Airline Name"].apply(lambda n:"LCC" if n in LCC_SET else "Legacy" if n in LEGACY_SET else "Other")
    df_eu=df_eu[df_eu["carrier_type"].isin(["LCC","Legacy"])].copy()
    log.info(f"European dataset: {len(df_eu):,} reviews (LCC={df_eu['carrier_type'].eq('LCC').sum():,}, Legacy={df_eu['carrier_type'].eq('Legacy').sum():,})")
    lcc_r=df_eu[df_eu["carrier_type"]=="LCC"]["Overall_Rating"].dropna()
    leg_r=df_eu[df_eu["carrier_type"]=="Legacy"]["Overall_Rating"].dropna()
    _,p=mannwhitneyu(lcc_r,leg_r,alternative="two-sided")
    log.info(f"  LCC mean={lcc_r.mean():.2f}  Legacy mean={leg_r.mean():.2f}  p={p:.4f}")
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("LCC vs Legacy: Performance Comparison",fontsize=13,fontweight="bold")
    cats=["positive","neutral","negative"]
    for i,(ctype,color) in enumerate([("LCC","#4C72B0"),("Legacy","#C44E52")]):
        sub=df_eu[df_eu["carrier_type"]==ctype]
        vals=[sub["sentiment"].eq(c).mean()*100 for c in cats]
        axes[0].bar(np.arange(3)+i*0.35,vals,0.35,label=ctype,color=color,edgecolor="k",alpha=0.85)
    axes[0].set_xticks([0.175,1.175,2.175]); axes[0].set_xticklabels(cats)
    axes[0].set_ylabel("% Reviews"); axes[0].set_title("Sentiment Distribution",fontweight="bold"); axes[0].legend()
    sm_lcc=df_eu[df_eu["carrier_type"]=="LCC"][SUB_COLS].mean()
    sm_leg=df_eu[df_eu["carrier_type"]=="Legacy"][SUB_COLS].mean()
    x=np.arange(len(SUB_COLS)); w=0.35
    axes[1].bar(x-w/2,sm_lcc.values,w,label="LCC",color="#4C72B0",edgecolor="k",alpha=0.85)
    axes[1].bar(x+w/2,sm_leg.values,w,label="Legacy",color="#C44E52",edgecolor="k",alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels([c.replace(" & ","\n& ").replace(" Service","\nSvc") for c in SUB_COLS],fontsize=8)
    axes[1].set_ylim(0,5); axes[1].set_title("Sub-Rating Comparison",fontweight="bold"); axes[1].legend()
    save_fig(f"{OUTPUTS_DIR}/lcc_vs_legacy_comparison.png")
    return df_eu
