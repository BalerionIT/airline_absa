"""Module 12: Traveller type x aspect."""
import re,logging,numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns
from config import ASPECTS,SUB_COLS,CLASSES,TRAVELLERS,TRAV_COLORS,OUTPUTS_DIR
from pipeline.utils import banner,save_fig
log=logging.getLogger(__name__)
DELAY_RE=r"\b(delay|delayed|late|cancelled|tarmac|cancel|missed|hours wait|waited)\b"
def run(df,aope_df):
    banner("MODULE 12 — TRAVELLER TYPE × ASPECT INTERACTION")
    df_tv=df[df["Type Of Traveller"].isin(TRAVELLERS)].copy()
    df_tv["seat_clean"]=df_tv["Seat Type"].where(df_tv["Seat Type"].isin(CLASSES),"Economy Class")
    pivot_rating=df_tv.groupby(["Type Of Traveller","seat_clean"])["Overall_Rating"].mean().unstack()
    pivot_rating=pivot_rating.reindex(index=TRAVELLERS,columns=[s for s in CLASSES if s in pivot_rating.columns])
    vfm_pivot=df_tv.groupby(["Type Of Traveller","seat_clean"])["Value For Money"].mean().unstack()
    vfm_pivot=vfm_pivot.reindex(index=TRAVELLERS,columns=[s for s in CLASSES if s in vfm_pivot.columns])
    aope_tv=aope_df.merge(df_tv[["Type Of Traveller"]].reset_index().rename(columns={"index":"review_idx"}),on="review_idx",how="left").dropna(subset=["Type Of Traveller"])
    asp_pol=aope_tv.groupby(["Type Of Traveller","aspect"])["polarity"].mean().unstack()
    asp_pol=asp_pol.reindex(index=TRAVELLERS,columns=[a for a in ASPECTS if a in asp_pol.columns])
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    fig.suptitle("MODULE 12 — Traveller Type × Seat Class",fontsize=14,fontweight="bold")
    sns.heatmap(pivot_rating,annot=True,fmt=".2f",cmap="RdYlGn",vmin=1,vmax=7,ax=axes[0,0],linewidths=0.5)
    axes[0,0].set_title("Mean Rating: Traveller × Class")
    x=np.arange(len(SUB_COLS)); w=0.2
    for i,trav in enumerate(TRAVELLERS):
        sm=df_tv[df_tv["Type Of Traveller"]==trav][SUB_COLS].mean()
        axes[0,1].bar(x+(i-1.5)*w,sm.values,w,label=trav.replace(" Leisure",""),color=list(TRAV_COLORS.values())[i],edgecolor="k",alpha=0.85)
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels([c.replace(" & ","\n& ").replace(" Service","\nSvc") for c in SUB_COLS],fontsize=8)
    axes[0,1].set_title("Sub-Rating by Traveller Type"); axes[0,1].legend(fontsize=8)
    sns.heatmap(vfm_pivot,annot=True,fmt=".2f",cmap="RdYlGn",vmin=1,vmax=5,ax=axes[1,0],linewidths=0.5)
    axes[1,0].set_title("Value For Money: Traveller × Class")
    if not asp_pol.empty:
        sns.heatmap(asp_pol,annot=True,fmt=".2f",cmap="RdYlGn",center=0,ax=axes[1,1],linewidths=0.5)
        axes[1,1].set_title("AOPE Polarity: Traveller × Aspect")
        axes[1,1].set_xticklabels(ASPECTS,rotation=30,ha="right",fontsize=8)
    save_fig(f"{OUTPUTS_DIR}/traveller_aspect_interaction.png")
    df_tv["delay_mention"]=df_tv["text"].apply(lambda t:1 if re.search(DELAY_RE,str(t).lower()) else 0)
    dr=df_tv.groupby(["Type Of Traveller","delay_mention"])["Overall_Rating"].mean().unstack()
    dr.columns=["No Delay","Delay Mentioned"]; dr=dr.reindex(TRAVELLERS).dropna()
    fig,ax=plt.subplots(figsize=(9,4)); x2=np.arange(len(dr)); w2=0.35
    b1=ax.bar(x2-w2/2,dr["No Delay"],w2,label="No Delay",color="#2ca02c",edgecolor="k",alpha=0.85)
    b2=ax.bar(x2+w2/2,dr["Delay Mentioned"],w2,label="Delay",color="#d62728",edgecolor="k",alpha=0.85)
    ax.set_xticks(x2); ax.set_xticklabels([t.replace(" Leisure","") for t in dr.index])
    ax.set_ylabel("Mean Rating"); ax.set_ylim(0,8); ax.set_title("Delay Sensitivity by Traveller Type",fontweight="bold"); ax.legend()
    for b,v in list(zip(b1,dr["No Delay"]))+list(zip(b2,dr["Delay Mentioned"])):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f"{v:.1f}",ha="center",fontsize=9)
    save_fig(f"{OUTPUTS_DIR}/traveller_delay_sensitivity.png")
    return df_tv
