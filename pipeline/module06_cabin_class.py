"""Module 6: Cabin class comparison."""
import logging,numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns
from scipy.stats import mannwhitneyu
from config import SUB_COLS,CLASSES,CLASS_COLORS,OUTPUTS_DIR
from pipeline.utils import banner,save_fig
log=logging.getLogger(__name__)
def run(df):
    banner("MODULE 6 — CABIN CLASS COMPARISON")
    df_c=df[df["Seat Type"].isin(CLASSES)].copy()
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    data_v=[df_c[df_c["Seat Type"]==c]["Overall_Rating"].dropna().values for c in CLASSES]
    parts=axes[0].violinplot(data_v,showmedians=True)
    for pc,c in zip(parts["bodies"],CLASSES): pc.set_facecolor(CLASS_COLORS[c]); pc.set_alpha(0.8)
    axes[0].set_xticks(range(1,len(CLASSES)+1)); axes[0].set_xticklabels([c.replace(" Class","") for c in CLASSES])
    axes[0].set_ylabel("Overall Rating"); axes[0].set_title("Rating by Cabin Class",fontweight="bold")
    heat=df_c.groupby("Seat Type")[SUB_COLS].mean().reindex(CLASSES)
    sns.heatmap(heat.T,annot=True,fmt=".2f",cmap="RdYlGn",vmin=1,vmax=5,ax=axes[1],linewidths=0.5)
    axes[1].set_title("Sub-Ratings by Cabin Class",fontweight="bold")
    axes[1].set_xticklabels([c.replace(" Class","") for c in CLASSES])
    save_fig(f"{OUTPUTS_DIR}/class_rating_overview.png")
    eco=df_c[df_c["Seat Type"]=="Economy Class"]; biz=df_c[df_c["Seat Type"]=="Business Class"]
    log.info("  Economy vs Business (Mann-Whitney U):")
    for col in ["Overall_Rating"]+SUB_COLS:
        e,b=eco[col].dropna(),biz[col].dropna()
        if len(e)<5 or len(b)<5: continue
        _,p=mannwhitneyu(e,b,alternative="two-sided")
        sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
        log.info(f"    {col:<30} Eco={e.mean():.2f} Biz={b.mean():.2f} {sig}")
    sub_labels=[c.replace(" & ","\n& ").replace(" Service","\nService") for c in SUB_COLS]
    N=len(SUB_COLS); angles=[n/float(N)*2*np.pi for n in range(N)]+[0]
    fig,ax=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))
    for c in CLASSES:
        means=df_c[df_c["Seat Type"]==c][SUB_COLS].mean().fillna(0).tolist(); vals=means+[means[0]]
        ax.plot(angles,vals,"o-",lw=2,label=c.replace(" Class",""),color=list(CLASS_COLORS.values())[CLASSES.index(c)])
        ax.fill(angles,vals,alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(sub_labels,fontsize=9)
    ax.set_ylim(0,5); ax.set_title("Sub-Rating Radar by Cabin Class",fontweight="bold",pad=20)
    ax.legend(loc="upper right",bbox_to_anchor=(1.4,1.1),fontsize=9)
    save_fig(f"{OUTPUTS_DIR}/class_radar.png")
    return df_c
