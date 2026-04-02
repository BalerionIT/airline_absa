"""Module 13: Aircraft type."""
import re,logging,numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns
from config import SUB_COLS,OUTPUTS_DIR,SENT_COLOR
from pipeline.utils import banner,save_fig
log=logging.getLogger(__name__)
AC_CAT={"A320-family":"Narrowbody","A220":"Narrowbody","Boeing 737":"Narrowbody","Boeing 757":"Narrowbody","A330":"Widebody","A340":"Widebody","A350":"Widebody","A380":"Widebody","A300/A310":"Widebody","Boeing 747":"Widebody","Boeing 767":"Widebody","Boeing 777":"Widebody","Boeing 787":"Widebody","Embraer":"Regional Jet","CRJ":"Regional Jet","Fokker":"Regional Jet","Dash 8/Q400":"Turboprop","ATR":"Turboprop"}
AC_COLORS={"Narrowbody":"#3498DB","Widebody":"#E74C3C","Regional Jet":"#27AE60","Turboprop":"#8E44AD"}
CATS_ORDER=["Narrowbody","Widebody","Regional Jet","Turboprop"]
def _normalise(raw):
    if pd.isna(raw): return None
    s=str(raw).strip().upper()
    if re.search(r"A3(19|20|21)|AIRBUS\s*3(19|20|21)",s): return "A320-family"
    if re.search(r"A ?220",s): return "A220"
    if re.search(r"A ?330",s): return "A330"
    if re.search(r"A ?340",s): return "A340"
    if re.search(r"A ?350",s): return "A350"
    if re.search(r"A ?380",s): return "A380"
    if re.search(r"A ?300|A ?310",s): return "A300/A310"
    if re.search(r"7(37|38)\b|B ?737|737[ -]?(MAX|800|700|900|500|400|300)?",s): return "Boeing 737"
    if re.search(r"757",s): return "Boeing 757"
    if re.search(r"747",s): return "Boeing 747"
    if re.search(r"767",s): return "Boeing 767"
    if re.search(r"777",s): return "Boeing 777"
    if re.search(r"787|DREAMLINER",s): return "Boeing 787"
    if re.search(r"EMB|E1(70|75|90|95)|E ?190|ERJ",s): return "Embraer"
    if re.search(r"CRJ|CANADAIR|CR ?9",s): return "CRJ"
    if re.search(r"Q ?400|DASH ?8",s): return "Dash 8/Q400"
    if re.search(r"ATR",s): return "ATR"
    if re.search(r"FOKKER",s): return "Fokker"
    return None
def run(df):
    banner("MODULE 13 — AIRCRAFT TYPE ANALYSIS")
    df_ac=df[df["Aircraft"].notna()].copy()
    if len(df_ac)<10: log.warning("Module 13: no aircraft data."); return df
    df_ac["ac_family"]=df_ac["Aircraft"].apply(_normalise)
    df_ac["ac_category"]=df_ac["ac_family"].map(AC_CAT)
    df_ac=df_ac.dropna(subset=["ac_family","ac_category"]).copy()
    df_cat=df_ac[df_ac["ac_category"].isin(CATS_ORDER)].copy()
    cat_stat=df_cat.groupby("ac_category").agg(n=("Overall_Rating","count"),mean_rating=("Overall_Rating","mean")).reindex(CATS_ORDER).dropna()
    fig,axes=plt.subplots(2,2,figsize=(13,9)); fig.suptitle("MODULE 13 — Aircraft Type Analysis",fontsize=14,fontweight="bold")
    bars=axes[0,0].bar(cat_stat.index,cat_stat["mean_rating"],color=[AC_COLORS[c] for c in cat_stat.index],edgecolor="k")
    axes[0,0].set_ylim(0,8); axes[0,0].set_title("Mean Rating by Aircraft Category",fontweight="bold")
    for b,v,n in zip(bars,cat_stat["mean_rating"],cat_stat["n"]):
        axes[0,0].text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f"{v:.2f}\n(n={n:,})",ha="center",fontsize=9,fontweight="bold")
    bot=np.zeros(len(cat_stat))
    for cat in ["positive","neutral","negative"]:
        vals=df_cat.groupby("ac_category")["sentiment"].apply(lambda x,c=cat:(x==c).mean()*100).reindex(CATS_ORDER).fillna(0)
        axes[0,1].bar(CATS_ORDER,vals,bottom=bot,label=cat,color=SENT_COLOR[cat],edgecolor="white"); bot+=vals.values
    axes[0,1].set_title("Sentiment by Aircraft Type",fontweight="bold"); axes[0,1].legend(fontsize=9)
    hmap_ac=df_cat.groupby("ac_category")[SUB_COLS].mean().reindex(CATS_ORDER)
    sns.heatmap(hmap_ac.T,annot=True,fmt=".2f",cmap="RdYlGn",vmin=1,vmax=5,ax=axes[1,0],linewidths=0.5)
    axes[1,0].set_title("Sub-Ratings by Aircraft Category",fontweight="bold")
    df_nw=df_cat[df_cat["ac_category"].isin(["Narrowbody","Widebody"])].copy()
    df_nw=df_nw[df_nw["Seat Type"].isin(["Economy Class","Business Class"])].copy()
    piv=df_nw.groupby(["ac_category","Seat Type"])["Overall_Rating"].mean().unstack()
    piv=piv.reindex(index=["Narrowbody","Widebody"],columns=[c for c in ["Economy Class","Business Class"] if c in piv.columns])
    x=np.arange(len(piv)); w=0.35
    for i,col in enumerate(piv.columns):
        color="#4C72B0" if "Economy" in col else "#C44E52"
        b2=axes[1,1].bar(x+(i-0.5)*w,piv[col].fillna(0),w,label=col.replace(" Class",""),color=color,edgecolor="k",alpha=0.85)
        for b,v in zip(b2,piv[col].fillna(0)):
            axes[1,1].text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f"{v:.2f}",ha="center",fontsize=10,fontweight="bold")
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(piv.index); axes[1,1].set_title("Narrowbody vs Widebody by Class",fontweight="bold"); axes[1,1].legend()
    save_fig(f"{OUTPUTS_DIR}/aircraft_analysis.png")
    return df_ac
