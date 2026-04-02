"""pipeline/eda.py — §2 EDA."""
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from config import SUB_COLS, OUTPUTS_DIR
from pipeline.utils import banner, save_fig
log = logging.getLogger(__name__)

def run(df):
    banner("§2  EDA")
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    vc=df["Overall_Rating"].value_counts().sort_index()
    axes[0].bar(vc.index,vc.values,color="#4C72B0",edgecolor="k",width=0.7)
    axes[0].set_title("Overall Rating Distribution",fontweight="bold")
    axes[0].set_xlabel("Rating (1–10)"); axes[0].set_ylabel("Reviews")
    axes[1].hist(df["word_count"].clip(upper=500),bins=50,color="#DD8452",edgecolor="k")
    axes[1].set_title("Review Length (words)",fontweight="bold")
    save_fig(f"{OUTPUTS_DIR}/eda_overview.png")
    sub_means=df[SUB_COLS].mean()
    fig,ax=plt.subplots(figsize=(10,4))
    colors_=sns.color_palette("Set2",len(SUB_COLS))
    bars=ax.bar([c.replace(" & ","\n& ") for c in SUB_COLS],sub_means.values,color=colors_,edgecolor="k")
    for b,v in zip(bars,sub_means.values):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f"{v:.2f}",ha="center",fontsize=9)
    ax.set_ylim(0,5.5); ax.set_ylabel("Mean Sub-Rating (1–5)")
    ax.set_title("Average Sub-Ratings Across All Reviews",fontweight="bold")
    save_fig(f"{OUTPUTS_DIR}/eda_sub_ratings.png")
    log.info("EDA complete.")
