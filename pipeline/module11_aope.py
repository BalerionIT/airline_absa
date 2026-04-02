"""Module 11: AOPE."""
import re,logging,pandas as pd,matplotlib.pyplot as plt
from config import ASPECT_TERMS,ASPECTS,OPINION_LEXICON,NEG_WORDS,OUTPUTS_DIR,SENT_COLOR
from pipeline.utils import banner,save_fig
log=logging.getLogger(__name__)
WINDOW=6
def extract_pairs(text):
    tokens=re.findall(r"\b[a-z\']+\b",text.lower()); n=len(tokens); pairs=[]; triggers=[]
    for asp,terms in ASPECT_TERMS.items():
        for term in sorted(terms,key=lambda t:-len(t.split())):
            tlen=len(term.split()); tt=term.lower().split()
            for i in range(n-tlen+1):
                if tokens[i:i+tlen]==tt: triggers.append((i,i+tlen-1,asp,term))
    used=set()
    for start,end,asp,term in sorted(triggers,key=lambda x:-(x[1]-x[0])):
        if any(j in used for j in range(start,end+1)): continue
        for j in range(start,end+1): used.add(j)
        win_start=max(0,start-WINDOW); win_end=min(n-1,end+WINDOW)
        win_toks=tokens[win_start:win_end+1]; neg_pos={k for k,t in enumerate(win_toks) if t in NEG_WORDS}
        best_op,best_dist=None,WINDOW+1
        for k,tok in enumerate(win_toks):
            if tok not in OPINION_LEXICON: continue
            dist=min(abs(k-(start-win_start)),abs(k-(end-win_start)))
            if dist<best_dist:
                negated=any(abs(k-np_)<=3 and np_<k for np_ in neg_pos)
                pol_raw=OPINION_LEXICON[tok]; pol=-pol_raw if negated else pol_raw
                best_op={"opinion_word":tok,"polarity":pol,"intensity":abs(pol),"negated":negated,"sentiment":"positive" if pol>0 else "negative" if pol<0 else "neutral"}
                best_dist=dist
        if best_op: pairs.append({"aspect":asp,"aspect_token":term,**best_op})
    return pairs
def run(df):
    banner("MODULE 11 — AOPE"); log.info("  Extracting AOPE pairs …")
    records=[]
    for idx,row in df.iterrows():
        for p in extract_pairs(row["text"]): p["review_idx"]=idx; p["overall_rating"]=row["Overall_Rating"]; records.append(p)
    aope_df=pd.DataFrame(records)
    log.info(f"  {len(aope_df):,} pairs | {aope_df['review_idx'].nunique():,} reviews | negation rate={aope_df['negated'].mean()*100:.1f}%")
    word_pol=aope_df.groupby("opinion_word")["polarity"].mean()
    fig,axes=plt.subplots(1,len(ASPECTS),figsize=(18,3))
    for ax,asp in zip(axes,ASPECTS):
        sub=aope_df[aope_df["aspect"]==asp]["opinion_word"].value_counts().head(8)
        colors=[SENT_COLOR["positive"] if word_pol.get(w,0)>0 else SENT_COLOR["negative"] if word_pol.get(w,0)<0 else SENT_COLOR["neutral"] for w in sub.index]
        ax.barh(sub.index[::-1],sub.values[::-1],color=colors[::-1],edgecolor="k")
        ax.set_title(asp.replace(" & ","\n& "),fontweight="bold",fontsize=8); ax.tick_params(labelsize=7)
    plt.suptitle("Top Opinion Words per Aspect (AOPE)",fontsize=11,fontweight="bold")
    save_fig(f"{OUTPUTS_DIR}/aope_opinion_words.png")
    return aope_df
