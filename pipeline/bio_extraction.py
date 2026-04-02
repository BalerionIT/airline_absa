"""pipeline/bio_extraction.py — §3-6: BIO, ASC, Alignment."""
import logging, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from config import ASPECT_TERMS,ASPECTS,ASP2SUB,SUB_COLS,OUTPUTS_DIR
from pipeline.utils import banner,save_fig,sr_to_label,rating_to_label,lexicon_score
log=logging.getLogger(__name__)

def bio_tag_sentence(text):
    tokens=[(m.group(),m.start()) for m in re.finditer(r"\S+",text)]
    tok_low=[t.lower() for t,_ in tokens]; n=len(tokens)
    tags=["O"]*n
    triggers=sorted([(asp,term,len(term.split())) for asp,terms in ASPECT_TERMS.items() for term in terms],key=lambda x:-x[2])
    matched=[False]*n
    for asp,term,tlen in triggers:
        tt=term.lower().split()
        for i in range(n-tlen+1):
            if any(matched[i:i+tlen]): continue
            if tok_low[i:i+tlen]==tt:
                tags[i]=f"B-{asp}"
                for j in range(1,tlen): tags[i+j]=f"I-{asp}"
                for j in range(tlen): matched[i+j]=True
    return [(tok,tag) for (tok,_),tag in zip(tokens,tags)]

def extract_spans(bio_seq):
    spans,cur_asp,cur_toks,cur_start=[],None,[],- 1
    for idx,(tok,tag) in enumerate(bio_seq):
        if tag.startswith("B-"):
            if cur_asp: spans.append({"aspect":cur_asp,"tokens":cur_toks,"start_idx":cur_start,"end_idx":idx-1})
            cur_asp,cur_toks,cur_start=tag[2:],[tok],idx
        elif tag.startswith("I-") and cur_asp and tag[2:]==cur_asp: cur_toks.append(tok)
        else:
            if cur_asp: spans.append({"aspect":cur_asp,"tokens":cur_toks,"start_idx":cur_start,"end_idx":idx-1})
            cur_asp,cur_toks,cur_start=None,[],- 1
    if cur_asp: spans.append({"aspect":cur_asp,"tokens":cur_toks,"start_idx":cur_start,"end_idx":len(bio_seq)-1})
    return spans

def run_bio(df):
    banner("§3-4  BIO ASPECT EXTRACTION"); log.info("Running BIO tagger …")
    bio_records,aspects_map={},{}
    for idx,row in df.iterrows():
        bio_seq=bio_tag_sentence(row["text"]); spans=extract_spans(bio_seq)
        asp_set=list(dict.fromkeys(s["aspect"] for s in spans))
        aspects_map[idx]=asp_set if asp_set else ["General"]
        tokens=[t for t,_ in bio_seq]
        for span in spans:
            s_idx=span["start_idx"]; e_idx=span["end_idx"]
            ctx=" ".join(tokens[max(0,s_idx-10):min(len(tokens)-1,e_idx+10)+1])
            sub_col=ASP2SUB.get(span["aspect"]) or ""
            bio_records.setdefault(idx,[])
            bio_records[idx].append({"review_idx":idx,"aspect":span["aspect"],
                "aspect_tokens":" ".join(span["tokens"]),"context":ctx,
                "overall_rating":row["Overall_Rating"],
                "sub_rating":row[sub_col] if sub_col and sub_col in df.columns else np.nan})
    df=df.copy()
    df["aspects_detected"]=df.index.map(aspects_map)
    df["n_aspects"]=df["aspects_detected"].apply(lambda x:len([a for a in x if a!="General"]))
    bio_df=pd.DataFrame([r for rows in bio_records.values() for r in rows])
    log.info(f"Total spans: {len(bio_df):,} | Reviews with ≥1: {bio_df['review_idx'].nunique():,} ({bio_df['review_idx'].nunique()/len(df)*100:.1f}%)")
    span_freq=bio_df["aspect"].value_counts(); n_asp_vc=df["n_aspects"].value_counts().sort_index()
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].barh(span_freq.index,span_freq.values,edgecolor="k",color=sns.color_palette("tab10",len(ASPECTS)))
    axes[0].set_title("Aspect Span Count (BIO)",fontweight="bold")
    axes[1].bar(n_asp_vc.index,n_asp_vc.values,color="#4C72B0",edgecolor="k")
    axes[1].set_title("Multi-Aspect Distribution",fontweight="bold")
    save_fig(f"{OUTPUTS_DIR}/bio_aspect_extraction.png")
    return df,bio_df

def run_asc(df,bio_df):
    banner("§5  ASPECT SENTIMENT CLASSIFICATION")
    def get_silver(row):
        sub_col=ASP2SUB.get(row["aspect"])
        if sub_col:
            lbl=sr_to_label(row["sub_rating"])
            if lbl: return lbl
        return rating_to_label(row["overall_rating"])
    bio_df=bio_df.copy()
    bio_df["silver_label"]=bio_df.apply(get_silver,axis=1)
    bio_df["pred_lexicon"]=bio_df["context"].apply(lexicon_score)
    labelled=bio_df.dropna(subset=["silver_label"]).copy()
    clf_results={}; all_true,all_pred=[],[]
    for asp in ASPECTS:
        sub=labelled[labelled["aspect"]==asp].copy()
        if len(sub)<50: continue
        X,y=sub["context"].values,sub["silver_label"].values
        vc=pd.Series(y).value_counts(); minn=vc.min()
        idx_bal=[]
        for lbl in vc.index: idx_bal.extend(np.random.choice(np.where(y==lbl)[0],minn,replace=False))
        X_b,y_b=X[idx_bal],y[idx_bal]
        X_tr,X_te,y_tr,y_te=train_test_split(X_b,y_b,test_size=0.2,stratify=y_b,random_state=42)
        pipe=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),max_features=15000,sublinear_tf=True,min_df=2)),
                       ("clf",LogisticRegression(max_iter=1000,class_weight="balanced",C=1.0,solver="lbfgs"))])
        pipe.fit(X_tr,y_tr); y_pr=pipe.predict(X_te)
        p,r,f,_=precision_recall_fscore_support(y_te,y_pr,average="macro",zero_division=0)
        clf_results[asp]={"pipe":pipe,"P":p,"R":r,"F1":f,"acc":accuracy_score(y_te,y_pr),"n_test":len(y_te)}
        all_true.extend(y_te); all_pred.extend(y_pr)
        log.info(f"  {asp:<25} P={p:.3f} R={r:.3f} F1={f:.3f}")
    p_ov,r_ov,f_ov,_=precision_recall_fscore_support(all_true,all_pred,average="macro",zero_division=0)
    acc_ov=accuracy_score(all_true,all_pred)
    p_lx,r_lx,f_lx,_=precision_recall_fscore_support(labelled["silver_label"],labelled["pred_lexicon"],average="macro",zero_division=0)
    log.info(f"  OVERALL ASC  F1={f_ov:.3f} | Lexicon baseline F1={f_lx:.3f}")
    def predict_sent(row):
        asp=row["aspect"]
        return clf_results[asp]["pipe"].predict([row["context"]])[0] if asp in clf_results else lexicon_score(row["context"])
    bio_df["pred_ml"]=bio_df.apply(predict_sent,axis=1)
    asp_ev=[a for a in ASPECTS if a in clf_results]; x=np.arange(len(asp_ev)); w=0.25
    fig,ax=plt.subplots(figsize=(11,5))
    ax.bar(x-w,[clf_results[a]["P"] for a in asp_ev],w,label="Precision",color="#4C72B0",edgecolor="k")
    ax.bar(x,  [clf_results[a]["R"] for a in asp_ev],w,label="Recall",color="#DD8452",edgecolor="k")
    ax.bar(x+w,[clf_results[a]["F1"]for a in asp_ev],w,label="F1-macro",color="#55A868",edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(asp_ev,rotation=15,ha="right"); ax.set_ylim(0,1.05)
    ax.set_title("Per-Aspect ASC Evaluation",fontweight="bold")
    ax.axhline(f_lx,ls="--",color="red",lw=1.2,label=f"Lexicon F1={f_lx:.2f}")
    ax.axhline(f_ov,ls="--",color="black",lw=1.2,label=f"Overall F1={f_ov:.2f}")
    ax.legend(fontsize=9); save_fig(f"{OUTPUTS_DIR}/per_aspect_evaluation.png")
    labs=["negative","neutral","positive"]
    fig,ax=plt.subplots(figsize=(6,5))
    cm=confusion_matrix(all_true,all_pred,labels=labs)
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax,xticklabels=labs,yticklabels=labs)
    ax.set_title("ASC Confusion Matrix",fontweight="bold")
    save_fig(f"{OUTPUTS_DIR}/asc_confusion_matrix.png")
    metrics={"p_ov":p_ov,"r_ov":r_ov,"f_ov":f_ov,"acc_ov":acc_ov,"p_lx":p_lx,"r_lx":r_lx,"f_lx":f_lx,"clf_results":clf_results,"asp_ev":asp_ev}
    return bio_df,metrics

def run_alignment(df,bio_df):
    banner("§6  RATING–ASPECT ALIGNMENT")
    corr_data={c:df[["Overall_Rating",c]].dropna()["Overall_Rating"].astype(float).corr(df[c].astype(float))
               for c in SUB_COLS if c in df.columns and df[c].notna().sum()>50}
    corr_s=pd.Series(corr_data).sort_values(ascending=False)
    if corr_s.empty:
        log.warning("No sub-rating correlations — sub-ratings may not be present in this CSV.")
        return corr_s
    fig,ax=plt.subplots(figsize=(9,5))
    col_bars=["#2ca02c" if v>0.55 else "#ff7f0e" if v>0.40 else "#d62728" for v in corr_s.values]
    bars=ax.barh(corr_s.index,corr_s.values,color=col_bars,edgecolor="k")
    ax.set_xlim(0,0.95); ax.set_xlabel("Pearson r with Overall Rating")
    ax.set_title("Which Aspects Drive Overall Satisfaction?",fontweight="bold")
    for b,v in zip(bars,corr_s.values):
        ax.text(b.get_width()+0.01,b.get_y()+b.get_height()/2,f"{v:.3f}",va="center",fontsize=10)
    save_fig(f"{OUTPUTS_DIR}/subrating_correlation.png")
    sub_labels=[c.replace(" & ","\n& ").replace(" Service","\nService") for c in SUB_COLS]
    pos_means=df[df["Overall_Rating"]>=7][SUB_COLS].mean()
    neg_means=df[df["Overall_Rating"]<=3][SUB_COLS].mean()
    N=len(SUB_COLS); angles=[n/float(N)*2*np.pi for n in range(N)]+[0]
    p_v=pos_means.fillna(0).tolist()+[pos_means.fillna(0).iloc[0]]
    n_v=neg_means.fillna(0).tolist()+[neg_means.fillna(0).iloc[0]]
    fig,ax=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))
    ax.plot(angles,p_v,"o-",lw=2,color="#2ca02c",label="Positive (≥7)")
    ax.fill(angles,p_v,alpha=0.2,color="#2ca02c")
    ax.plot(angles,n_v,"o-",lw=2,color="#d62728",label="Negative (≤3)")
    ax.fill(angles,n_v,alpha=0.2,color="#d62728")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(sub_labels,fontsize=9)
    ax.set_ylim(0,5); ax.set_title("Sub-Rating Radar",fontweight="bold",pad=20)
    ax.legend(loc="upper right",bbox_to_anchor=(1.3,1.1))
    save_fig(f"{OUTPUTS_DIR}/radar_chart.png")
    log.info(f"Strongest driver: {corr_s.idxmax()} (r={corr_s.max():.3f})")
    return corr_s
