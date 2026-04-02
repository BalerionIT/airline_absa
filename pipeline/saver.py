"""pipeline/saver.py — Save outputs and print summary."""
import logging,pandas as pd
from config import SUB_COLS,OUTPUTS_DIR
from pipeline.utils import banner
log=logging.getLogger(__name__)
def save_all(df,bio_df,aope_df,df_fake,metrics,corr_s):
    banner("SAVING ALL OUTPUTS")
    out_cols=(["Airline Name","Overall_Rating","Seat Type","Type Of Traveller","text","n_aspects","sentiment","lex_sentiment","disc_flag","haul_type","has_connection"]+SUB_COLS+["Recommended_bin"])
    out_df=df[[c for c in out_cols if c in df.columns]].copy()
    out_df.to_csv(f"{OUTPUTS_DIR}/annotated_reviews_full.csv",index=False)
    log.info(f"  annotated_reviews_full.csv    ({len(out_df):,} rows)")
    bio_df[["review_idx","aspect","aspect_tokens","context","silver_label","pred_lexicon","pred_ml"]].to_csv(f"{OUTPUTS_DIR}/aspect_span_predictions.csv",index=False)
    log.info(f"  aspect_span_predictions.csv   ({len(bio_df):,} spans)")
    aope_df[["review_idx","aspect","aspect_token","opinion_word","polarity","intensity","negated","sentiment","overall_rating"]].to_csv(f"{OUTPUTS_DIR}/aope_pairs.csv",index=False)
    log.info(f"  aope_pairs.csv                ({len(aope_df):,} pairs)")
    flagged=(df_fake[df_fake["flagged_fake"]==1][["Airline Name","Overall_Rating","Seat Type","word_count","verified_num","suspicion_score","ttr","title_clean","text"]].sort_values("suspicion_score",ascending=False))
    flagged.to_csv(f"{OUTPUTS_DIR}/flagged_fake_reviews.csv",index=False)
    log.info(f"  flagged_fake_reviews.csv      ({len(flagged):,} flagged)")
    eval_rows=[{"Model":"Lexicon Baseline","P":metrics.get("p_lx"),"R":metrics.get("r_lx"),"F1":metrics.get("f_lx"),"Acc":None},{"Model":"TF-IDF + LogReg ASC","P":metrics.get("p_ov"),"R":metrics.get("r_ov"),"F1":metrics.get("f_ov"),"Acc":metrics.get("acc_ov")}]
    for asp in metrics.get("asp_ev",[]):
        r=metrics["clf_results"][asp]; eval_rows.append({"Model":f"  → {asp}","P":r["P"],"R":r["R"],"F1":r["F1"],"Acc":r["acc"]})
    pd.DataFrame(eval_rows).to_csv(f"{OUTPUTS_DIR}/evaluation_summary.csv",index=False)
    if corr_s is not None and not corr_s.empty: corr_s.to_csv(f"{OUTPUTS_DIR}/subrating_correlation.csv",header=["Pearson_r"])
    log.info("  evaluation_summary.csv + subrating_correlation.csv")
def print_summary(df,bio_df,aope_df,df_fake,metrics,corr_s):
    banner("PIPELINE COMPLETE — SUMMARY")
    disc_count=df["disc_flag"].sum(); multi_asp=(df["n_aspects"]>1).sum()
    driver=corr_s.idxmax() if (corr_s is not None and not corr_s.empty) else "N/A"
    r_val=f"{corr_s.max():.3f}" if (corr_s is not None and not corr_s.empty) else "N/A"
    log.info(f"""
  Dataset             : {len(df):,} reviews | {df["Airline Name"].nunique()} airlines
  BIO spans           : {len(bio_df):,} | {bio_df["review_idx"].nunique():,} reviews covered
  AOPE pairs          : {len(aope_df):,} | negation rate {aope_df["negated"].mean()*100:.1f}%
  Multi-aspect        : {multi_asp:,} reviews ({multi_asp/len(df)*100:.1f}%)
  ASC lexicon F1      : {metrics.get("f_lx",0):.3f}
  ASC TF-IDF+LR F1    : {metrics.get("f_ov",0):.3f}  (Δ={metrics.get("f_ov",0)-metrics.get("f_lx",0):+.3f})
  Satisfaction driver : {driver} (r={r_val})
  Discrepancies       : {disc_count:,} ({disc_count/len(df)*100:.1f}%)
  Fake flagged        : {df_fake["flagged_fake"].sum():,} ({df_fake["flagged_fake"].mean()*100:.1f}%)
  All outputs → {OUTPUTS_DIR}/
""")
