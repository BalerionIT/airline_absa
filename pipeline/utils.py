"""pipeline/utils.py — Shared helpers."""
import re, logging
import numpy as np
import matplotlib.pyplot as plt
from config import POS_LEX, NEG_LEX, NEG_WORDS

log = logging.getLogger(__name__)

def banner(msg):
    log.info("\n"+"═"*65+f"\n{msg}\n"+"═"*65)

def clean_text(t):
    return re.sub(r"\s+"," ",re.sub(r"[^\x00-\x7F]+"," ",str(t))).strip()

def rating_to_label(r):
    r=int(r); return "negative" if r<=3 else "neutral" if r<=6 else "positive"

def sr_to_label(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return None
    v=float(v); return "negative" if v<=2.0 else "neutral" if v<=3.0 else "positive"

def lexicon_score(text):
    toks=re.findall(r"\b\w[\w']*\b",text.lower())
    score,negate=0,False
    for tok in toks:
        if tok in NEG_WORDS: negate=True; continue
        if tok in POS_LEX: score+=(-1 if negate else 1)
        elif tok in NEG_LEX: score+=(1 if negate else -1)
        negate=False
    return "positive" if score>0 else "negative" if score<0 else "neutral"

def save_fig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path,dpi=dpi,bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")
