"""
pipeline/module18_bert.py — Fine-tuned DistilBERT for Aspect Sentiment Classification

Architecture
------------
Input format (Sun et al., 2019 — "Utilising BERT for Aspect-Based Sentiment Analysis"):

    [CLS] review context (max 100 tokens) [SEP] aspect category [SEP]

This auxiliary-sentence approach encodes both the aspect identity and its textual
context in a single forward pass, allowing the model to attend jointly to aspect
and opinion.

One DistilBERT classifier is fine-tuned per aspect, matching the TF-IDF setup for
a fair head-to-head comparison.

Model: distilbert-base-uncased
  - 66 MB download (much smaller than BERT-base 440 MB)
  - 97% of BERT-base performance on GLUE
  - Runs on CPU in 2-4 hours for this dataset size

Training details
----------------
- Same time-based split as Module 5/8: train on date_flown < 2023-02, test after
- Silver labels from sub-ratings (same generation logic as Module 5)
- Max sequence length: 128 tokens
- Batch size: 16 (CPU-safe; increase to 32 if GPU available)
- Epochs: 3 (standard for BERT fine-tuning)
- Optimiser: AdamW, lr=2e-5, weight_decay=0.01
- Scheduler: linear warmup (10% of steps) + linear decay

Checkpoint behaviour
--------------------
If a checkpoint already exists in outputs/bert_checkpoints/<aspect>/,
the module loads it instead of retraining. Delete the directory to force retraining.

Outputs
-------
  bert_asc_comparison.png     — Head-to-head F1 bar chart: Lexicon vs TF-IDF vs DistilBERT
  bert_per_aspect.png         — Per-aspect F1 comparison with error bars
  bert_confusion_<aspect>.png — Confusion matrix per aspect
  bert_results.csv            — Full metrics table
"""
import os
import sys
import json
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, accuracy_score)
from sklearn.utils import resample

from config import OUTPUTS_DIR
from pipeline.utils import banner, save_fig

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.1
TRAIN_CUTOFF = "2023-02-01"
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, "bert_checkpoints")
MIN_PER_CLASS  = 50    # minimum samples per class for training

ASPECTS = [
    "Punctuality",
    "Cabin Comfort",
    "Cabin Crew Service",
    "Food & Beverages",
    "Value For Money",
]

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# TF-IDF baseline numbers from Module 5 (for comparison chart)
TFIDF_F1 = {
    "Punctuality":        0.618,
    "Cabin Comfort":      0.606,
    "Cabin Crew Service": 0.602,
    "Food & Beverages":   0.617,
    "Value For Money":    0.597,
    "overall":            0.607,
}
LEXICON_F1 = {
    "Punctuality":        0.462,
    "Cabin Comfort":      0.462,
    "Cabin Crew Service": 0.462,
    "Food & Beverages":   0.462,
    "Value For Money":    0.462,
    "overall":            0.462,
}


# ── Dependency check ──────────────────────────────────────────────────────────
def _check_deps():
    try:
        import torch
        from transformers import (DistilBertTokenizerFast,
                                   DistilBertForSequenceClassification,
                                   get_linear_schedule_with_warmup)
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        return True
    except ImportError:
        return False


# ── Silver label generation ───────────────────────────────────────────────────
def _silver_label(rating, sub_rating=None):
    """Convert numeric rating to sentiment class."""
    r = sub_rating if (sub_rating is not None and not np.isnan(sub_rating)) else rating
    if r <= 2.0:  return "negative"
    if r <= 3.0:  return "neutral"
    return "positive"


ASPECT_TO_SUBCOL = {
    "Punctuality":        None,            # no direct sub-rating; use overall
    "Cabin Comfort":      "Seat Comfort",
    "Cabin Crew Service": "Cabin Staff Service",
    "Food & Beverages":   "Food & Beverages",
    "Value For Money":    "Value For Money",
}


_TRIGGERS = {
    "Punctuality": [
        "delay", "delayed", "delays", "late", "on time", "ontime", "punctual",
        "punctuality", "cancelled", "cancel", "cancellation", "missed",
        "connection", "departure", "tarmac", "ground stop", "diverted",
    ],
    "Cabin Comfort": [
        "seat", "seats", "legroom", "leg room", "comfortable", "uncomfortable",
        "cabin", "space", "recline", "overhead", "storage", "narrow", "cramped",
        "headrest", "cushion", "pillow", "blanket", "aisle", "window",
    ],
    "Cabin Crew Service": [
        "crew", "staff", "flight attendant", "steward", "stewardess",
        "attendant", "service", "friendly", "rude", "helpful", "professional",
        "polite", "attentive", "ignored", "attitude", "courteous",
    ],
    "Food & Beverages": [
        "food", "meal", "meals", "snack", "drink", "drinks", "beverage",
        "catering", "menu", "dining", "breakfast", "lunch", "dinner",
        "wine", "water", "coffee", "tea", "buy on board",
    ],
    "Value For Money": [
        "value", "worth", "price", "priced", "expensive", "cheap", "cost",
        "fare", "ticket", "fee", "fees", "overpriced", "affordable",
        "money", "pay", "paid", "charge", "charged",
    ],
}


def _prepare_data(df, aspect):
    """
    Build a DataFrame of (context, aspect_name, label, date_flown) rows
    from the review corpus using a lightweight trigger scan.
    """
    triggers = _TRIGGERS.get(aspect, [])
    if not triggers:
        log.warning(f"  No triggers defined for {aspect} — skipping.")
        return pd.DataFrame()

    import re
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in triggers) + r")\b", re.I)

    sub_col = ASPECT_TO_SUBCOL.get(aspect)
    rows = []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        if not pattern.search(text):
            continue
        # Extract context: window of 120 chars around first trigger match
        m = pattern.search(text)
        start = max(0, m.start() - 60)
        end   = min(len(text), m.end() + 60)
        context = text[start:end].strip()

        sub_r = row.get(sub_col, np.nan) if sub_col else np.nan
        label = _silver_label(row["Overall_Rating"], sub_r)

        rows.append({
            "context":    context,
            "aspect":     aspect,
            "label":      label,
            "label_id":   LABEL2ID[label],
            "date_flown": row.get("date_flown"),
        })
    return pd.DataFrame(rows)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
def _make_dataset_class():
    import torch
    from torch.utils.data import Dataset

    class AspectDataset(Dataset):
        def __init__(self, contexts, aspects, labels, tokenizer, max_len):
            self.enc = tokenizer(
                contexts,
                aspects,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels":         self.labels[idx],
            }

    return AspectDataset


# ── Training loop ─────────────────────────────────────────────────────────────
def _train_aspect(aspect, df, tokenizer, checkpoint_path):
    import torch
    from transformers import (DistilBertForSequenceClassification,
                               get_linear_schedule_with_warmup)
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    AspectDataset = _make_dataset_class()

    # Prepare data
    aspect_df = _prepare_data(df, aspect)
    if aspect_df.empty or len(aspect_df) < MIN_PER_CLASS * 3:
        log.warning(f"  {aspect}: insufficient data ({len(aspect_df)} rows) — skipping.")
        return None

    # Time-based split
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    train_df = aspect_df[aspect_df["date_flown"] < cutoff].copy()
    test_df  = aspect_df[aspect_df["date_flown"] >= cutoff].copy()

    if len(train_df) < MIN_PER_CLASS * 3 or len(test_df) < 30:
        log.warning(f"  {aspect}: split too small "
                    f"(train={len(train_df)}, test={len(test_df)}) — skipping.")
        return None

    # Balance training set via undersampling
    min_class = train_df["label"].value_counts().min()
    min_class = max(min_class, MIN_PER_CLASS)
    balanced  = pd.concat([
        resample(grp, n_samples=min_class, random_state=42)
        for _, grp in train_df.groupby("label")
    ])

    log.info(f"  {aspect}: train={len(balanced):,} balanced, test={len(test_df):,}")

    train_ds = AspectDataset(
        balanced["context"].tolist(), balanced["aspect"].tolist(),
        balanced["label_id"].tolist(), tokenizer, MAX_LEN)
    test_ds  = AspectDataset(
        test_df["context"].tolist(), test_df["aspect"].tolist(),
        test_df["label_id"].tolist(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3,
        id2label=ID2LABEL, label2id=LABEL2ID)
    model.to(device)

    # Optimiser + scheduler
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimiser = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)

    # Training
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            optimiser.zero_grad()
            out  = model(input_ids=batch["input_ids"].to(device),
                         attention_mask=batch["attention_mask"].to(device),
                         labels=batch["labels"].to(device))
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        log.info(f"    {aspect} epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}")

    # Save checkpoint
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    log.info(f"    Saved checkpoint → {checkpoint_path}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds = torch.argmax(out.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["negative", "neutral", "positive"],
        zero_division=0, output_dict=True)

    log.info(f"    {aspect} — F1={f1:.3f}  Acc={acc:.3f}")

    return {
        "aspect":     aspect,
        "f1":         f1,
        "accuracy":   acc,
        "report":     report,
        "preds":      all_preds,
        "labels":     all_labels,
        "n_train":    len(balanced),
        "n_test":     len(test_df),
    }


# ── Load from checkpoint ──────────────────────────────────────────────────────
def _eval_from_checkpoint(aspect, df, tokenizer, checkpoint_path):
    import torch
    from transformers import DistilBertForSequenceClassification
    from torch.utils.data import DataLoader

    AspectDataset = _make_dataset_class()

    aspect_df = _prepare_data(df, aspect)
    if aspect_df.empty:
        return None

    cutoff   = pd.Timestamp(TRAIN_CUTOFF)
    test_df  = aspect_df[aspect_df["date_flown"] >= cutoff].copy()
    if len(test_df) < 30:
        return None

    test_ds     = AspectDataset(
        test_df["context"].tolist(), test_df["aspect"].tolist(),
        test_df["label_id"].tolist(), tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device); model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out   = model(input_ids=batch["input_ids"].to(device),
                          attention_mask=batch["attention_mask"].to(device))
            preds = torch.argmax(out.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["negative", "neutral", "positive"],
        zero_division=0, output_dict=True)

    log.info(f"  {aspect} (from checkpoint) — F1={f1:.3f}  Acc={acc:.3f}")
    return {
        "aspect": aspect, "f1": f1, "accuracy": acc,
        "report": report, "preds": all_preds, "labels": all_labels,
        "n_train": 0, "n_test": len(test_df),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def _plot_comparison(bert_results):
    aspects       = [r["aspect"] for r in bert_results]
    bert_f1s      = [r["f1"] for r in bert_results]
    tfidf_f1s     = [TFIDF_F1.get(a, 0) for a in aspects]
    lexicon_f1s   = [LEXICON_F1.get(a, 0) for a in aspects]
    overall_bert  = np.mean(bert_f1s)

    # Figure 1: head-to-head overall + per-aspect
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Aspect Sentiment Classification: Model Comparison\n"
                 "Lexicon vs TF-IDF + LogReg vs Fine-tuned DistilBERT",
                 fontsize=13, fontweight="bold")

    # Left: overall F1 comparison
    models     = ["Lexicon\nBaseline", "TF-IDF\n+ LogReg", "DistilBERT\n(fine-tuned)"]
    overall_f1 = [LEXICON_F1["overall"], TFIDF_F1["overall"], overall_bert]
    colors_m   = ["#95A5A6", "#4C72B0", "#E74C3C"]
    bars = axes[0].bar(models, overall_f1, color=colors_m, edgecolor="k",
                       alpha=0.87, width=0.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Macro-averaged F1")
    axes[0].set_title("Overall ASC Performance", fontweight="bold")
    axes[0].axhline(0.5, color="gray", lw=1, ls="--", alpha=0.5,
                    label="Random baseline (3-class)")
    axes[0].legend(fontsize=8)
    for b, v in zip(bars, overall_f1):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.015,
                     f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    # Annotate improvements
    delta1 = TFIDF_F1["overall"]  - LEXICON_F1["overall"]
    delta2 = overall_bert         - TFIDF_F1["overall"]
    axes[0].annotate(f"+{delta1:.3f}", xy=(1, TFIDF_F1["overall"]),
                     xytext=(0.5, TFIDF_F1["overall"] + 0.06),
                     ha="center", fontsize=9, color="#4C72B0",
                     arrowprops=dict(arrowstyle="->", color="#4C72B0"))
    axes[0].annotate(f"+{delta2:.3f}", xy=(2, overall_bert),
                     xytext=(1.5, overall_bert + 0.06),
                     ha="center", fontsize=9, color="#E74C3C",
                     arrowprops=dict(arrowstyle="->", color="#E74C3C"))

    # Right: per-aspect grouped bars
    x     = np.arange(len(aspects))
    width = 0.26
    axes[1].bar(x - width, lexicon_f1s, width, label="Lexicon",
                color="#95A5A6", edgecolor="k", alpha=0.87)
    axes[1].bar(x,         tfidf_f1s,   width, label="TF-IDF + LogReg",
                color="#4C72B0", edgecolor="k", alpha=0.87)
    axes[1].bar(x + width, bert_f1s,    width, label="DistilBERT",
                color="#E74C3C", edgecolor="k", alpha=0.87)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([a.replace(" & ", "\n& ") for a in aspects], fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Per-Aspect F1 Comparison", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(f"{OUTPUTS_DIR}/bert_asc_comparison.png")

    # Figure 2: confusion matrices
    n_asp  = len(bert_results)
    ncols  = min(3, n_asp)
    nrows  = (n_asp + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                               squeeze=False)
    fig2.suptitle("DistilBERT Confusion Matrices by Aspect", fontsize=13,
                  fontweight="bold")
    labels_txt = ["negative", "neutral", "positive"]
    for i, r in enumerate(bert_results):
        ax_i = axes2[i // ncols][i % ncols]
        cm   = confusion_matrix(r["labels"], r["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels_txt, yticklabels=labels_txt,
                    ax=ax_i, cbar=False)
        ax_i.set_title(f"{r['aspect']}\nF1={r['f1']:.3f}", fontweight="bold")
        ax_i.set_xlabel("Predicted"); ax_i.set_ylabel("True")
    for j in range(len(bert_results), nrows * ncols):
        axes2[j // ncols][j % ncols].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(f"{OUTPUTS_DIR}/bert_confusion_matrices.png")


# ── Main entry point ──────────────────────────────────────────────────────────
def run(df):
    banner("MODULE 18 — FINE-TUNED DISTILBERT ASC")

    if not _check_deps():
        log.warning(
            "\n  ╔══════════════════════════════════════════════════════════╗"
            "\n  ║  torch / transformers not installed.                     ║"
            "\n  ║  Run:  pip install torch transformers datasets           ║"
            "\n  ║  Then re-run:  python main.py --no-scrape                ║"
            "\n  ║  First run downloads ~350 MB (DistilBERT weights).       ║"
            "\n  ╚══════════════════════════════════════════════════════════╝"
        )
        return df

    import torch
    from transformers import DistilBertTokenizerFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  Device: {device}")
    log.info(f"  Model:  {MODEL_NAME}")
    log.info(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  MaxLen: {MAX_LEN}")
    log.info(f"  Train cutoff: {TRAIN_CUTOFF}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load tokenizer (downloads once, then caches)
    log.info(f"  Loading tokenizer ({MODEL_NAME})…")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    results = []
    for aspect in ASPECTS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, aspect.replace(" ", "_")
                                                       .replace("&", "and"))
        if os.path.isdir(ckpt_path) and os.path.exists(
                os.path.join(ckpt_path, "config.json")):
            log.info(f"  {aspect}: checkpoint found — loading…")
            r = _eval_from_checkpoint(aspect, df, tokenizer, ckpt_path)
        else:
            log.info(f"  {aspect}: training…")
            r = _train_aspect(aspect, df, tokenizer, ckpt_path)

        if r is not None:
            results.append(r)

    if not results:
        log.warning("  No aspects produced valid results.")
        return df

    # Overall F1
    overall_bert = np.mean([r["f1"] for r in results])
    overall_acc  = np.mean([r["accuracy"] for r in results])

    log.info(f"""
  BERT MODULE SUMMARY
  ─────────────────────────────────────────────────────────────────────
  {'Aspect':<25} {'DistilBERT F1':>14} {'TF-IDF F1':>10} {'Delta':>8}
  ─────────────────────────────────────────────────────────────────────""")
    for r in results:
        delta = r["f1"] - TFIDF_F1.get(r["aspect"], 0)
        sign  = "+" if delta >= 0 else ""
        log.info(f"  {r['aspect']:<25} {r['f1']:>14.3f} "
                 f"{TFIDF_F1.get(r['aspect'],0):>10.3f} {sign}{delta:>7.3f}")
    log.info(f"  {'OVERALL':<25} {overall_bert:>14.3f} "
             f"{TFIDF_F1['overall']:>10.3f} "
             f"{'+' if overall_bert>=TFIDF_F1['overall'] else ''}"
             f"{overall_bert-TFIDF_F1['overall']:>7.3f}")
    log.info(f"  {'Lexicon baseline':<25} {'—':>14} {LEXICON_F1['overall']:>10.3f}")
    log.info("  ─────────────────────────────────────────────────────────────────────")

    # Save metrics CSV
    rows = []
    for r in results:
        rows.append({
            "aspect":         r["aspect"],
            "model":          "DistilBERT",
            "f1_macro":       round(r["f1"], 4),
            "accuracy":       round(r["accuracy"], 4),
            "n_test":         r["n_test"],
            "f1_negative":    round(r["report"]["negative"]["f1-score"], 4),
            "f1_neutral":     round(r["report"]["neutral"]["f1-score"], 4),
            "f1_positive":    round(r["report"]["positive"]["f1-score"], 4),
        })
        rows.append({
            "aspect":         r["aspect"],
            "model":          "TF-IDF+LogReg",
            "f1_macro":       TFIDF_F1.get(r["aspect"], 0),
            "accuracy":       None, "n_test": None,
            "f1_negative":    None, "f1_neutral": None, "f1_positive": None,
        })
    pd.DataFrame(rows).to_csv(f"{OUTPUTS_DIR}/bert_results.csv", index=False)
    log.info(f"  Saved: {OUTPUTS_DIR}/bert_results.csv")

    _plot_comparison(results)

    return df
