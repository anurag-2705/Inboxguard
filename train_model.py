"""Train the InboxGuard text model and write model.pkl + metrics.json.

Usage:
    python build_dataset.py --fetch        # one-off: build data/emails.csv from public corpora
    python train_model.py                  # train on data/emails.csv
    python train_model.py path/to/file.csv # or on a CSV with text + label columns (e.g. the Kaggle set)
    python train_model.py --demo           # tiny synthetic model for smoke tests ONLY (never commit it)

The model is TF-IDF + Logistic Regression, then probability-calibrated on a separate validation
split so that "80" really means "about 80% of emails like this were phishing".
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from preprocess import ML_EXTRA_STOPWORDS, clean_text, looks_english, ml_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "emails.csv")
STRESS_PATH = os.path.join(BASE_DIR, "data", "stress_modern_legit.csv")
SEED = 42
MODERN_WEIGHT = 3.0   # each modern-legit example counts this many times (the set is tiny)

TEXT_COLS = ["text_combined", "text", "body", "email_text", "email text", "message", "content", "email"]
LABEL_COLS = ["label", "class", "target", "email_type", "email type", "type", "category", "label_num"]


def to_binary(v):
    """Map many label styles to 1 (phishing) or 0 (legitimate); None if unknown."""
    s = str(v).strip().lower()
    if s in {"1", "1.0", "true"}:
        return 1
    if s in {"0", "0.0", "false"}:
        return 0
    if any(k in s for k in ("phish", "unsafe", "fraud", "malicious")):
        return 1
    if any(k in s for k in ("safe", "ham", "legit", "benign")):
        return 0
    return None   # "spam" is deliberately NOT mapped: spam and phishing are different problems


def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = next((cols[c] for c in TEXT_COLS if c in cols), None)
    label_col = next((cols[c] for c in LABEL_COLS if c in cols), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns in {path}. Found: {list(df.columns)}")
    out = pd.DataFrame({"text": df[text_col].fillna("").astype(str),
                        "label": df[label_col].map(to_binary)})
    out["source"] = df["source"] if "source" in df.columns else os.path.basename(path)
    out["raw"] = df["raw"].fillna("").astype(str) if "raw" in df.columns else df[text_col].fillna("").astype(str)
    out = out.dropna(subset=["label"])
    out = out[out.text.map(looks_english)]                      # English only (see preprocess.py)
    out["text"] = out.text.map(ml_text)
    out = out[out.text.str.len() >= 40].drop_duplicates(subset="text")
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


def load_modern_legit():
    """Small set of modern legitimate emails (receipts, bank alerts, newsletters, campus and work mail).

    The public phishing corpora are modern but the public ham corpora are ~20 years old, so without
    this the model treats every modern transactional email as phishing. Emails are split BY TEMPLATE
    (same subject line) so the hold-out half measures generalisation to unseen kinds of email.
    Returns (train_rows, holdout_rows) or (None, None) if the file is absent.
    """
    if not os.path.exists(STRESS_PATH):
        return None, None
    d = pd.read_csv(STRESS_PATH, encoding_errors="ignore")
    d["title"] = d["text"].fillna("").str.split("\n").str[0].str.strip()
    d["text"] = d["text"].fillna("").astype(str).map(ml_text)
    d = d[d.text.str.len() >= 40]
    titles = sorted(d.title.unique())
    train_titles = set(titles[::2])
    d["label"], d["source"] = 0, "modern_legit"
    return d[d.title.isin(train_titles)][["text", "label", "source"]], d[~d.title.isin(train_titles)][["text", "label", "source"]]


def demo_frame():
    """Tiny synthetic set so the app can start in a fresh checkout. Metrics from it mean nothing."""
    ph = ["Your account is suspended. Verify your password immediately at urltoken or it will be closed.",
          "Congratulations winner! Claim your gift card now: urltoken",
          "Payment failed. Update your billing information within 24 hours: urltoken"] * 40
    lg = ["Hi team, the weekly sync is tomorrow at 10am. Agenda is in the shared doc.",
          "Thanks for your order. Your receipt is attached and items ship in 3 business days.",
          "Lunch on Friday? Let me know and I will book a table near the office."] * 40
    return pd.DataFrame({"text": ph + lg, "label": [1] * len(ph) + [0] * len(lg), "source": "synthetic demo"})


def fpr_tpr(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"true_neg": int(tn), "false_pos": int(fp), "false_neg": int(fn), "true_pos": int(tp)}


def train(data_path=None, demo=False, verbose=False):
    if demo:
        df, source = demo_frame(), "synthetic demo data (NOT a real evaluation)"
    else:
        path = data_path or DEFAULT_DATA
        if not os.path.exists(path):
            sys.exit(f"No dataset at {path}. Run `python build_dataset.py --fetch` first, "
                     "pass a CSV path, or use --demo for a smoke-test model.")
        df = load_csv(path)
        source = ", ".join(sorted(map(str, df.source.unique()))) + " (English only)"

    strat = df.source.astype(str) + "_" + df.label.astype(str)
    rest, test = train_test_split(df, test_size=0.20, random_state=SEED, stratify=strat)
    train_df, val = train_test_split(rest, test_size=0.125, random_state=SEED,
                                     stratify=rest.source.astype(str) + "_" + rest.label.astype(str))

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_df=0.6, max_features=60000,
                          sublinear_tf=True, strip_accents="unicode",
                          stop_words=sorted(ENGLISH_STOP_WORDS | set(ML_EXTRA_STOPWORDS)))
    lr = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")
    modern_train, modern_hold = (None, None) if demo else load_modern_legit()
    weights = np.ones(len(train_df))
    if modern_train is not None:
        train_df = pd.concat([train_df, modern_train.assign(raw="")], ignore_index=True)
        weights = np.concatenate([weights, np.full(len(modern_train), MODERN_WEIGHT)])
    pipe = make_pipeline(vec, lr).fit(train_df.text, train_df.label, logisticregression__sample_weight=weights)
    model = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid").fit(val.text, val.label)

    proba = model.predict_proba(test.text)[:, 1]
    pred = (proba >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(test.label, pred, average="binary", zero_division=0)
    conf = fpr_tpr(test.label, pred)
    per_source = {}
    for src, g in test.assign(pred=pred).groupby("source"):
        per_source[str(src)] = {"n": int(len(g)), "accuracy": round(float((g.pred == g.label).mean()), 4)}

    if not demo:     # raw test emails (links intact) for evaluate.py's end-to-end check
        test.assign(proba=proba).to_csv(os.path.join(BASE_DIR, "data", "test_split.csv"), index=False)
    metrics = {
        "source": source,
        "train_size": int(len(train_df)), "validation_size": int(len(val)), "test_size": int(len(test)),
        "class_balance_test": {"phishing": int(test.label.sum()), "legitimate": int((test.label == 0).sum())},
        "accuracy": round(accuracy_score(test.label, pred), 4),
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        "roc_auc": round(roc_auc_score(test.label, proba), 4) if test.label.nunique() > 1 else None,
        "pr_auc": round(average_precision_score(test.label, proba), 4) if test.label.nunique() > 1 else None,
        "false_positive_rate": round(conf["false_pos"] / max(conf["false_pos"] + conf["true_neg"], 1), 4),
        "confusion": conf,
        "per_source": per_source,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "caveat": ("Random split inside one data collection. Public corpora are older than today's "
                   "phishing, so treat these numbers as an upper bound."),
    }
    if modern_hold is not None:
        sp = model.predict_proba(modern_hold.text)[:, 1]
        metrics["modern_legit_holdout"] = {
            "n": int(len(modern_hold)), "train_n": int(len(modern_train)),
            "flagged_as_phishing": int((sp >= 0.5).sum()), "false_positive_rate": round(float((sp >= 0.5).mean()), 4),
            "note": "synthetic templates, split by template; small sample, so treat as indicative only"}
        holdout_path = os.path.join(BASE_DIR, "data", "modern_holdout.csv")
        modern_hold.to_csv(holdout_path, index=False)

    names = np.array(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    metrics["top_phishing_terms"] = names[order[-15:][::-1]].tolist()
    metrics["top_legitimate_terms"] = names[order[:15]].tolist()

    joblib.dump({"pipeline": model, "vectorizer": vec, "lr": lr, "demo": bool(demo)}, MODEL_PATH, compress=3)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    if verbose:
        print(json.dumps(metrics, indent=2))
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="CSV with text + label columns (default: data/emails.csv)")
    ap.add_argument("--demo", action="store_true", help="train a tiny synthetic smoke-test model")
    a = ap.parse_args()
    train(a.csv, demo=a.demo, verbose=True)
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")
