"""Step 1 - Train the ML model (TF-IDF + Logistic Regression) and save model.pkl + metrics.json.

Usage:
    python train_model.py                      # auto-finds phishing_email.csv / data/*.csv, else synthetic demo data
    python train_model.py path\\to\\file.csv    # train on a specific CSV
"""
import glob
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
random.seed(42)

BRANDS = ["PayPal", "Chase", "Amazon", "Microsoft", "Netflix", "DHL", "Apple", "HDFC Bank"]
BAD_URLS = [
    "http://paypa1-secure-login.xyz/verify", "http://192.168.44.12/login.php",
    "http://bit.ly/3xKq9Zp", "http://account-update.chase-support.top/signin",
    "http://amazon.verify-billing.click/confirm", "http://microsoft-office365.work/auth",
    "http://dhl-parcel-redelivery.info/pay", "http://netflix-billing-update.tk/card",
]
GOOD_URLS = [
    "https://www.paypal.com/myaccount", "https://www.amazon.com/orders",
    "https://github.com/notifications", "https://docs.google.com/document/d/abc",
    "https://www.microsoft.com/account", "https://zoom.us/j/123456789",
    "https://www.linkedin.com/feed", "https://www.dhl.com/track",
]

PHISH = [
    "URGENT: Your {brand} account has been suspended due to unusual activity. Verify your identity immediately at {url} or it will be permanently closed within {h} hours.",
    "Dear customer, we detected unauthorized login attempts. Confirm your password and credit card details now: {url}. Failure to act will result in account restriction.",
    "Congratulations! You are the winner of a {brand} gift card worth ${amt}. Claim your prize now before it expires: {url}",
    "Your package could not be delivered. Pay the small redelivery fee of ${small} immediately: {url}. Final notice before the parcel is returned.",
    "Password reset required. Your {brand} password expires today. Log in now to keep your account active: {url}",
    "Payment failed. Update your billing information within {h} hours to avoid service interruption: {url}",
    "Action required: suspicious sign-in blocked. Verify your account, enter your security code and SSN to unlock it: {url}",
    "You have a pending refund of ${amt}. Provide your bank account number and OTP to receive the payment: {url}",
]
LEGIT = [
    "Hi team, a reminder that our weekly sync is tomorrow at 10am. Agenda is in the shared doc: {url}. Let me know if you cannot make it.",
    "Thanks for your order. Your {brand} receipt is attached and your items will ship within 3 business days. Track the order here: {url}",
    "Here is this month's newsletter with product updates, a new blog post and upcoming webinars. Read more: {url}",
    "Hello, attached is the invoice for last month's consulting work. Payment is due in 30 days, no action is needed on your side today.",
    "Your {brand} password was changed successfully. If this was you, no further action is needed. Manage settings at {url}",
    "Lunch on Friday? A few of us are heading to the new place near the office. Let me know and I will book a table.",
    "Meeting notes from today: we agreed on the Q3 roadmap, design review next week, and the release candidate on Thursday. Notes: {url}",
    "Your {brand} verification code is {code}. Do not share this code. If you did not request it, you can ignore this email.",
]
FILLER = ["", "", "Thank you.", "Regards, Support Team", "Best wishes", "Please do not reply to this email."]


def fill(t, bad):
    return t.format(
        brand=random.choice(BRANDS), h=random.choice([12, 24, 48]), amt=random.randint(50, 900),
        small=round(random.uniform(1, 5), 2), code=random.randint(100000, 999999),
        url=random.choice(BAD_URLS if bad else GOOD_URLS),
    ) + " " + random.choice(FILLER)


def build_synthetic(n=700):
    X, y = [], []
    for _ in range(n):
        X.append(fill(random.choice(PHISH), True)); y.append(1)
        X.append(fill(random.choice(LEGIT), False)); y.append(0)
    return X, y


TEXT_COLS = ["text_combined", "text", "body", "email_text", "email text", "message", "content", "email"]
LABEL_COLS = ["label", "class", "target", "email_type", "email type", "type", "category"]


def to_binary(v):
    """Map many label styles to 1 (phishing/spam) or 0 (legit)."""
    s = str(v).strip().lower()
    if s in {"1", "1.0", "true"}:
        return 1
    if s in {"0", "0.0", "false"}:
        return 0
    if "phish" in s or "spam" in s or "unsafe" in s or "fraud" in s or "malicious" in s:
        return 1
    if "safe" in s or "ham" in s or "legit" in s or "benign" in s:
        return 0
    return None


def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = next((cols[c] for c in TEXT_COLS if c in cols), None)
    label_col = next((cols[c] for c in LABEL_COLS if c in cols), None)
    if label_col is None or (text_col is None and "body" not in cols):
        raise ValueError(f"Could not find text/label columns in {path}. Columns found: {list(df.columns)}")
    if "subject" in cols and text_col is not None and cols["subject"] != text_col:
        text = df[cols["subject"]].fillna("").astype(str) + " " + df[text_col].fillna("").astype(str)
    else:
        text = df[text_col].fillna("").astype(str)
    out = pd.DataFrame({"text": text.str.strip(), "label": df[label_col].map(to_binary)})
    before = len(out)
    out = out[(out.text.str.len() > 15)].dropna().drop_duplicates(subset="text")
    out["label"] = out["label"].astype(int)
    print(f"Loaded {path}: {before} rows -> {len(out)} usable after cleaning "
          f"({out.label.sum()} phishing / {(out.label == 0).sum()} legit)")
    return out.text.tolist(), out.label.tolist()


def build_dataset(csv_path=None):
    path = csv_path or next(iter(glob.glob(os.path.join(BASE_DIR, "phishing_email*.csv")) + glob.glob(os.path.join(BASE_DIR, "data", "*.csv"))), None)
    if path:
        X, y = load_csv(path)
        return X, y, os.path.basename(path)
    print("WARNING: no CSV dataset found - using small SYNTHETIC demo data (accuracy numbers are not meaningful).")
    X, y = build_synthetic()
    return X, y, "synthetic demo data"


def train(csv_path=None, verbose=False):
    X, y, source = build_dataset(csv_path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=50000,
                        sublinear_tf=True, strip_accents="unicode"),
        LogisticRegression(max_iter=1000, C=3.0),
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    metrics = {"source": source, "train_size": len(X_tr), "test_size": len(X_te),
               "accuracy": round(accuracy_score(y_te, pred), 4), "precision": round(p, 4),
               "recall": round(r, 4), "f1": round(f, 4),
               "confusion": {"true_neg": int(tn), "false_pos": int(fp), "false_neg": int(fn), "true_pos": int(tp)},
               "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    json.dump(metrics, open(METRICS_PATH, "w"), indent=2)
    joblib.dump(model, MODEL_PATH, compress=3)
    if verbose:
        print(json.dumps(metrics, indent=2))
    return model


if __name__ == "__main__":
    train(csv_path=sys.argv[1] if len(sys.argv) > 1 else None, verbose=True)
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")
