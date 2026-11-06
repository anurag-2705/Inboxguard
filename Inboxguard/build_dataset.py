"""Build data/emails.csv from public, real-world email corpora hosted on GitHub.

  phishing (label 1): rf-peixoto/phishing_pot   - real phishing .eml files
  legitimate (label 0): MWiechmann/enron_spam_data - real corporate email (the "ham" half)

Usage:
    python build_dataset.py --fetch          # git clone both repos into data/raw, then build
    python build_dataset.py --phish-dir PATH --enron-zip PATH   # use local copies

Note: phishing_pot is ~2.5 GB because it keeps attachments. Only text is extracted here.
Check each dataset's licence before redistributing it.
The output CSV is git-ignored; commit only model.pkl and metrics.json.
"""
import argparse
import csv
import email
import glob
import multiprocessing
import os
import subprocess
import sys
import time
import zipfile
from email import policy

import pandas as pd

from preprocess import clean_text

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data", "emails.csv")
MIN_CHARS = 40
RAW_KEEP = 8000      # characters of the original text kept in the "raw" column (for end-to-end evaluation)
RAW_LIMIT = 60000   # characters read from a body before cleaning (keeps huge HTML mails fast)
csv.field_size_limit(sys.maxsize)


def read_eml(path):
    """Subject + best text body of an .eml file, or '' if unreadable."""
    try:
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        subject = str(msg.get("Subject") or "")
        body = msg.get_body(preferencelist=("plain", "html"))
        text = body.get_content() if body is not None else ""
        return subject + "\n" + str(text)[:RAW_LIMIT]
    except Exception:
        return ""


def _parse(path):
    raw = read_eml(path)
    return clean_text(raw), raw[:RAW_KEEP]


def load_phishing(phish_dir, budget=None):
    """Parse every .eml. Progress is cached in data/.phish_cache.csv so an interrupted run resumes.
    `budget` (seconds) stops early; re-run the command to continue."""
    files = sorted(glob.glob(os.path.join(phish_dir, "*.eml")))
    cache = os.path.join(os.path.dirname(OUT), ".phish_cache2.csv")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    done = {}
    if os.path.exists(cache):
        old = pd.read_csv(cache, keep_default_na=False)
        done = {f: (t, r) for f, t, r in zip(old["file"], old["text"], old["raw"])}
    todo = [f for f in files if os.path.basename(f) not in done]
    start, batch = time.time(), []
    workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(workers) if workers > 1 else None
    results = pool.imap(_parse, todo, chunksize=50) if pool else map(_parse, todo)
    for i, (f, (text, raw)) in enumerate(zip(todo, results), 1):
        batch.append((os.path.basename(f), text, raw))
        if len(batch) >= 500 or i == len(todo):
            pd.DataFrame(batch, columns=["file", "text", "raw"]).to_csv(cache, mode="a", header=not os.path.exists(cache), index=False)
            done.update({f: (t, r) for f, t, r in batch})
            batch = []
            print(f"  parsed {len(done)}/{len(files)} phishing emails", flush=True)
        if budget and time.time() - start > budget and len(done) < len(files):
            if pool:
                pool.terminate()
            sys.exit(f"Time budget reached ({len(done)}/{len(files)} parsed). Run the same command again to continue.")
    if pool:
        pool.close()
    return [(*done[os.path.basename(f)], 1, "phishing_pot") for f in files]


def load_enron_ham(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f, encoding_errors="ignore")
    df = df[df["Spam/Ham"].str.lower() == "ham"]
    text = df["Subject"].fillna("").astype(str) + "\n" + df["Message"].fillna("").astype(str).str[:RAW_LIMIT]
    return [(clean_text(t), t[:RAW_KEEP], 0, "enron") for t in text]


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, name in (("https://github.com/rf-peixoto/phishing_pot.git", "phishing_pot"),
                      ("https://github.com/MWiechmann/enron_spam_data.git", "enron_spam_data")):
        dest = os.path.join(RAW, name)
        if not os.path.isdir(dest):
            subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
    return os.path.join(RAW, "phishing_pot", "email"), os.path.join(RAW, "enron_spam_data", "enron_spam_data.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--phish-dir")
    ap.add_argument("--enron-zip")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--budget", type=int, help="stop after N seconds; re-run to resume")
    a = ap.parse_args()
    phish_dir, enron_zip = fetch() if a.fetch else (a.phish_dir, a.enron_zip)
    if not (phish_dir and enron_zip):
        ap.error("pass --fetch, or both --phish-dir and --enron-zip")

    print("Parsing phishing emails ...")
    rows = load_phishing(phish_dir, a.budget)
    print("Parsing legitimate emails ...")
    rows += load_enron_ham(enron_zip)

    df = pd.DataFrame(rows, columns=["text", "raw", "label", "source"])
    n0 = len(df)
    df = df[df.text.str.len() >= MIN_CHARS]
    # exact + near duplicates (same first 300 normalised characters)
    key = df.text.str.lower().str.replace(r"\W+", "", regex=True).str[:300]
    df = df[~key.duplicated()].reset_index(drop=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{n0} parsed -> {len(df)} usable after cleaning and de-duplication")
    print(df.groupby(["source", "label"]).size())
    print("Saved", a.out)


if __name__ == "__main__":
    main()
