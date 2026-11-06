"""End-to-end evaluation of the full analyzer (ML + link + keyword + header rules).

    python evaluate.py            # needs data/test_split.csv (written by train_model.py)

Reports, for several ML weights, how often phishing is flagged and how often legitimate mail is
flagged, on (a) the held-out test split and (b) modern legitimate emails from unseen templates.
Use it to choose ML_WEIGHT / WARN_AT / CRITICAL_AT in analyzer.py.
"""
import os
import sys

import pandas as pd

import analyzer
from preprocess import ml_text

BASE = os.path.dirname(os.path.abspath(__file__))


def components(texts):
    rows = []
    for t in texts:
        r = analyzer.analyze(t)
        rows.append((r["ml_score"], r["rule_score"], len(analyzer.clean_text(t).split())))
    return rows


def flagged(rows, w, warn, crit):
    out = []
    for ml, heur, words in rows:
        if ml is None:
            s = heur
        else:
            ww = w if words >= 12 else w / 2 if words >= 4 else 0.0
            s = ww * ml + (1 - ww) * heur
            if heur >= analyzer.STRONG_RULES:
                s = max(s, heur)
        out.append((s >= warn, s >= crit))
    return out


def main(n=1200):
    path = os.path.join(BASE, "data", "test_split.csv")
    if not os.path.exists(path):
        sys.exit("Run train_model.py first (it writes data/test_split.csv).")
    t = pd.read_csv(path)
    t = pd.concat([t[t.label == k].sample(min((t.label == k).sum(), n // 2), random_state=1) for k in (0, 1)])
    print(f"scoring {len(t)} test emails ...", flush=True)
    comp = components(t.raw.fillna("").astype(str))
    ph = [c for c, y in zip(comp, t.label) if y == 1]
    lg = [c for c, y in zip(comp, t.label) if y == 0]

    sp = os.path.join(BASE, "data", "stress_modern_legit.csv")
    mod = []
    if os.path.exists(sp):
        d = pd.read_csv(sp, encoding_errors="ignore")
        d["title"] = d.text.fillna("").str.split("\n").str[0].str.strip()
        hold = d[~d.title.isin(set(sorted(d.title.unique())[::2]))]
        mod = components(hold.text.fillna("").astype(str))

    print(f"{'ML weight':>9} {'warn':>5} {'crit':>5} | phish>=Warn phish=Crit | enron FP>=Warn FP=Crit | modern FP>=Warn FP=Crit")
    for w in (0.0, 0.25, 0.35, 0.5):
        for warn, crit in ((35, 65), (45, 65), (45, 75)):
            a, b, c = flagged(ph, w, warn, crit), flagged(lg, w, warn, crit), flagged(mod, w, warn, crit) if mod else []
            f = lambda L, i: f"{100 * sum(x[i] for x in L) / max(len(L), 1):5.1f}%"
            print(f"{w:>9} {warn:>5} {crit:>5} | {f(a,0):>10} {f(a,1):>10} | {f(b,0):>12} {f(b,1):>8} | {f(c,0):>13} {f(c,1):>8}")


if __name__ == "__main__":
    main()
