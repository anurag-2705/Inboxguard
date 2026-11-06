# InboxGuard

Paste an email or a link and get a risk level (**Low risk / Warning / Critical**) with the reasons behind it.
You can also upload a `.eml` file. Links are analysed as text and never opened, and nothing you paste is stored or logged.

## How it scores

| Signal | What it looks at |
|---|---|
| **Message model** | TF-IDF + Logistic Regression, probability-calibrated, trained on real phishing and legitimate email |
| **Links & domains** | look-alike and typosquatted brands (`paypa1`, `paypall`, `arnazon`), IP hosts, punycode, shorteners, suspicious TLDs, hidden redirect targets, link text that differs from the real destination, obfuscated links (`hxxp`, `[.]`) |
| **Headers** (when present) | SPF/DKIM/DMARC results (topmost header only), brand name in the sender name vs. the real domain, free-mail impersonation, Reply-To / Return-Path mismatch |
| **Attachments** (`.eml`) | executables, double extensions (`invoice.pdf.exe`), HTML/archive/macro files |
| **Keywords** | urgency, credential requests, threats, rewards. Weak evidence: capped, can never produce a verdict alone |

`score = 0.5 * model + 0.5 * rules` (rules alone override the blend when they reach 60). Warning starts at 45, Critical at 65.
Any single **high-severity** finding (look-alike or TLD-swapped brand domain, raw/obfuscated IP host, failed SPF/DKIM/DMARC, spoofed sender, executable attachment) raises the result to at least **Warning**, so a confirmed red flag never reads "Low risk".
The constants are at the top of `analyzer.py`. The model's word-level reasons are shown in the UI.

## Honest evaluation

Run `python evaluate.py` to reproduce. Numbers from the committed model:

**Message model alone** (random 20% test split, 3,503 English emails; `metrics.json`)

| Accuracy | Precision | Recall | F1 | ROC-AUC | False-positive rate |
|---|---|---|---|---|---|
| 98.7% | 97.7% | 96.7% | 97.2% | 0.999 | 0.7% |

**Whole system** (ML + rules, 1,200 sampled test emails plus modern legitimate emails from unseen templates)

| | Result |
|---|---|
| Phishing flagged Warning or Critical | 87.2% |
| Phishing flagged Critical | 40.3% |
| Enron legitimate mail flagged Warning / Critical | 0.2% / 0.0% |
| Modern legitimate mail (unseen templates, n=77) flagged Warning / Critical | 10.4% / 0.0% |

### Limitations: please read
- **Training data is old.** The public legitimate-email corpora are about 20 years old (Enron). A first model trained on them flagged 69% of modern legitimate mail (bank alerts, receipts, newsletters) as phishing. Removing shortcuts (years, dataset tags, non-English text) and adding a small set of modern legitimate emails cut the model's own error on unseen modern templates to 32%. Rules, score thresholds and trusting official-domain links lower the system's false alarms further, but **Warning means "check this", not "this is phishing"**.
- **The modern legitimate set is synthetic and small** (150 emails, 60 templates, from PhishLens, MIT). Treat that row as indicative only.
- **About 60% of phishing is only a Warning**, not Critical. Phishing without links, or with clean links, relies on the model alone.
- The random-split numbers are an upper bound. They do not cover new campaigns, AI-written phishing, QR-code phishing, or non-English mail (English only).
- Pasted headers can be forged, so header results are only as trustworthy as the source.
- This is a decision aid, not a guarantee. Do not act on a message just because it scored low.

## Tests and CI

```bash
pip install -r requirements-dev.txt
python -m pytest            # ~80 tests: domains, IPs, headers, attachments, scoring, HTTP API, hostile-input timing
```
GitHub Actions (`.github/workflows/ci.yml`) runs the suite and a Docker build + smoke test on every push.
After changing rules or thresholds, re-run `python evaluate.py` and update the numbers above.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:5000
```

Windows one-click: `run_windows.bat` (uses waitress). Linux/cloud: `gunicorn app:app` (see `Procfile`).

### Retrain
```bash
pip install -r requirements-train.txt
python build_dataset.py --fetch    # clones public datasets into data/raw (phishing_pot is ~2.5 GB), writes data/emails.csv
python train_model.py              # writes model.pkl + metrics.json
python evaluate.py                 # end-to-end check; use it before changing weights/thresholds
```
`python train_model.py your.csv` trains on any CSV with text and label columns (for example the Kaggle phishing set).
`--demo` makes a tiny synthetic smoke-test model; the app ignores a demo model and a test fails if one is committed.
Check each dataset's licence before redistributing it.

### Docker / Render
```bash
docker build -t inboxguard . && docker run -p 8000:8000 inboxguard
```
Behind a reverse proxy set `TRUSTED_PROXIES=1` (already set in `render.yaml`) so the rate limiter sees real client IPs.
Leave it at `0` when clients connect directly.

## API
- `POST /analyze` with JSON `{"text": "..."}` or multipart field `file` (`.eml`, max 1 MB). Text limit 20,000 characters. 30 requests/min per IP.
- `GET /api/model-info`, `GET /health`

## Project layout
```
app.py            Flask app: validation, rate limit, security headers
analyzer.py       scoring engine (rules + model)
preprocess.py     text cleaning shared by training and inference
train_model.py    training, calibration, metrics
build_dataset.py  builds data/emails.csv from public corpora
evaluate.py       end-to-end evaluation / threshold tuning
templates/        dashboard
tests/            pytest suite (run: python -m pytest)
```

## Security notes
Input is bounded everywhere: pasted text 20,000 chars, uploads 1 MB, and at most 50,000 characters of any message are scanned.
Regexes that touch untrusted text are length-bounded and anchored (a 1 MB upload of `a.b.a.b.…` once took minutes; it now takes about 0.1 s, and a test guards this).
Malformed URLs (stray `[`/`]`) cannot crash the analysis.
Per-request CSP with nonce, no inline user content (everything is rendered with `textContent`), bounded input sizes,
non-root container, JSON error handlers, bounded rate-limiter memory, pinned dependencies.
