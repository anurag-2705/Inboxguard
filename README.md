# InboxGuard

**Check before you click.** InboxGuard is a Flask web app that analyzes a pasted email or link and returns a phishing risk level (Safe / Warning / Critical), a confidence score, and a plain-English list of the reasons behind it.

![InboxGuard dashboard](docs/screenshot.png)
<!-- Add a screenshot of the dashboard at docs/screenshot.png -->

## Features

- **ML detection:** TF-IDF + Logistic Regression (scikit-learn) gives a phishing probability from the message text.
- **Link and domain analysis:** flags look-alike domains (`paypa1-secure-login.xyz`), raw IP hosts, URL shorteners, suspicious TLDs, punycode, `@` tricks, excessive subdomains and plain HTTP.
- **Email header analysis:** paste the raw headers to check SPF/DKIM/DMARC results, spoofed sender display names, and Reply-To / Return-Path mismatches.
- **HTML link mismatch:** catches `<a href="evil.top">https://www.paypal.com</a>` style deception.
- **Keyword detection:** urgency, credential requests, threats and reward/payment language.
- **Explainable results:** every flag is listed with a severity and where it was found.
- **Quick tests:** one-click Bank alert, Password reset, Package delivery, Spoofed headers and Normal email samples.
- **Deployment-ready:** Dockerfile, Procfile, `render.yaml`, health check, rate limiting and security headers.

## How it works

```
 pasted text ──► split headers / body
                    │
      ┌─────────────┼───────────────┬──────────────────┐
      ▼             ▼               ▼                  ▼
  ML model     link & domain     keyword          header checks
 (0-100)         checks           checks         (only if headers
                (rule score)    (rule score)         are pasted)
      └─────────────┴───────┬───────┴──────────────────┘
                            ▼
        score = 0.6 × ML + 0.4 × rules   (rules weigh more for bare links)
        rules ≥ 60 override the blend
                            ▼
        ≥ 65 Critical   |   35-64 Warning   |   < 35 Safe
```

## Quick start (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000
```

Or double-click `run_windows.bat`. For a production-style server on Windows use `python serve.py` (waitress). On Linux/macOS use `gunicorn app:app`.

The first start trains a small **demo model on synthetic data**. For real results, train on a real dataset (next section).

## Train on a real dataset

1. Download **Phishing Email Detection** (Naser Abdullah Alam) from Kaggle and place the CSV in this folder, for example `phishing_email.csv`.
2. Run `python train_model.py` (or `python train_model.py path\to\file.csv`).
3. Review `metrics.json`. It is shown in the dashboard footer.

The loader detects the text and label columns automatically, accepts labels such as `1/0`, `spam/ham` and `Phishing Email/Safe Email`, and removes duplicates and near-empty rows.

| Metric | Value |
|---|---|
| Dataset | _fill in after training_ |
| Accuracy | _..._ |
| Precision / Recall / F1 | _..._ |

Do not commit the dataset itself (it is git-ignored). Commit `model.pkl` and `metrics.json` so deployments do not need it.

## API

`POST /analyze` with JSON `{"text": "<email or link>"}`.

```json
{
  "score": 94, "level": "Critical", "ml_score": 98, "rule_score": 88,
  "summary": "7 warning sign(s) found. The strongest: ...",
  "indicators": [{"severity": "high", "text": "Impersonates 'paypal' ...", "where": "paypa1-secure-login.xyz"}],
  "links": [{"url": "...", "domain": "...", "flags": []}],
  "keywords": {"Urgency": ["urgent", "immediately"]},
  "has_headers": false
}
```

Other routes: `GET /health`, `GET /api/model-info`. Input is limited to 20,000 characters and 30 requests per minute per IP (`RATE_LIMIT` environment variable).

## Project structure

```
inboxguard/
├── app.py              Flask routes, rate limit, security headers
├── analyzer.py         ML + link + keyword + header analysis
├── train_model.py      dataset loader, training, metrics
├── serve.py            waitress server (Windows)
├── templates/index.html  dashboard
├── tests/              pytest suite
├── Dockerfile · Procfile · render.yaml
└── .github/workflows/ci.yml   runs the tests on every push
```

## Testing

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

## Deploy

- **Render (free tier):** push to GitHub, then New Web Service. It reads `render.yaml` and uses `/health`.
- **Docker:** `docker build -t inboxguard . && docker run -p 8000:8000 inboxguard`

After deploying, check the dashboard footer: it must name your real dataset, not "synthetic demo data".

## Limitations

- Not a replacement for an enterprise email security gateway. It is an educational and analyst-assist tool.
- Public phishing datasets are old and do not reflect modern AI-written or QR-code phishing.
- Rules use a short brand list and a simplified registered-domain rule; production use should add `tldextract`.
- No live threat intelligence yet (URLhaus, VirusTotal, WHOIS domain age).
- Header checks rely on the `Authentication-Results` header written by your own mail provider.
- The rate limiter is in-memory and per process.

## Roadmap

- [ ] Threat-intel lookups with SQLite caching
- [ ] `.eml` file upload
- [ ] Scan history and trend charts
- [ ] Transformer model (DistilBERT) comparison

## Privacy and safety

Pasted emails may contain personal data. Request bodies are not logged or stored. Links are displayed as text and are never opened or fetched.

## License

MIT, see [LICENSE](LICENSE).
