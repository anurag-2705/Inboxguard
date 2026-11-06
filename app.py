"""InboxGuard web server (Flask).

Local dev:        python app.py                 -> http://127.0.0.1:5000
Windows prod:     python serve.py               (waitress)
Linux / cloud:    gunicorn app:app              (see Procfile)
"""
import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict, deque

from flask import Flask, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

import analyzer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("inboxguard")      # never log message bodies: they may contain personal data

MAX_CHARS = 20000                                 # pasted text
MAX_UPLOAD = 1_000_000                            # .eml upload, bytes
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", 30))  # requests per minute per client IP
TRUSTED_PROXIES = int(os.environ.get("TRUSTED_PROXIES", 0))  # proxies in front of the app: 0 = none (default), 1 = Render/Heroku

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD + 10_000
if TRUSTED_PROXIES:
    # Only the address added by OUR proxy is trusted; a client-supplied X-Forwarded-For cannot spoof it.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=TRUSTED_PROXIES, x_proto=TRUSTED_PROXIES)


class RateLimiter:
    """Sliding-window limiter. In memory and per process: use Redis + Flask-Limiter if you scale out."""

    def __init__(self, limit, window=60, max_clients=10_000):
        self.limit, self.window, self.max_clients = limit, window, max_clients
        self.hits = OrderedDict()
        self.lock = threading.Lock()

    def blocked(self, key):
        now = time.time()
        with self.lock:
            q = self.hits.get(key)
            if q is None:
                if len(self.hits) >= self.max_clients:      # bounded memory: drop the least recently seen
                    self.hits.popitem(last=False)
                q = self.hits[key] = deque()
            self.hits.move_to_end(key)
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return True
            q.append(now)
            return False


limiter = RateLimiter(RATE_LIMIT)
threading.Thread(target=analyzer.get_model, daemon=True).start()   # warm the model without delaying startup


@app.after_request
def security_headers(resp):
    nonce = getattr(request, "csp_nonce", None)
    csp = ["default-src 'none'", "connect-src 'self'", "img-src 'self' data:", "base-uri 'none'",
           "form-action 'none'", "frame-ancestors 'none'"]
    csp.append(f"script-src 'nonce-{nonce}'" if nonce else "script-src 'none'")
    csp.append(f"style-src 'nonce-{nonce}'" if nonce else "style-src 'none'")
    resp.headers["Content-Security-Policy"] = "; ".join(csp)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.is_secure:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if request.path.startswith("/api") or request.path == "/analyze":
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    request.csp_nonce = secrets.token_urlsafe(16)
    return render_template("index.html", nonce=request.csp_nonce)


@app.get("/health")
def health():
    return jsonify(status="ok")          # deliberately does not load the model


@app.get("/api/model-info")
def model_info():
    info = {"source": "unknown"}
    try:
        with open(os.path.join(BASE_DIR, "metrics.json")) as f:
            info = json.load(f)
    except (FileNotFoundError, ValueError):
        pass
    bundle = analyzer.get_model()
    info["ml_active"] = bool(bundle) and not bundle.get("demo")
    return jsonify(info)


@app.post("/analyze")
def analyze_route():
    if limiter.blocked(request.remote_addr or "?"):
        return jsonify(error="Too many requests. Please wait a minute."), 429
    upload = request.files.get("file")
    if upload is not None:                                         # .eml upload
        data = upload.read(MAX_UPLOAD + 1)
        if not data.strip():
            return jsonify(error="The uploaded file is empty."), 400
        if len(data) > MAX_UPLOAD:
            return jsonify(error="File is too large. Limit is 1 MB."), 413
        try:
            return jsonify(analyzer.analyze_eml(data))
        except Exception:
            log.exception("could not analyze uploaded file")
            return jsonify(error="Could not read that file. Upload a plain .eml email."), 422
    payload = request.get_json(silent=True)
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        return jsonify(error="Send JSON like {\"text\": \"...\"} or upload an .eml file."), 400
    text = text.strip()
    if not text:
        return jsonify(error="Paste an email or a link to analyze."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Input is too long. Limit is {MAX_CHARS} characters."), 413
    return jsonify(analyzer.analyze(text))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Request is too large."), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


@app.errorhandler(Exception)
def unexpected(err):
    if hasattr(err, "code") and isinstance(err.code, int) and err.code < 500:
        return jsonify(error=getattr(err, "description", "Bad request.")), err.code
    log.exception("unhandled error")
    return jsonify(error="Something went wrong on our side. Please try again."), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
