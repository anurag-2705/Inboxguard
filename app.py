"""Step 3 - Flask web server.
Local dev:   python app.py            -> http://127.0.0.1:5000
Windows prod: python serve.py         (waitress)
Linux/cloud:  gunicorn app:app        (see Procfile)
"""
import json
import os
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, render_template, request

from analyzer import analyze

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
MAX_CHARS = 20000
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", 30))  # requests per minute per IP (in-memory, per process)
_hits = defaultdict(deque)


def rate_limited(ip):
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/api/model-info")
def model_info():
    try:
        with open(os.path.join(BASE_DIR, "metrics.json")) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify(source="unknown")


@app.post("/analyze")
def analyze_route():
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "?").split(",")[0].strip()
    if rate_limited(ip):
        return jsonify(error="Too many requests. Please wait a minute."), 429
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Paste an email or a link to analyze."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Input is too long. Limit is {MAX_CHARS} characters."), 413
    return jsonify(analyze(text))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
