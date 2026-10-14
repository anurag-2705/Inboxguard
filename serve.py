"""Production-style server that works on Windows (gunicorn does not). Run: python serve.py"""
import os
from waitress import serve
from app import app

port = int(os.environ.get("PORT", 8000))
print(f"InboxGuard running on http://127.0.0.1:{port}")
serve(app, host="0.0.0.0", port=port, threads=4)
