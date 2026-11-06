"""Production server for Windows (and anywhere gunicorn is unavailable)."""
import os

from waitress import serve

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"InboxGuard running on http://127.0.0.1:{port}")
    serve(app, host="0.0.0.0", port=port, threads=4)
