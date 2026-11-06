FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py analyzer.py preprocess.py model.pkl metrics.json ./
COPY templates ./templates

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health', timeout=4)"

# One worker + threads keeps memory low (the model is loaded once). Behind a proxy, set TRUSTED_PROXIES=1.
CMD ["sh", "-c", "exec gunicorn app:app --workers 1 --threads 4 --timeout 60 --bind 0.0.0.0:${PORT}"]
