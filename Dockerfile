# Dockerfile for Fly.io deployment. Mirrors the Render runtime (Python 3.11.6).
FROM python:3.11.6-slim

# System deps — only needed for pandas/numpy wheels on slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so they're cached when only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then app code.
COPY server.py .
COPY backtest.py .
COPY events.json .
COPY index.html .
COPY manifest.json .

# Fly's $PORT default is 8080 (set in fly.toml [env]).
ENV PYTHONUNBUFFERED=1

# Single worker because the engine has in-process state (chain cache, regime, OI history).
# Multiple workers would each run their own scan loop and double up on Angel One.
CMD ["sh", "-c", "gunicorn server:app --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 1 --threads 4"]
