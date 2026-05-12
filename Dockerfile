# Dockerfile used by Railway (and Fly.io). Forces explicit control over the
# start command, bypassing Railway's UI Custom Start Command + Procfile chain
# that was failing to expand $PORT.
FROM python:3.11-slim

# System deps — needed for pandas/numpy and SmartAPI SSL on slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so they're cached when only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code — only the files the server needs at runtime.
COPY server.py .
COPY backtest.py .
COPY data_layer.py .
COPY verify_data_layer.py .
COPY signal_v2.py .
COPY regime.py .
COPY events.json .
COPY index.html .
COPY manifest.json .
COPY start.sh .
RUN chmod +x start.sh
# Persistent data cache directory — gitignored locally but writable in container.
RUN mkdir -p /app/data /app/reports

ENV PYTHONUNBUFFERED=1

# Shell form — `sh -c` ensures ${PORT:-8080} expands at container start.
# Single worker because the engine has in-process state (chain cache, regime, OI history).
# Multiple workers would each run their own scan loop and double up on Angel One.
CMD ["sh", "-c", "gunicorn server:app --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 1 --threads 4"]
