#!/usr/bin/env bash
# Startup wrapper for Railway. Routes gunicorn through a real shell so
# $PORT is always expanded correctly regardless of how Railway invokes us.

# Don't exit on diagnostics failures — only on the final gunicorn exec.
echo "=== Startup diagnostics ==="
echo "PORT env var: ${PORT:-NOT_SET}"
echo "Working dir:  $(pwd)"
python3 --version 2>&1 || true
gunicorn --version 2>&1 || true
echo "==========================="

# Railway sets PORT for web services. Fall back to 5050 for local 'foreman start'.
PORT="${PORT:-5050}"

# exec replaces the shell so signals reach gunicorn directly (graceful shutdown).
exec gunicorn server:app \
    --bind "0.0.0.0:${PORT}" \
    --timeout 120 \
    --workers 1 \
    --threads 4 \
    --log-level info \
    --access-logfile - \
    --error-logfile -
