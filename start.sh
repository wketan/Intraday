#!/usr/bin/env bash
# Startup wrapper for Railway / nixpacks. Procfile-only invocation can fail
# to expand $PORT depending on how the platform parses Procfile lines —
# routing through this script forces a real shell with proper expansion.
set -e

echo "=== Startup diagnostics ==="
echo "PORT env var: ${PORT:-NOT_SET}"
echo "Working dir:  $(pwd)"
echo "Python:       $(python3 --version 2>&1)"
echo "Gunicorn:     $(gunicorn --version 2>&1 | head -1)"
echo "==========================="

# Railway sets PORT for web services. Fall back to 5050 for local 'foreman start'.
PORT="${PORT:-5050}"

# `exec` replaces the shell process so signals (SIGTERM, SIGINT) reach gunicorn
# directly — important for graceful shutdown when Railway redeploys.
exec gunicorn server:app \
    --bind "0.0.0.0:${PORT}" \
    --timeout 120 \
    --workers 1 \
    --threads 4
