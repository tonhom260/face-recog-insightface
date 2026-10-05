#!/bin/sh
set -eu
python scripts/prepare_models.py --only backend --verify-only
python scripts/check_runtime.py
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers \
    --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}"
