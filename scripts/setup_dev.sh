#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# ATTENDANCE_VENV allows cloud tasks to retain dependencies outside the checkout.
attendance_venv="${ATTENDANCE_VENV:-$PWD/.venv}"
if [ ! -x "$attendance_venv/bin/python" ]; then
  "${ATTENDANCE_PYTHON:-python3}" -m venv "$attendance_venv"
fi
"$attendance_venv/bin/python" -m pip install -r requirements-dev.txt
"$attendance_venv/bin/python" scripts/prepare_models.py
npm --prefix frontend ci --no-audit --no-fund
"$attendance_venv/bin/python" scripts/check_runtime.py --image face1.jpg
npm --prefix frontend run build
