"""Fail the container readiness check if the database/models are unavailable."""
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=5) as response:
    health = json.load(response)
if health.get("status") != "ok" or not health.get("database_ready") or not health.get("models_ready"):
    raise SystemExit("Attendance service is not ready")
