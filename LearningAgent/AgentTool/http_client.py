# http_client.py
import os
import requests
from datetime import datetime, timezone  # <-- add timezone

ASSESSMENT_BASE = os.getenv("ASSESSMENT_BASE", "http://localhost:8000")
CLASSIFIER_BASE = os.getenv("CLASSIFIER_BASE", "http://localhost:8000")
THERAPY_BASE = os.getenv("THERAPY_BASE", "http://localhost:8000")
MONITOR_BASE = os.getenv("MONITOR_BASE", "http://localhost:8000")

def post_json(url: str, payload: dict, timeout=20):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_json(url: str, timeout=20):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def utc_now_iso() -> str:
    # Always Z-terminated for Mongo parsing consistency
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def report(agent_name: str, user_id: str, data: dict, session_id: str | None = None, version: int = 1):
    payload = {
        "agent_name": agent_name,
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": utc_now_iso(),
        "version": version,
        "data": data
    }
    return post_json(f"{MONITOR_BASE}/monitor/report", payload)
