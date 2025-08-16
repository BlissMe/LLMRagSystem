# services/agent.py
import os
import re
import json
from typing import Dict, Any
from groq import Groq

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

class AgentError(Exception):
    pass

def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise AgentError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)

# ------------------------------
# Helpers
# ------------------------------
_EMOTION_ALLOWED = {"happy", "neutral", "sad", "angry", "fearful"}

def _extract_json(text: str):
    """Try to parse JSON directly; if that fails, grab the first {...} block."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def _to_int(x, default=0, lo=None, hi=None):
    try:
        v = int(x)
    except Exception:
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v

def _to_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "yes", "y", "1", "detected", "present"}:
            return True
        if s in {"false", "no", "n", "0", "not_detected", "absent"}:
            return False
    return default

# ------------------------------
# Main: single-call analysis
# ------------------------------
def analyze_text_for_depression_and_emotion(text: str) -> Dict[str, Any]:
    """
    Calls Groq and returns:
    {
      "depression": {
        "label": "Depression Signs Detected" | "No Depression Signs Detected",
        "confidence_detected": 0-100  # confidence that depression IS detected
      },
      "emotion": {"label": "happy|neutral|sad|angry|fearful", "confidence": 0-100},
      "rationale": "short reason"
    }
    """
    client = _client()

    system = (
        "You are a careful mental-health signal classifier. "
        "You DO NOT diagnose. Output STRICT JSON only, no extra text."
    )
    user = f"""
Analyze the following conversation material for *signals* of depression (not a diagnosis) and the dominant emotion.

TEXT:
{text}

Return JSON with this exact schema:
{{
  "depression": {{
    "detected": true,            // true if signs of depression are present, else false
    "confidence": 0              // integer 0-100 indicating confidence in your detected/not-detected choice
  }},
  "emotion": {{
    "label": "happy|neutral|sad|angry|fearful",
    "confidence": 0
  }},
  "rationale": "one concise sentence explaining your reasoning"
}}
Only JSON. No extra words.
"""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    # Prefer strict JSON from response_format; fallback to regex if needed
    raw = resp.choices[0].message.content if resp.choices else ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _extract_json(raw)
        if data is None:
            raise AgentError(f"Groq returned non-JSON: {raw[:200]}...")

    # --- Light validation / normalization ---
    dep_in = data.get("depression", {}) if isinstance(data, dict) else {}
    emo_in = data.get("emotion", {}) if isinstance(data, dict) else {}

    detected_bool = _to_bool(dep_in.get("detected"), default=False)
    model_conf = _to_int(dep_in.get("confidence"), default=50, lo=0, hi=100)

    # Convert to requested label text
    dep_label = "Depression Signs Detected" if detected_bool else "No Depression Signs Detected"

    # Always report confidence *for detected* (i.e., away-from-no)
    confidence_detected = model_conf if detected_bool else (100 - model_conf)

    emo_label = str(emo_in.get("label", "neutral")).strip().lower()
    if emo_label not in _EMOTION_ALLOWED:
        emo_label = "neutral"
    emo_conf = _to_int(emo_in.get("confidence"), default=50, lo=0, hi=100)

    rationale = str(data.get("rationale", "")).strip()

    return {
        "depression": {
            "label": dep_label,
            "confidence_detected": confidence_detected,
        },
        "emotion": {"label": emo_label, "confidence": emo_conf},
        "rationale": rationale,
    }
