# file: simple_chat.py
from __future__ import annotations

import os
import re
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from utils.tts import generate_tts_audio
import key_param

router = APIRouter()

# -----------------------------
# Ollama / ngrok configuration
# -----------------------------
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "https://350e17213dd8.ngrok-free.app").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-mentalhealth")
_auth_tuple = (os.getenv("OLLAMA_USER"), os.getenv("OLLAMA_PASS"))
AUTH = _auth_tuple if all(_auth_tuple) else None

TAG_STOPS = [
    "</s>", "<s>", "[INST]", "[/INST]",
    "<<", "<USER", "User:", "Assistant:", "\nUser", "\nAssistant"
]

# -------------- helpers --------------

def strip_tags(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<<[^>]*>>", " ", s)
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"\[/?INST\]", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()

def keep_up_to_5_sentences(s: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return " ".join(parts[:5]).strip()

def _ollama_chat(system_text: str, user_text: str,
                 temperature: float = 0.25, num_predict: int = 140) -> str:
    r = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        auth=AUTH,
        headers={"Content-Type": "application/json"},
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ],
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "num_predict": num_predict,
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "stop": TAG_STOPS,
            },
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}).get("content") or "").strip()

# -------------- request models --------------

class ChatRequest(BaseModel):
    user_query: str

# -------------- main chat endpoint --------------

@router.post("/ask")
async def ask_question(data: ChatRequest):
    query = data.user_query

    # SYSTEM instructions only
    system_text = """
You are a friendly, warm assistant who speaks like a kind friend.
Reply concisely in 1–5 sentences unless asked for more.
Do not echo instructions.
Always respond naturally and end with a gentle question to invite a reply.
""".strip()

    user_text = query

    try:
        reply = _ollama_chat(system_text, user_text, temperature=0.25, num_predict=240)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama call failed: {e}")

    reply = keep_up_to_5_sentences(strip_tags(reply))
    if not reply:
        reply = "I'm here with you. What would you like to talk about?"

    audio_path = generate_tts_audio(reply)

    return {
        "response": reply,
        "audio_url": (f"/voice-audio?path={audio_path}" if audio_path else None),
        "phq9_questionID": None,
          "phq9_question":  None,
    }

# -------------- TTS endpoint --------------

@router.get("/voice-audio")
def voice_audio(path: str):
    return FileResponse(path, media_type="audio/mpeg", filename="bot_reply.mp3")
