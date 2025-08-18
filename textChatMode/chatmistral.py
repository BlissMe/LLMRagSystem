from __future__ import annotations

import os
import re
import json
import requests

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pymongo import MongoClient

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

from utils.phq9_questions import PHQ9_QUESTIONS
from utils.tts import generate_tts_audio
import key_param

router = APIRouter()

# -----------------------------
# Ollama / ngrok configuration
# -----------------------------
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "https://d53cb0fd37cb.ngrok-free.app").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-mentalhealth")
_auth_tuple = (os.getenv("OLLAMA_USER"), os.getenv("OLLAMA_PASS"))
AUTH = _auth_tuple if all(_auth_tuple) else None

# Stops we want the model to respect
TAG_STOPS = [
    "</s>", "<s>", "[INST]", "[/INST]",  # llama-ish tokens
    "<<", "<USER",                       # typical SYS/USER tag starters
    "User:", "Assistant:", "\nUser", "\nAssistant"
]

# -------------- helpers --------------

def strip_tags(s: str) -> str:
    """Remove <<...>>, <...>, [INST] markers, and collapse whitespace."""
    if not s:
        return ""
    s = re.sub(r"<<[^>]*>>", " ", s)
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"\[/?INST\]", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()

def keep_1_to_2_sentences(s: str) -> str:
    """Keep only the first 1–2 sentences to avoid rambling."""
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return " ".join(parts[:2]).strip()

def _ollama_chat(system_text: str, user_text: str,
                 temperature: float = 0.25, num_predict: int = 140) -> str:
    """Preferred: /api/chat (respects system/user roles)."""
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

def _ollama_generate(system_text: str, user_text: str,
                     temperature: float = 0.25, num_predict: int = 140) -> str:
    """Fallback: /api/generate (relies on Modelfile TEMPLATE)."""
    prompt = f"<<SYS>>\n{system_text}\n<</SYS>>\n\n{user_text}"
    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        auth=AUTH,
        headers={"Content-Type": "application/json"},
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
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
    return (r.json().get("response") or "").strip()

# -------------- request models --------------

class SummaryRequest(BaseModel):
    history: str

@router.post("/summarize")
async def summarize_chat(data: SummaryRequest):
    return {"summary": " ".join(data.history.split())}

class QueryRequest(BaseModel):
    user_query: str
    history: str
    summaries: list[str] = []
    asked_phq_ids: list[int] = []

# -------------- main chat endpoint --------------

@router.post("/ask")
async def ask_question(data: QueryRequest):
    query, history = data.user_query, data.history

    # --- RAG: MongoDB vector search ---
    client = MongoClient(key_param.MONGO_URI)
    try:
        vs = MongoDBAtlasVectorSearch(
            collection=client["Depression_Knowledge_Base"]["depression"],
            embedding=OpenAIEmbeddings(openai_api_key=key_param.openai_api_key),
            index_name="default1",
        )
        docs = vs.similarity_search(query, k=3)
        context_texts = [strip_tags(d.page_content[:500]) for d in docs]
    finally:
        client.close()

    summary_text = "\n".join(data.summaries) if data.summaries else "No previous summaries available."

    # PHQ-9 flow
    unasked = [q for q in PHQ9_QUESTIONS if q["id"] not in data.asked_phq_ids]
    next_phq = unasked[0] if unasked else None
    user_turns = [ln for ln in history.splitlines() if ln.lower().startswith(("you:", "user:"))]
    early_stage = len(user_turns) < 10

    phq_instruction = ""
    if next_phq and not early_stage:
        if not data.asked_phq_ids:
            phq_instruction += (
                "Open with: To better understand how you're doing, I'd like to ask a few "
                "short questions about the past two weeks.\n"
            )
        phq_instruction += (
            f'Ask exactly this next question (one per message): "{next_phq["question"]}" '
            f'(meaning: {next_phq["meaning"]}). '
            "Let user answer with: not at all / several days / more than half the days / nearly every day.\n"
        )

    # SYSTEM message: strict rules + context for the model (never echoed)
    system_text = f"""
You are a friendly assistant who speaks like a kind friend.
Be warm, concise, and non-repetitive. Reply in between 1 to 5 sentences unless asked for more.
always try to build conversation with user.
Use the context only to inform your reply—do NOT quote or repeat the context/history/user verbatim.
Never output markup/tags like <...>, <<...>>, [INST], User:, Assistant:, </s>, <s>.
Ask PHQ-9 questions naturally when ready, but never mention “PHQ-9”.
Avoid medical/crisis terms unless asked.
Always end your message with one gentle question that invites a reply.
Output only your reply to the user—no prefaces or labels.

Conversation summary (for you):
{summary_text}

Relevant context (for you):
{context_texts}

Guidance for the next turn (for you):
{phq_instruction}
""".strip()

    user_text = query

    # Call Ollama: chat first, fallback to generate
    try:
        reply = _ollama_chat(system_text, user_text, temperature=0.25, num_predict=140)
        if not reply:
            reply = _ollama_generate(system_text, user_text, temperature=0.25, num_predict=140)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama call failed: {e}")

    # Post-clean and keep short
    reply = keep_1_to_2_sentences(strip_tags(reply))
    if not reply:
        reply = "I'm here with you. What feels hardest right now?"

    audio_path = generate_tts_audio(reply)  # returns None on empty
    return {
        "response": reply,
        "audio_url": (f"/voice-audio?path={audio_path}" if audio_path else None),
        "phq9_questionID": (next_phq["id"] if (next_phq and not early_stage) else None),
        "phq9_question": (next_phq["question"] if (next_phq and not early_stage) else None),
    }

@router.get("/voice-audio")
def voice_audio(path: str):
    return FileResponse(path, media_type="audio/mpeg", filename="bot_reply.mp3")
