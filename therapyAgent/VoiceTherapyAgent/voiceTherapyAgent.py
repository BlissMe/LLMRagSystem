from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment
from io import BytesIO
import json
import requests
import re
from datetime import datetime
from pymongo import MongoClient
import os

import key_param
from textChatMode.chat import QueryRequest
from ..utils.therapy_selector import get_therapy_recommendation
from ..utils.history_tracker import save_therapy_history, get_user_therapy_history
from langchain_openai import ChatOpenAI
from fastapi.responses import StreamingResponse
from io import BytesIO

voice_router = APIRouter(prefix="/voice-therapy-agent", tags=["Voice Therapy Agent"])
tts_router = APIRouter(prefix="/tts", tags=["TTS"])

OPENAI_API_KEY = key_param.openai_api_key

# Ensure audio folder exists
@tts_router.post("")
async def generate_tts(payload: dict):
    try:
        text = payload.get("text")
        if not text:
            return JSONResponse(status_code=400, content={"error": "Text required"})

        res = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": "gpt-4o-mini-tts", "input": text},
        )

        if res.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "TTS failed"})

        audio_bytes = BytesIO(res.content)

        return StreamingResponse(
            audio_bytes,
            media_type="audio/mpeg"
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@voice_router.post("/chat")
async def voice_therapy_chat(
    audio: UploadFile = File(...),
    asked_phq_ids: str = Form("[]"),
    summaries: str = Form("[]"),
    history: str = Form(""),
    user_id: str = Form(...),
    session_id: str = Form(...),
    level: str = Form(...),
    therapy_feedback: str = Form(None)
):
    try:
        asked_ids = json.loads(asked_phq_ids)
        summary_list = json.loads(summaries)
        audio_bytes = await audio.read()

        # ------------------------------
        # CASE 1: NO AUDIO
        # ------------------------------
        if len(audio_bytes) == 0:
            user_query = (
                history.strip().split("\n")[-1].replace("User: ", "")
                if history else ""
            )
        else:
            # ------------------------------
            # Whisper STT
            # ------------------------------
            files = {"file": ("audio.webm", BytesIO(audio_bytes), audio.content_type)}
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            data = {"model": "whisper-1", "language": "en"}

            whisper_response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data
            )

            if whisper_response.status_code != 200:
                return JSONResponse(status_code=500, content={"error": "Whisper transcription failed"})

            user_query = whisper_response.json()["text"].strip()

        # ------------------------------
        # DB + THERAPY LOGIC
        # ------------------------------
        client = MongoClient(key_param.MONGO_URI)
        db = client["blissMe"]

        history_records = get_user_therapy_history(db, user_id)

        recent_history = "\n".join(
            f"{h['therapy_name']} on {h['date']} (duration {h['duration']} mins)"
            for h in history_records
        ) if history_records else "No prior therapies found."

        suggestion = get_therapy_recommendation(db, level, history_records)

        therapy_name = suggestion.get("name")
        therapy_id = suggestion.get("id")
        therapy_path = suggestion.get("path")
        therapy_description = suggestion.get("description", "")

        # ------------------------------
        # LLM
        # ------------------------------
        system_prompt = f"""
You are a warm, friendly therapy assistant. 
Your main job is to support the user emotionally AND suggest a therapy when APPROPRIATE.
Don't suggest therapies every time—only when it fits naturally in the conversation.
If the user seems distressed, PRIORITIZE empathy and understanding first.
DON'T push therapies if the user is not open to it.User therapy history:
{recent_history}

User: "{user_query}"
Depression level: {level}

Rules:
- Keep responses short, caring, simple.
- Never mention depression level.
- Suggest therapies gently when appropriate.
- If the user seems distressed, PRIORITIZE empathy and understanding first.
- If the user declines a therapy, respect their choice and continue the chat supportively.
- Don't suggest therapies every time—only when it fits naturally in the conversation.
- also can ask about previously done therapies, which those helped or not.
After responding kindly, you MAY suggest:
"Would you like to start the {therapy_name} therapy now?"

IF USER AGREES, return EXACTLY:
ACTION:START_THERAPY:{therapy_id}
"""

        bot = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

        response = bot.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ])

        bot_reply = response.content.strip()

        # ------------------------------
        # ACTION DETECTION
        # ------------------------------
        action_pattern = r"action\s*:\s*start[_\- ]therapy\s*:\s*([A-Za-z0-9]+)"
        match = re.search(action_pattern, bot_reply.lower())
        action_detected = match.group(1).strip() if match else None

        if action_detected:
            save_therapy_history(
                db, user_id, session_id,
                therapy_name, therapy_id,
                duration=None, feedback=None
            )

        # ------------------------------
        # SUGGESTION DETECTION
        # ------------------------------
        suggestion_phrases = [
            f"start the {therapy_name.lower()} therapy",
            f"try the {therapy_name.lower()} therapy",
            f"{therapy_name.lower()} therapy now",
        ]

        is_therapy_suggested = any(p in bot_reply.lower() for p in suggestion_phrases)
        if action_detected:
            is_therapy_suggested = False

        # ------------------------------
        # CLEAN BOT MESSAGE
        # ------------------------------
        clean_reply = re.sub(
            r"ACTION\s*:\s*START[_\- ]THERAPY\s*:\s*[A-Za-z0-9]+",
            "",
            bot_reply,
            flags=re.IGNORECASE
        ).strip()

        # ------------------------------
        # TEXT → SPEECH  (NO FILE SAVE)
        # ------------------------------
        audio_base64 = None

        tts_res = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"model": "gpt-4o-mini-tts", "input": clean_reply}
        )

        if tts_res.status_code == 200:
            import base64
            audio_base64 = base64.b64encode(tts_res.content).decode("utf-8")

        client.close()

        # ------------------------------
        # FINAL RESPONSE
        # ------------------------------
        return {
            "user_query": user_query,
            "bot_response": clean_reply,

            "action": "START_THERAPY" if (action_detected or is_therapy_suggested) else None,
            "therapy_id": therapy_id if (action_detected or is_therapy_suggested) else None,
            "therapy_name": therapy_name if (action_detected or is_therapy_suggested) else None,
            "therapy_description": therapy_description if (action_detected or is_therapy_suggested) else None,
            "therapy_path": therapy_path,

            "isTherapySuggested": is_therapy_suggested,
            "therapySuggestion": {
                "id": therapy_id,
                "name": therapy_name,
                "path": therapy_path,
            } if is_therapy_suggested else None,

            # AUDIO (NO FILE)
            "audio_base64": audio_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============================================================
# SERVE AUDIO FILES
# ============================================================
@voice_router.get("/audio/{filename}")
async def get_audio(filename: str):
    path = f"audio/{filename}"
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Audio not found"})
    return FileResponse(path, media_type="audio/mpeg")