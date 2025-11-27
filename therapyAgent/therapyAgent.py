from fastapi import APIRouter
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from datetime import datetime
import key_param
import re
import requests

from .utils.therapy_selector import get_therapy_recommendation
from .utils.history_tracker import save_therapy_history, get_user_therapy_history

router = APIRouter(prefix="/therapy-agent", tags=["Therapy Agent"])

class TherapyRequest(BaseModel):
    user_query: str
    depression_level: str
    user_id: int
    session_id: int
    session_summaries: list[str] = []

class TherapyFeedback(BaseModel):
    user_id: int
    session_id: int | None = None
    therapy_id: str
    duration: float  | None = None
    feedback: str | None = None


def send_monitor_event(event_name: str, data: dict, user_id: int, session_id: int):
    payload = {
        "agent_name": "therapy",
        "user_id": user_id,
        "session_id": session_id,
        "input_data": {"event": event_name, **data.get("input", {})},
        "output_data": data.get("output", {}),
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        requests.post(MONITOR_URL, json=payload, timeout=10)
        print(f"Logged Therapy Event → {event_name}")
    except Exception as e:
        print("Monitor Agent Logging Failed:", e)

def send_therapy_progress_event(user_id: int, session_id: int, therapy_id: str, progress: float):
    send_monitor_event(
        "THERAPY_IN_PROGRESS",
        {"input": {}, "output": {"therapy_id": therapy_id, "progress": progress}},
        user_id,
        session_id
    )

def send_end_therapy_event(user_id: int, session_id: int, therapy_id: str, feedback: str, duration: float | None):
    # Send final 100% progress
    send_therapy_progress_event(user_id, session_id, therapy_id, progress=1.0)
    send_monitor_event(
        "THERAPY_ENDED",
        {"input": {}, "output": {"therapy_id": therapy_id, "feedback": feedback, "duration": duration}},
        user_id,
        session_id
    )

@router.post("/feedback")
async def save_therapy_feedback(data: TherapyFeedback):
    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]
    history_collection = db["TherapyHistory"]

    history_collection.update_one(
        {"user_id": data.user_id, "session_id": data.session_id, "therapy_id": data.therapy_id},
        {
            "$set": {"duration": data.duration, "feedback": data.feedback, "feedback_time": datetime.utcnow()},
            "$setOnInsert": {"user_id": data.user_id, "session_id": data.session_id, "therapy_id": data.therapy_id},
        },
        upsert=True
    )

    # Mark session as ended with final progress
    send_end_therapy_event(data.user_id, data.session_id, data.therapy_id, data.feedback, data.duration)

    client.close()
    return {"success": True, "message": "Therapy feedback saved and session ended"}


@router.post("/feedback")
async def save_therapy_feedback(data: TherapyFeedback):

    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]

    history_collection = db["TherapyHistory"]

    # Update the latest therapy session
    history_collection.update_one(
        {
            "user_id": data.user_id,
            "session_id": data.session_id,
            "therapy_id": data.therapy_id
        },
        {
            "$set": {
                "duration": data.duration,
                "feedback": data.feedback,
                "feedback_time": datetime.utcnow()
            },
            "$setOnInsert": {
                "user_id": data.user_id,
                "session_id": data.session_id,
                "therapy_id": data.therapy_id
            }
        },
        upsert=True
    )

    client.close()

    return {"success": True, "message": "Therapy feedback saved"}

@router.post("/chat")
async def therapy_chat(data: TherapyRequest):

    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]

    # Fetch therapy history
    history_records = get_user_therapy_history(db, data.user_id)
    recent_history = "\n".join(
        [
            f"{h['therapy_name']} on {h['date']} (duration {h['duration']} mins)"
            for h in history_records
        ]
    ) if history_records else "No prior therapies found."

    # Get new suggestion
    therapy_suggestion = get_therapy_recommendation(
        db, data.depression_level, history_records
    )

    therapy_name = therapy_suggestion.get("name")
    therapy_id = therapy_suggestion.get("id")
    therapy_path = therapy_suggestion.get("path", None)
    therapy_description = therapy_suggestion.get("description", "")

    # =======================
    #       BASE PROMPT
    # =======================
    prompt = f"""
You are a warm, friendly therapy assistant. 
Your main job is to support the user emotionally AND suggest a therapy when appropriate.

Rules:
- Keep responses short, caring, simple.
- Never mention depression level.
- Suggest therapies gently when appropriate.
- If suggesting a therapy, ask:
  "Would you like to start the {therapy_name} therapy now?"

If the user agrees, you MUST respond with EXACT format:
ACTION:START_THERAPY:{therapy_id}

User history:
{recent_history}

User message: "{data.user_query}"
"""

    bot = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=key_param.openai_api_key)
    response = bot.invoke([{"role": "user", "content": prompt}])

    reply_text = response.content.strip().lower()

    # ===============================================================
    # 1. Detect if model SUGGESTED the therapy (frontend uses this)
    # ===============================================================
    suggestion_phrases = [
        f"start the {therapy_name.lower()} therapy",
        f"try the {therapy_name.lower()} therapy",
        f"{therapy_name.lower()} therapy now",
    ]

    is_therapy_suggested = any(p in reply_text for p in suggestion_phrases)

    # ===============================================================
    # 2. Detect ACTION:START_THERAPY in ALL POSSIBLE MODEL VARIATIONS
    # ===============================================================
    # regex covers:
    # ACTION:START_THERAPY:ID
    # action: start_therapy : id
    # Action:Start-Therapy:id
    # ---------------------------------------------------------------
    action_pattern = r"action\s*:\s*start[_\- ]therapy\s*:\s*([A-Za-z0-9]+)"

    match = re.search(action_pattern, reply_text, re.IGNORECASE)

    action_detected = match.group(1).strip() if match else None

    # If ACTION detected → save history
    if action_detected:
        save_therapy_history(
            db,
            data.user_id,
            data.session_id,
            therapy_name,
            therapy_id,
            duration=None,
            feedback=None
        )
        is_therapy_suggested = False  # because user already started
        send_monitor_event(
            "THERAPY_STARTED",
            {"input": {"user_query": data.user_query}, "output": {"therapy_id": therapy_id, "therapy_name": therapy_name}},
            data.user_id,
            data.session_id
        )
        send_therapy_progress_event(data.user_id, data.session_id, therapy_id, progress=0.0)



    client.close()

    # ===============================================
    # FINAL RESPONSE TO FRONTEND (Fully Consistent)
    # ===============================================
    return {
        "response": response.content.replace("ACTION:START_THERAPY", "").strip(),
        "action": "START_THERAPY" if action_detected else None,
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
    }
