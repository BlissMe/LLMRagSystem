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
MONITOR_URL = "http://localhost:8000/monitor-agent/track-activity"


# =======================
# MODELS
# =======================
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
    duration: float | None = None
    feedback: str | None = None


# =======================
# MONITOR HELPERS
# =======================
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


def send_therapy_progress_event(user_id: int, session_id: int, therapy_id: str, therapy_name: str, progress: float):
    send_monitor_event(
        "THERAPY_IN_PROGRESS",
        {"input": {}, "output": {"therapy_id": therapy_id, "therapy_name": therapy_name, "progress": progress}},
        user_id,
        session_id
    )


def send_end_therapy_event(user_id: int, session_id: int, therapy_id: str, therapy_name: str, feedback: str, duration: float | None):
    # mark progress 100%
    send_therapy_progress_event(user_id, session_id, therapy_id, therapy_name, progress=1.0)

    send_monitor_event(
        "THERAPY_ENDED",
        {
            "input": {},
            "output": {
                "therapy_id": therapy_id,
                "therapy_name": therapy_name,
                "feedback": feedback,
                "duration": duration
            }
        },
        user_id,
        session_id
    )


# =======================
# FEEDBACK ENDPOINT
# =======================
@router.post("/feedback")
async def save_therapy_feedback(data: TherapyFeedback):
    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]
    history_collection = db["TherapyHistory"]

    history_record = history_collection.find_one({
        "user_id": data.user_id,
        "session_id": data.session_id,
        "therapy_id": data.therapy_id
    })

    therapy_name = history_record.get("therapy_name", "Unknown Therapy") if history_record else "Unknown Therapy"

    # Save feedback
    history_collection.update_one(
        {"user_id": data.user_id, "session_id": data.session_id, "therapy_id": data.therapy_id},
        {
            "$set": {
                "duration": data.duration,
                "feedback": data.feedback,
                "feedback_time": datetime.utcnow()
            },
            "$setOnInsert": {
                "user_id": data.user_id,
                "session_id": data.session_id,
                "therapy_id": data.therapy_id,
                "therapy_name": therapy_name
            }
        },
        upsert=True
    )

    # log event
    send_end_therapy_event(
        data.user_id,
        data.session_id,
        data.therapy_id,
        therapy_name,
        data.feedback,
        data.duration
    )

    client.close()
    return {"success": True, "message": "Therapy feedback saved and session ended"}


# =======================
# CHAT ENDPOINT
# =======================
@router.post("/chat")
async def therapy_chat(data: TherapyRequest):

    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]

    history_records = get_user_therapy_history(db, data.user_id)
    recent_history = "\n".join(
        [
            f"{h.get('therapy_name', 'Unknown Therapy')} on {h.get('date', 'Unknown Date')} "
            f"(duration {h.get('duration', 'N/A')} mins)"
            for h in history_records
        ]
    ) if history_records else "No prior therapies found."

    # therapy suggestion
    therapy_suggestion = get_therapy_recommendation(db, data.depression_level, history_records)
    therapy_name = therapy_suggestion.get("name")
    therapy_id = therapy_suggestion.get("id")
    therapy_path = therapy_suggestion.get("path", None)
    therapy_description = therapy_suggestion.get("description", "")

    # =======================
    # LLM prompt
    # =======================
    prompt = f"""
You are a warm, friendly therapy assistant.

Rules:
- Keep responses short and caring.
- Never mention depression level.
- Suggest a therapy gently when appropriate.
- If suggesting, ask: "Would you like to start the {therapy_name} therapy now?"
If the user agrees you MUST respond:
ACTION:START_THERAPY:{therapy_id}

User history:
{recent_history}

User message: "{data.user_query}"
"""

    bot = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=key_param.openai_api_key)
    response = bot.invoke([{"role": "user", "content": prompt}])

    original_reply = response.content.strip()
    reply_lower = original_reply.lower()

    # detect suggestion
    suggestion_phrases = [
        "start the",
        "would you like to start",
        "try the",
        "therapy now",
        therapy_name.lower()
    ]
    is_therapy_suggested = any(p in reply_lower for p in suggestion_phrases)

    if is_therapy_suggested:
        send_monitor_event(
            "THERAPY_SUGGESTED",
            {
                "input": {"user_query": data.user_query},
                "output": {"therapy_id": therapy_id, "therapy_name": therapy_name}
            },
            data.user_id,
            data.session_id
        )

    # detect ACTION
    action_pattern = r"action\s*[:\- ]+\s*start[_\- ]?therapy\s*[:\- ]+\s*([A-Za-z0-9]+)"
    match = re.search(action_pattern, reply_lower)
    action_detected = match.group(1).strip() if match else None

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

        send_therapy_progress_event(
            data.user_id,
            data.session_id,
            therapy_id,
            therapy_name,
            progress=0.0
        )

        is_therapy_suggested = False

    client.close()

    clean_reply = original_reply.replace("ACTION:START_THERAPY", "").strip()

    return {
        "response": clean_reply,
        "action": "START_THERAPY" if action_detected else None,
        "therapy_id": therapy_id if (action_detected or is_therapy_suggested) else None,
        "therapy_name": therapy_name if (action_detected or is_therapy_suggested) else None,
        "therapy_description": therapy_description,
        "therapy_path": therapy_path,
        "isTherapySuggested": is_therapy_suggested,
        "therapySuggestion": {
            "id": therapy_id,
            "name": therapy_name,
            "path": therapy_path
        } if is_therapy_suggested else None,
    }


class ManualStartRequest(BaseModel):
    user_id: int
    session_id: int
    therapy_id: str
    therapy_name: str


@router.post("/end-start")
async def manual_start_therapy(data: ManualStartRequest):
    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]

    # save history
    save_therapy_history(
        db,
        data.user_id,
        data.session_id,
        data.therapy_name,
        data.therapy_id,
        duration=None,
        feedback=None
    )

    # log start
    send_monitor_event(
        "THERAPY_STARTED",
        {
            "input": {},
            "output": {
                "therapy_id": data.therapy_id,
                "therapy_name": data.therapy_name
            }
        },
        data.user_id,
        data.session_id
    )

    # progress 0%
    send_therapy_progress_event(
        data.user_id,
        data.session_id,
        data.therapy_id,
        data.therapy_name,
        progress=0.0
    )

    client.close()

    return {
        "success": True,
        "message": "Therapy session started",
        "therapy_id": data.therapy_id,
        "therapy_name": data.therapy_name
    }
