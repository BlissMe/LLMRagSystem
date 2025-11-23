from fastapi import APIRouter
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from datetime import datetime
import key_param
import re

from .utils.therapy_selector import get_therapy_recommendation
from .utils.history_tracker import save_therapy_history, get_user_therapy_history

# =============== MONITOR AGENT CLIENT ===============
import httpx

async def log_to_monitor(event_type: str, payload: dict):
    """Send event to Monitor Agent"""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{key_param.MONITOR_AGENT_URL}/log",
                json={"event_type": event_type, "payload": payload},
                timeout=5
            )
    except Exception as e:
        print("⚠ Monitor Agent Logging Error:", e)

# ====================================================

router = APIRouter(prefix="/therapy-agent", tags=["Therapy Agent"])

class TherapyFeedback(BaseModel):
    user_id: int
    session_id: int | None = None
    therapy_id: str
    duration: float  | None = None
    feedback: str | None = None

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

class TherapyRequest(BaseModel):
    user_query: str
    depression_level: str
    user_id: str
    session_id: str
    session_summaries: list[str] = []


@router.post("/chat")
async def therapy_chat(data: TherapyRequest):

    # ======= MONITOR: Request received ============
    await log_to_monitor("THERAPY_AGENT_REQUEST", {
        "user_id": data.user_id,
        "session_id": data.session_id,
        "user_query": data.user_query,
        "depression_level": data.depression_level,
    })

    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]

    history_records = get_user_therapy_history(db, data.user_id)
    recent_history = "\n".join(
        [
            f"{h['therapy_name']} on {h['date']} (duration {h['duration']} mins)"
            for h in history_records
        ]
    ) if history_records else "No prior therapies found."

    therapy_suggestion = get_therapy_recommendation(
        db, data.depression_level, history_records
    )

    therapy_name = therapy_suggestion.get("name")
    therapy_id = therapy_suggestion.get("id")
    therapy_path = therapy_suggestion.get("path", None)

    # ======= MONITOR: Therapy suggested ============
    await log_to_monitor("THERAPY_SUGGESTION_COMPUTED", {
        "user_id": data.user_id,
        "session_id": data.session_id,
        "therapy_id": therapy_id,
        "therapy_name": therapy_name,
    })

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

    suggestion_phrases = [
        f"start the {therapy_name.lower()} therapy",
        f"try the {therapy_name.lower()} therapy",
        f"{therapy_name.lower()} therapy now",
    ]

    is_therapy_suggested = any(p in reply_text for p in suggestion_phrases)

    action_pattern = r"action\s*:\s*start[_\- ]therapy\s*:\s*([A-Za-z0-9]+)"
    match = re.search(action_pattern, reply_text, re.IGNORECASE)
    action_detected = match.group(1).strip() if match else None

    # ===============================
    # ACTION DETECTED → start therapy
    # ===============================
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

        # ======= MONITOR: Therapy started ============
        await log_to_monitor("THERAPY_STARTED", {
            "user_id": data.user_id,
            "session_id": data.session_id,
            "therapy_id": therapy_id,
            "therapy_name": therapy_name,
        })

        is_therapy_suggested = False

    client.close()

    # ======= MONITOR: Response sent ============
    await log_to_monitor("THERAPY_AGENT_RESPONSE", {
        "user_id": data.user_id,
        "session_id": data.session_id,
        "response": response.content,
        "action": "START_THERAPY" if action_detected else None,
        "therapy_id": therapy_id if action_detected else None,
    })

    return {
        "response": response.content.replace("ACTION:START_THERAPY", "").strip(),
        "action": "START_THERAPY" if action_detected else None,
        "therapy_id": therapy_id if action_detected else None,
        "therapy_name": therapy_name if action_detected else None,
        "therapy_path": therapy_path,
        "isTherapySuggested": is_therapy_suggested,
        "therapySuggestion": {
            "id": therapy_id,
            "name": therapy_name,
            "path": therapy_path,
        } if is_therapy_suggested else None,
    }
