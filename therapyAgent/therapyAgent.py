from fastapi import APIRouter,BackgroundTasks
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
MONITOR_URL = f"{key_param.llm_base}/monitor-agent/track-activity"


# =======================
# MODELS
# =======================
class TherapyRequest(BaseModel):
    user_query: str
    depression_level: str
    user_id: str
    session_id: str
    session_summaries: list[str] = []
    therapy_feedback_conclusion: str | None = None


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
        requests.post(MONITOR_URL, json=payload, timeout=0.2)
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
async def save_therapy_feedback( data: TherapyFeedback,
    background_tasks: BackgroundTasks):
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
    background_tasks.add_task(
        send_end_therapy_event,
        data.user_id,
        data.session_id,
        data.therapy_id,
        therapy_name,
        data.feedback,
        data.duration
    )

    client.close()

    return {"success": True, "message": "Therapy feedback saved"}

# =======================
# CHAT ENDPOINT
# ======================

@router.post("/chat")
async def therapy_chat(data: TherapyRequest, background_tasks: BackgroundTasks):

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
    feedback_summary = (
    f"\n\nTherapy feedback analysis:\n{data.therapy_feedback_conclusion}\n"
    if data.therapy_feedback_conclusion
    else ""
)

    # =======================
    #       BASE PROMPT
    # =======================
    prompt = f"""
You are a warm, friendly therapy assistant. 
Your main job is to support the user emotionally AND suggest a therapy when APPROPRIATE.
Don't suggest therapies every time—only when it fits naturally in the conversation.
If the user seems distressed, PRIORITIZE empathy and understanding first.
DON'T push therapies if the user is not open to it.

The user has previous therapy feedback and usage history. 
You can consider this feedback when recommending therapies.

Therapy outcome summary (very important):
{data.therapy_feedback_conclusion or "No feedback summary available."}

Rules:
- Keep responses short, caring, simple.
- Never mention depression level.
- Suggest therapies gently when appropriate.
- If the user seems distressed, PRIORITIZE empathy and understanding first.
- If the user declines a therapy, respect their choice and continue the chat supportively.
- Don't suggest therapies every time—only when it fits naturally in the conversation.
- also can ask about previously done therapies, which those helped or not.
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

        background_tasks.add_task(
        send_monitor_event,
        "THERAPY_STARTED",
        {
            "input": {},
            "output": {
                "therapy_id": therapy_id,
                "therapy_name": therapy_name
            }
        },
        data.user_id,     # no int()
        data.session_id  # no int()
    )
        is_therapy_suggested = False  # because user already started


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

class ManualStartRequest(BaseModel):
    user_id: int
    session_id: int
    therapy_id: str
    therapy_name: str

from fastapi import BackgroundTasks

@router.post("/end-start")
async def manual_start_therapy(
    data: ManualStartRequest,
    background_tasks: BackgroundTasks
):
    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]

    # ----------------------
    # Save therapy history
    # ----------------------
    save_therapy_history(
        db,
        data.user_id,
        data.session_id,
        data.therapy_name,
        data.therapy_id,
        duration=None,
        feedback=None
    )

    # ----------------------
    # Run monitor logs in background (NON-BLOCKING)
    # ----------------------
    background_tasks.add_task(
        send_monitor_event,
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

    background_tasks.add_task(
        send_therapy_progress_event,
        data.user_id,
        data.session_id,
        data.therapy_id,
        data.therapy_name,
        0.0
    )

    # ----------------------
    # Close DB immediately
    # ----------------------
    client.close()

    # ----------------------
    #  Fast response
    # ----------------------
    return {
        "success": True,
        "message": "Therapy session started",
        "therapy_id": data.therapy_id,
        "therapy_name": data.therapy_name
    }
