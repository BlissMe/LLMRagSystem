from fastapi import APIRouter
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from datetime import datetime
import key_param
from .utils.therapy_selector import get_therapy_recommendation
from .utils.history_tracker import save_therapy_history, get_user_therapy_history

router = APIRouter(prefix="/therapy-agent", tags=["Therapy Agent"])

class TherapyRequest(BaseModel):
    user_query: str
    depression_level: str
    user_id: str
    session_id: str

@router.post("/chat")
async def therapy_chat(data: TherapyRequest):
    """
    Therapy Agent main route:
    Handles chat, suggests therapies, tracks user progress.
    """

    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]

    # 🩺 Fetch therapy history
    history_records = get_user_therapy_history(db, data.user_id)
    recent_history = "\n".join(
        [f"{h['therapy_name']} on {h['date']} (duration {h['duration']} mins)" for h in history_records]
    ) if history_records else "No prior therapies found."

    # 🧘 Suggest new therapy
    therapy_suggestion = get_therapy_recommendation(db, data.depression_level, history_records)

    # 🧠 Base prompt
    prompt = f"""
You are a friendly therapy assistant designed to support users with {data.depression_level} depression.
You talk like a warm and caring friend. don't always suggest therapies, suggestwhen appropriate based on the user's emotional state other times keep chatting as caring friend BUT your main duty is suggesting therapies.

Current user history:
{recent_history}

If the user has moderate or minimal depression, suggest small helpful activities or therapies from the system.
Therapies can include relaxation breathing, mindfulness, journaling, or gratitude reflection.
don't use log sentences. keep it short and simple.
don't mention about depression level or depression to the user.


If a therapy matches one from the system, gently ask:
"Would you like to start the {therapy_suggestion['name']} therapy now?"

If the user agrees, return:
ACTION:START_THERAPY:{therapy_suggestion['id']}

Otherwise, continue gentle conversation and emotional support.

User message: "{data.user_query}"
"""

    bot = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=key_param.openai_api_key)
    response = bot.invoke([{"role": "user", "content": prompt}])

    reply_text = response.content.strip()
    action_detected = None
    is_therapy_suggested = False  

    
    if f"start the {therapy_suggestion['name']} therapy" in reply_text.lower():
        is_therapy_suggested = True

   
    if "ACTION:START_THERAPY" in reply_text:
        action_detected = reply_text.split("ACTION:START_THERAPY:")[-1].strip()
        save_therapy_history(
            db,
            data.user_id,
            data.session_id,
            therapy_suggestion["name"],
            therapy_suggestion["id"]
        )
        is_therapy_suggested = False 


    client.close()

    return {
    "response": reply_text.replace("ACTION:START_THERAPY", "").strip(),
    "action": "START_THERAPY" if action_detected else None,
    "therapy_id": action_detected,
    "therapy_name": therapy_suggestion["name"] if action_detected else None,
    "therapy_path": therapy_suggestion.get("path"),  
    "isTherapySuggested": is_therapy_suggested, 
}

