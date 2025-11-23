from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
import key_param
from fastapi import Query
from typing import List
router = APIRouter(prefix="/monitor-agent", tags=["Monitor Agent"])

class AgentActivity(BaseModel):
    agent_name: str  # "chat", "classifier", "therapy"
    user_id: int
    session_id: int
    input_data: dict
    output_data: dict
    timestamp: datetime = datetime.utcnow()

class MonitorFeedbackRequest(BaseModel):
    recent_activities: list[AgentActivity]

@router.post("/track-activity")
async def track_agent_activity(activity: AgentActivity):
    """
    Logs an agent's activity in MongoDB for monitoring.
    """
    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]
    collection = db["agent_activity_logs"]
    collection.insert_one(activity.dict())
    client.close()
    return {"status": "logged", "agent": activity.agent_name}

@router.get("/get-session-events")
async def get_session_events(user_id: int = Query(...)):
    """
    Retrieve all events for a given user.
    """
    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]
    collection = db["agent_activity_logs"]

    events_cursor = collection.find(
        {"user_id": user_id}
    ).sort("timestamp", 1)

    events = []
    for event in events_cursor:
        event["_id"] = str(event["_id"])
        if isinstance(event.get("timestamp"), datetime):
            event["timestamp"] = event["timestamp"].isoformat()
        events.append(event)

    client.close()
    return {
        "user_id": user_id,
        "events": events
    }
