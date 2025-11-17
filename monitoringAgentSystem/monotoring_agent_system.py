# monitorAgent/router/monitorAgent.py
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


@router.post("/analyze-feedback")
async def analyze_feedback(req: MonitorFeedbackRequest):
    """
    Analyzes multiple agents’ behaviors and generates feedback using an LLM.
    """
    activities_summary = "\n".join([
        f"{a.timestamp} | {a.agent_name.upper()} | Input: {a.input_data} | Output: {a.output_data}"
        for a in req.recent_activities
    ])

    prompt = f"""
You are a monitoring supervisor analyzing AI agent behavior.

Here are the latest activities of agents:
{activities_summary}

Your job:
- Detect inconsistencies or performance issues.
- Identify agents that might have repeated or conflicting outputs.
- Give feedback in short, clear bullet points.

Provide feedback for each agent.
"""

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=key_param.openai_api_key)
    response = llm.invoke([{"role": "user", "content": prompt}])

    return {"feedback": response.content.strip()}

@router.get("/get-session-events")
async def get_session_events(user_id: int = Query(...), session_id: int = Query(...)):
    """
    Retrieve all events for a given user and session.
    """
    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]
    collection = db["agent_activity_logs"]

    events_cursor = collection.find(
        {"user_id": user_id, "session_id": session_id}
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
        "session_id": session_id,
        "events": events
    }
