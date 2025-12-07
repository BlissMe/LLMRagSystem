from fastapi import APIRouter, Query
from pydantic import BaseModel
from datetime import datetime
from pymongo import MongoClient
from openai import OpenAI
import key_param
from typing import List, Optional

router = APIRouter(prefix="/monitor-agent", tags=["Monitor Agent"])

clientAI = OpenAI(api_key=key_param.openai_api_key)


# -------------------------------
# MODELS
# -------------------------------

class AgentActivity(BaseModel):
    agent_name: str
    user_id: int
    session_id: int
    input_data: dict
    output_data: dict
    timestamp: datetime = datetime.utcnow()

class MonitorFeedbackRequest(BaseModel):
    recent_activities: list[AgentActivity]


# -------------------------------
# AI AGENT FUNCTION
# -------------------------------

async def run_monitoring_agent(activity: AgentActivity):
    """
    Sends agent activity to an OpenAI monitoring agent that
    evaluates risk level, identifies anomalies, and summarizes behavior.
    """

    prompt = f"""
    You are the BLISS-ME Monitoring AI Agent.
    Your job is to analyze a single agent activity event.

    Provide output in strict JSON:
    {{
        "summary": "...",
        "risk_level": "low | medium | high",
        "possible_issue": "...",
        "anomaly_detected": true | false
    }}

    Activity Data:
    - Agent: {activity.agent_name}
    - User ID: {activity.user_id}
    - Session ID: {activity.session_id}

    Input:
    {activity.input_data}

    Output:
    {activity.output_data}
    """

    response = clientAI.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=300
    )

    ai_output_text = response.output_text
    return eval(ai_output_text)  # JSON from AI agent


# -------------------------------
# ENDPOINT :: TRACK AGENT ACTIVITY
# -------------------------------

@router.post("/track-activity")
async def track_agent_activity(activity: AgentActivity):
    """
    Logs an agent's activity AND uses an AI agent to analyze it.
    """

    # 1. Run AI agent reasoning
    monitor_result = await run_monitoring_agent(activity)

    # 2. Store in DB
    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]
    collection = db["agent_activity_logs"]

    final_record = {
        **activity.dict(),
        "monitor_summary": monitor_result["summary"],
        "risk_level": monitor_result["risk_level"],
        "possible_issue": monitor_result["possible_issue"],
        "anomaly_detected": monitor_result["anomaly_detected"],
    }

    collection.insert_one(final_record)
    client.close()

    return {
        "status": "logged",
        "agent": activity.agent_name,
        "monitoring": monitor_result
    }


# -------------------------------
# ENDPOINT :: GET SESSION EVENTS
# -------------------------------

@router.get("/get-session-events")
async def get_session_events(user_id: int = Query(...)):
    """
    Retrieve all events for a given user.
    Includes AI-monitor summaries.
    """
    client = MongoClient(key_param.MONGO_URI)
    db = client["blissMe"]
    collection = db["agent_activity_logs"]

    cursor = collection.find({"user_id": user_id}).sort("timestamp", 1)

    events = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()
        events.append(doc)

    client.close()

    return {
        "user_id": user_id,
        "events": events
    }
