from fastapi import APIRouter, Query, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
from pymongo import MongoClient
import key_param
from typing import List
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

router = APIRouter(prefix="/monitor-agent", tags=["Monitor Agent"])


# -------------------------------
# MODELS
# -------------------------------

class AgentActivity(BaseModel):
    agent_name: str
    user_id: int
    session_id: int
    input_data: dict
    output_data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)  # ✅ fixed timestamp bug


class MonitorFeedbackRequest(BaseModel):
    recent_activities: List[AgentActivity]


# -------------------------------
# AI MONITORING FUNCTION
# -------------------------------

async def run_monitoring_agent(activity: AgentActivity):

    class MonitorResult(BaseModel):
        summary: str
        risk_level: str
        possible_issue: str
        anomaly_detected: bool

    parser = JsonOutputParser(pydantic_object=MonitorResult)

    prompt = ChatPromptTemplate.from_template("""
You are the BLISS-ME Monitoring AI Agent.
Analyze the following agent activity and produce STRICT JSON ONLY.

AGENT NAME: {agent_name}
USER ID: {user_id}
SESSION ID: {session_id}

INPUT DATA:
{input_data}

OUTPUT DATA:
{output_data}

RISK LEVEL RULES:
- "low": Normal behaviour.
- "medium": Minor anomalies.
- "high": Critical anomalies.

ANOMALY RULES:
- anomaly_detected = true if ANY suspicious pattern exists.
- anomaly_detected = false otherwise.

JSON FORMAT REQUIRED:
{{
    "summary": "short description",
    "risk_level": "low | medium | high",
    "possible_issue": "string or 'none'",
    "anomaly_detected": true or false
}}

Return ONLY JSON.
""")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=key_param.openai_api_key,
    )

    chain = prompt | llm | parser

    result = await chain.ainvoke({
        "agent_name": activity.agent_name,
        "user_id": activity.user_id,
        "session_id": activity.session_id,
        "input_data": json.dumps(activity.input_data, indent=2),
        "output_data": json.dumps(activity.output_data, indent=2),
    })

    return result


# -------------------------------
# ✅ BACKGROUND MONITORING WORKER (NON-BLOCKING)
# -------------------------------

async def process_monitoring(activity: AgentActivity):

    monitor_result = await run_monitoring_agent(activity)

    client = MongoClient(key_param.MONGO_URI_KB)
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


# -------------------------------
# ✅ ENDPOINT :: TRACK AGENT ACTIVITY (TIMEOUT FIXED)
# -------------------------------

@router.post("/track-activity")
async def track_agent_activity(
    activity: AgentActivity,
    background_tasks: BackgroundTasks
):
    """
    Logs agent activity and runs AI monitoring in background to avoid timeout.
    """

    # ✅ Run monitoring asynchronously (no blocking)
    background_tasks.add_task(process_monitoring, activity)

    # ✅ Immediate response (prevents Therapy Agent timeout)
    return {
        "status": "queued",
        "agent": activity.agent_name
    }


# -------------------------------
# ENDPOINT :: GET SESSION EVENTS
# -------------------------------

@router.get("/get-session-events")
async def get_session_events(user_id: int = Query(...)):

    client = MongoClient(key_param.MONGO_URI_KB)
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
