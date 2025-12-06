# monitor_ai_agent.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from typing import List, Optional
from langchain_openai import ChatOpenAI
import key_param
import re
import json

router = APIRouter(prefix="/monitor-agent", tags=["Monitor Agent"])

# ---------- Models ----------
class AgentActivity(BaseModel):
    agent_name: str  # "chat", "classifier", "therapy", etc.
    user_id: int
    session_id: int
    input_data: dict
    output_data: dict
    timestamp: Optional[datetime] = None

    def dict_with_timestamp(self):
        d = self.dict()
        if d.get("timestamp") is None:
            d["timestamp"] = datetime.utcnow()
        return d

class MonitorFeedbackRequest(BaseModel):
    recent_activities: List[AgentActivity]

class AnalysisResult(BaseModel):
    summary: str
    issues_detected: List[str]
    therapy_progress: Optional[str]
    recommendations: List[str]
    raw_llm_output: Optional[str] = None

# ---------- DB helpers ----------
def get_mongo_client():
    return MongoClient(key_param.MONGO_URI)

def ensure_indexes():
    client = get_mongo_client()
    db = client["blissMe"]
    coll = db["agent_activity_logs"]
    coll.create_index([("user_id", ASCENDING), ("session_id", ASCENDING)])
    coll.create_index([("timestamp", ASCENDING)])
    client.close()

ensure_indexes()

def fetch_recent_activities_from_db(user_id: int, limit: int = 100):
    client = get_mongo_client()
    db = client["blissMe"]
    coll = db["agent_activity_logs"]
    cursor = coll.find({"user_id": user_id}).sort("timestamp", ASCENDING).limit(limit)
    events = []
    for ev in cursor:
        ev["_id"] = str(ev["_id"])
        ts = ev.get("timestamp")
        if isinstance(ts, datetime):
            ev["timestamp"] = ts.isoformat()
        events.append(ev)
    client.close()
    return events

# ---------- Simple rule-based anomaly checks ----------
def simple_anomaly_checks(events: List[dict]) -> List[str]:
    issues = []
    if not events:
        return ["no_events_found"]

    # repeated identical outputs (same output_data repeated consecutively)
    repeat_count = 0
    last_output = None
    for ev in events:
        out = re.sub(r"\s+", " ", str(ev.get("output_data", ""))).strip()
        if out == last_output:
            repeat_count += 1
        else:
            repeat_count = 0
        last_output = out
        if repeat_count >= 2:
            issues.append("repeated_same_output_detected")
            break

    # high error rate
    errors = [e for e in events if "error" in str(e.get("output_data", "")).lower() or "failed" in str(e.get("output_data", "")).lower()]
    if len(errors) >= max(1, len(events) * 0.2):
        issues.append("high_error_rate")

    # long gaps between events (>1 hour)
    times = [e.get("timestamp") for e in events if e.get("timestamp")]
    parsed_times = []
    for t in times:
        try:
            parsed_times.append(datetime.fromisoformat(t))
        except Exception:
            continue
    if len(parsed_times) >= 2:
        parsed_times.sort()
        max_gap = max((parsed_times[i+1] - parsed_times[i]).total_seconds() for i in range(len(parsed_times)-1))
        if max_gap > 3600:
            issues.append("long_gap_between_events")

    # therapy stagnation heuristic
    therapy_events = [e for e in events if e.get("agent_name") == "therapy"]
    if therapy_events and all(not e.get("output_data", {}).get("therapy_progress") for e in therapy_events):
        issues.append("therapy_no_progress_detected")

    return list(set(issues))

# ---------- LLM integration ----------
def build_analysis_prompt(events: List[dict]) -> str:
    snippet_lines = []
    for e in events[-50:]:
        ts = e.get("timestamp", "")
        agent = e.get("agent_name", "")
        inp = e.get("input_data", {})
        out = e.get("output_data", {})
        snippet_lines.append(f"{ts} | {agent} | IN: {json.dumps(inp, default=str)} | OUT: {json.dumps(out, default=str)}")
    snippet = "\n".join(snippet_lines)

    prompt = f"""
You are a monitoring assistant for the BlissMe multi-agent system.
Analyze the following chronological activity log (most recent last). Be concise and return a JSON object with keys:
- summary: single-paragraph summary
- issues_detected: list of short issue strings
- therapy_progress: short description about therapy progression or 'unknown'
- recommendations: list of concrete recommendations (one-liners)
Return only valid JSON.

Activity log:
{snippet}

Rules:
1) If you detect repeated failures, say so in issues_detected.
2) If therapy shows improvement or worsening mention it briefly.
3) If you recommend a follow-up action that can be auto-triggered (e.g. restart agent, re-run classifier, escalate to human), put it as "AUTO_TRIGGER: <action>" in recommendations.
4) Keep JSON keys exactly as requested.
"""
    return prompt

def call_llm_for_analysis(prompt: str) -> str:
    if not key_param.openai_api_key:
        raise RuntimeError("OPENAI_KEY not set in environment (key_param.openai_api_key)")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=key_param.openai_api_key)
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)

def parse_llm_json(raw_text: str) -> dict:
    text = raw_text.strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        candidate = text[start:end]
        return json.loads(candidate)
    except Exception:
        return {"summary": "", "issues_detected": [], "therapy_progress": "unknown", "recommendations": [], "raw": raw_text}

# ---------- Endpoints ----------
@router.post("/track-activity")
async def track_agent_activity(activity: AgentActivity):
    client = get_mongo_client()
    db = client["blissMe"]
    collection = db["agent_activity_logs"]
    doc = activity.dict_with_timestamp()
    # ensure timestamp is ISO string to keep frontend simple
    if isinstance(doc["timestamp"], datetime):
        doc["timestamp"] = doc["timestamp"].isoformat()
    collection.insert_one(doc)
    client.close()
    return {"status": "logged", "agent": activity.agent_name}

@router.get("/get-session-events")
async def get_session_events(user_id: int):
    events = fetch_recent_activities_from_db(user_id=user_id, limit=500)
    return {"user_id": user_id, "events": events}

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_agent_behaviour(request: MonitorFeedbackRequest):
    events = []
    for e in request.recent_activities:
        d = e.dict_with_timestamp()
        if isinstance(d.get("timestamp"), datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        events.append(d)

    rule_issues = simple_anomaly_checks(events)
    prompt = build_analysis_prompt(events)
    raw_llm = call_llm_for_analysis(prompt)
    llm_json = parse_llm_json(raw_llm)

    issues = list({*rule_issues, *llm_json.get("issues_detected", [])})

    result = {
        "summary": llm_json.get("summary", ""),
        "issues_detected": issues,
        "therapy_progress": llm_json.get("therapy_progress", "unknown"),
        "recommendations": llm_json.get("recommendations", []),
        "raw_llm_output": raw_llm
    }
    return result

@router.post("/analyze-from-db/{user_id}", response_model=AnalysisResult)
async def analyze_from_db(user_id: int, limit: int = 200):
    events = fetch_recent_activities_from_db(user_id=user_id, limit=limit)
    req_activities = []
    for e in events:
        req_activities.append(AgentActivity(
            agent_name=e.get("agent_name", "unknown"),
            user_id=e.get("user_id"),
            session_id=e.get("session_id", 0),
            input_data=e.get("input_data", {}) or {},
            output_data=e.get("output_data", {}) or {},
            timestamp=e.get("timestamp")
        ))
    return await analyze_agent_behaviour(MonitorFeedbackRequest(recent_activities=req_activities))

@router.post("/auto-trigger/{user_id}")
async def analyze_and_auto_trigger(user_id: int, limit: int = 200):
    analysis = await analyze_from_db(user_id=user_id, limit=limit)
    triggers = [r for r in analysis.recommendations if isinstance(r, str) and r.startswith("AUTO_TRIGGER:")]
    inserted = []
    if triggers:
        client = get_mongo_client()
        db = client["blissMe"]
        coll = db["agent_activity_logs"]
        for t in triggers:
            action = t.split("AUTO_TRIGGER:", 1)[1].strip()
            follow_event = {
                "agent_name": "monitor_agent",
                "user_id": user_id,
                "session_id": 0,
                "input_data": {"auto_trigger_action": action},
                "output_data": {"status": "pending"},
                "timestamp": datetime.utcnow().isoformat()
            }
            coll.insert_one(follow_event)
            inserted.append(action)
        client.close()
    return {"analysis_summary": analysis.summary, "triggers_inserted": inserted}
