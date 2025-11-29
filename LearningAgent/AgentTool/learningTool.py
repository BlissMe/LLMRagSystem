# learningTool.py
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from .http_client import (
    ASSESSMENT_BASE, CLASSIFIER_BASE, THERAPY_BASE, MONITOR_BASE,
    post_json, get_json, report
)
from .http_client import report
from pydantic import BaseModel
# --------- Input Schemas ---------
class SummarizeInput(BaseModel):
    user_id: str = Field(..., description="User ID")
    history: str = Field(..., description="Plain chat history")

class DetectInput(BaseModel):
    user_id: str
    history: str
    summaries: Optional[list[str]] = []

class TherapyInput(BaseModel):
    user_id: str
    session_id: str
    user_query: str
    depression_level: str
    session_summaries: Optional[list[str]] = []

class AggregateInput(BaseModel):
    user_id: str

# --------- Tools ---------
class SummarizeTool(BaseTool):
    name = "summarize_history"
    description = "POST /summarize to assessment agent to get a single summary string"
    args_schema: Type[BaseModel] = SummarizeInput

    def _run(self, user_id: str, history: str):
        result = post_json(f"{ASSESSMENT_BASE}/summarize", {"history": history})
        # normalized report
        data = {
            "type": "assessment.summary",
            "summary": result.get("summary", ""),
            "source": "summarize_history",
            "tokens_used": result.get("tokens_used", 0)
        }
        report("assessment", user_id, data=data)
        return result

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError()

class DetectTool(BaseTool):
    name = "detect_depression_emotion"
    description = "POST /detect to classifier agent with history and summaries"
    args_schema: Type[BaseModel] = DetectInput

    def _run(self, user_id: str, history: str, summaries: Optional[list[str]] = None):
        payload = {"history": history}
        if summaries:
            payload["summaries"] = summaries
        result = post_json(f"{CLASSIFIER_BASE}/detect", payload)

        data = {
            "type": "classifier.detection",
            "depression_label": result.get("depression_label"),
            "depression_confidence_detected": result.get("depression_confidence_detected"),
            "emotion": result.get("emotion"),
            "emotion_confidence": result.get("emotion_confidence"),
            "inputs": {
                "has_summaries": bool(summaries),
                "history_chars": len(history or "")
            }
        }
        report("classifier", user_id, data=data)
        return result

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError()

class TherapyChatTool(BaseTool):
    name = "therapy_chat"
    description = "POST /therapy-agent/chat to suggest/launch therapy and log outcome"
    args_schema: Type[BaseModel] = TherapyInput

    def _run(self, user_id: str, session_id: str, user_query: str, depression_level: str, session_summaries: Optional[list[str]] = None):
        payload = {
            "user_query": user_query,
            "depression_level": depression_level,
            "user_id": user_id,
            "session_id": session_id,
            "session_summaries": session_summaries or []
        }
        result = post_json(f"{THERAPY_BASE}/therapy-agent/chat", payload)

        data = {
            "type": "therapy.chat",
            "request": {
                "user_query": user_query,
                "depression_level": depression_level,
                "session_summaries": session_summaries or []
            },
            "response": {
                "text": result.get("response"),
                "action": result.get("action"),
                "therapy_id": result.get("therapy_id"),
                "therapy_name": result.get("therapy_name"),
                "isTherapySuggested": result.get("isTherapySuggested"),
            }
        }
        report("therapy", user_id, data=data, session_id=session_id)
        return result

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError()

class GetAggregateTool(BaseTool):
    name = "get_user_summary"
    description = "GET /monitor/summary/{user_id}/aggregate to fetch aggregate monitoring summary"
    args_schema: Type[BaseModel] = AggregateInput

    def _run(self, user_id: str):
        return get_json(f"{MONITOR_BASE}/monitor/summary/{user_id}/aggregate")

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError()
    


class PHQ9Input(BaseModel):
    user_id: str
    session_id: str | None = None
    items: list[dict]  # [{id, text, answer_choice}]

class SubmitPHQ9Tool(BaseTool):
    name = "submit_phq9"
    description = "Compute PHQ-9 score from answers and log it to monitoring"
    args_schema: Type[BaseModel] = PHQ9Input

    def _run(self, user_id: str, items: list[dict], session_id: str | None = None):
        from .utils.phq9 import score_phq9
        payload = score_phq9(items)
        # Just log; no downstream service required
        report("assessment", user_id, data=payload, session_id=session_id)
        return {"ok": True, **payload}

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError()
