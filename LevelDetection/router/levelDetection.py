from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from ..service.levelDetection import detect_from_history_and_summary
from datetime import datetime
import requests
import key_param

router = APIRouter()

# Define labels
EmotionLabel = Literal["happy", "neutral", "sad", "angry", "fearful"]
DepressionDetectedLabel = Literal["Depression Signs Detected", "No Depression Signs Detected"]

# Request model
class DetectionRequest(BaseModel):
    history: str = Field(..., description="Raw chat history (plain text)")
    summaries: List[str] = Field(default_factory=list)
    summary: Optional[str] = None  # legacy single-summary field
    user_id: int
    session_id: int

# Response model
class DetectionResponse(BaseModel):
    depression_label: DepressionDetectedLabel
    depression_confidence_detected: int  # 0..100
    emotion: EmotionLabel
    emotion_confidence: int  # 0..100
    rationale: str

@router.post("/detect", response_model=DetectionResponse)
async def detect(req: DetectionRequest):
    try:
        # Combine summaries
        all_summaries = list(req.summaries or [])
        if req.summary:
            all_summaries.append(req.summary)

        joined_summary = " ".join(s for s in all_summaries if s).strip() or None

        # Perform detection
        result = detect_from_history_and_summary(req.history, joined_summary)
        dep = result["depression"]
        emo = result["emotion"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ---------------------------
    # 1️⃣ Log depression detection
    # ---------------------------
    try:
        depression_event = {
            "agent_name": "classifier",
            "user_id": req.user_id,
            "session_id": req.session_id,
            "input_data": {
                "history": req.history,
                "summaries": all_summaries,
            },
            "output_data": {
                "event": "depression_detection",
                "depression_label": dep["label"],
                "depression_confidence_detected": dep["confidence_detected"],
                "emotion": emo["label"],
                "emotion_confidence": emo["confidence"],
                "rationale": result.get("rationale", "")
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        requests.post(
            "http://localhost:8000/monitor-agent/track-activity",
            json=depression_event,
            timeout=10
        )
        print("📌 Logged depression detection event")
    except Exception as e:
        print("⚠️ Failed to log depression detection:", e)

    # ---------------------------
    # 2️⃣ Log session end
    # ---------------------------
    try:
        session_end_event = {
            "agent_name": "classifier",
            "user_id": req.user_id,
            "session_id": req.session_id,
            "input_data": {},
            "output_data": {
                "event": "session_end"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        requests.post(
            "http://localhost:8000/monitor-agent/track-activity",
            json=session_end_event,
            timeout=10
        )
        print("📌 Logged session end event")
    except Exception as e:
        print("⚠️ Failed to log session end:", e)

    # ---------------------------
    # Return detection response
    # ---------------------------
    return DetectionResponse(
        depression_label=dep["label"],
        depression_confidence_detected=dep["confidence_detected"],
        emotion=emo["label"],
        emotion_confidence=emo["confidence"],
        rationale=result.get("rationale", ""),
    )
