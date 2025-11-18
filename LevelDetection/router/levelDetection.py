from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from ..service.levelDetection import detect_from_history_and_summary
from datetime import datetime
import requests
import key_param

router = APIRouter()

EmotionLabel = Literal["happy", "neutral", "sad", "angry", "fearful"]
DepressionDetectedLabel = Literal["Depression Signs Detected", "No Depression Signs Detected"]

class DetectionRequest(BaseModel):
    history: str = Field(..., description="Raw chat history (plain text)")
    summaries: List[str] = Field(default_factory=list)
    summary: Optional[str] = None  # legacy single-summary field
    user_id: int
    session_id: int

class DetectionResponse(BaseModel):
    # NEW: human-readable label + confidence that detection IS present
    depression_label: DepressionDetectedLabel
    depression_confidence_detected: int  # 0..100, confidence that signs ARE detected

    emotion: EmotionLabel
    emotion_confidence: int  # 0..100
    rationale: str

@router.post("/detect", response_model=DetectionResponse)
async def detect(req: DetectionRequest):
    try:
        # unify shapes (keep support for old clients)
        all_summaries = list(req.summaries or [])
        if req.summary:
            all_summaries.append(req.summary)

        # join summaries to one string for the service (or None if empty)
        joined_summary = " ".join(s for s in all_summaries if s).strip() or None

        result = detect_from_history_and_summary(req.history, joined_summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    dep = result["depression"]
    emo = result["emotion"]

    # Send activity log to Monitor Agent (NO FEEDBACK)
    try:
        monitor_payload = {
            "agent_name": "classifier",
            "user_id": req.user_id,
            "session_id": req.session_id,
            "input_data": {
                "history": req.history,
                "summaries": all_summaries,
            },
            "output_data": {
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
            json=monitor_payload,
            timeout=10
        )
        print("📌 Classifier logged activity to Monitor Agent")
    except Exception as e:
        print("⚠️ Failed to log classifier activity:", e)

    # -------------------------------------------------

    return DetectionResponse(
        depression_label=dep["label"],
        depression_confidence_detected=dep["confidence_detected"],
        emotion=emo["label"],
        emotion_confidence=emo["confidence"],
        rationale=result.get("rationale", ""),
    )
