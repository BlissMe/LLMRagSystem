from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from .db_handler import save_report, get_reports_by_user, get_user_summary

router = APIRouter(
    prefix="/monitor",
    tags=["Monitoring Agent"]
)

class Report(BaseModel):
    agent_name: str
    user_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    data: dict

@router.post("/report")
async def receive_report(report: Report):
    await save_report(report.dict())
    return {"message": "Report stored in MongoDB"}

@router.get("/summary/{user_id}")
async def get_user_logs(user_id: str):
    reports = await get_reports_by_user(user_id)
    if not reports:
        raise HTTPException(status_code=404, detail="No reports found for user.")
    return {"user_id": user_id, "reports": reports}

@router.get("/summary/{user_id}/aggregate")
async def get_user_aggregate(user_id: str):
    summary = await get_user_summary(user_id)
    if not summary:
        raise HTTPException(status_code=404, detail="No summary data available.")
    return {
        "user_id": user_id,
        "average_depression_confidence": round(summary.get("avg_confidence", 0), 2),
        "last_therapy": summary.get("last_therapy"),
        "last_assessment": summary.get("last_assessment"),
        "total_reports": summary.get("total_reports", 0)
    }
