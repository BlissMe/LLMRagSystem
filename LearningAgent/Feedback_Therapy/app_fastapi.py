# app_fastapi.py
from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel
from .pipeline import compute_therapy_report_for_user
from .conclusion_utils import generate_conclusion

router = APIRouter()
app = FastAPI(title="Therapy Recommender")

class UserRequest(BaseModel):
    user_id: int

@router.post("/therapyFeedbackReport")
def get_report(req: UserRequest):
    report = compute_therapy_report_for_user(req.user_id)
    if not report.get("therapies"):
        raise HTTPException(status_code=404, detail="No therapy history for this user.")
    conclusion = generate_conclusion(report)
    return {"report": report, "conclusion": conclusion}
