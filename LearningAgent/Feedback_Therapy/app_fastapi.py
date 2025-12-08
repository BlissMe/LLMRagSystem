# app_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import compute_therapy_report_for_user
from conclusion_utils import generate_conclusion

from apscheduler.schedulers.background import BackgroundScheduler
from weekly_learning_job import run_weekly_learning

app = FastAPI(title="Therapy Recommender")

class UserRequest(BaseModel):
    user_id: int

@app.post("/report")
def get_report(req: UserRequest):
    report = compute_therapy_report_for_user(req.user_id)
    if not report.get("therapies"):
        raise HTTPException(status_code=404, detail="No therapy history for this user.")
    conclusion = generate_conclusion(report)
    return {"report": report, "conclusion": conclusion}


scheduler = BackgroundScheduler()

@app.on_event("startup")
def start_scheduler():
    """
    Start the background scheduler and register the weekly job.
    This runs once FastAPI starts.
    """
    # Add weekly job if not already registered
    if not scheduler.get_job("weekly_learning"):
        # Every Sunday at 00:30 (server local time)
        scheduler.add_job(
            run_weekly_learning,
            "cron",
            day_of_week="sun",
            hour=0,
            minute=30,
            id="weekly_learning",
            replace_existing=True,
        )

    if not scheduler.running:
        scheduler.start()
        print(" APScheduler started: weekly_learning job registered.")


@app.on_event("shutdown")
def shutdown_scheduler():
    """
    Clean shutdown for the scheduler when FastAPI stops.
    """
    if scheduler.running:
        scheduler.shutdown()
        print("APScheduler shut down.")