# weekly_learning_job.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pymongo import MongoClient
import key_param
from pipeline import compute_therapy_report_for_user
from conclusion_utils import generate_conclusion
from kb_writer import save_learning_summary_to_kb

# connect to main app DB (blissMe)
app_client = MongoClient(getattr(key_param, "MONGO_URI", key_param.MONGO_URI))
app_db = app_client["blissMe"]
therapy_history_collection = app_db["TherapyHistory"]


def run_weekly_learning():
    """
    1. Get all distinct user_ids that have therapy history.
    2. For each user:
       - compute therapy report
       - generate conclusion
       - save top therapy + conclusion into KB (no user IDs).
    """
    user_ids = therapy_history_collection.distinct("user_id")
    total_saved = 0

    for uid in user_ids:
        try:
            user_id = int(uid)
        except Exception:
            user_id = uid

        report = compute_therapy_report_for_user(user_id)
        therapies = report.get("therapies", [])
        if not therapies:
            continue

        # get final text summary
        conclusion = generate_conclusion(report)

        # you said it's ok to track therapy, just not user
        top_therapy_id = therapies[0].get("therapy_id", "UNKNOWN")

        try:
            save_learning_summary_to_kb(str(top_therapy_id), conclusion)
            total_saved += 1
        except Exception as e:
            print(f"[WARN] Failed to save learning for user {user_id}: {e}")

    print(f" Weekly learning job completed. Stored {total_saved} therapy summaries into KB.")


if __name__ == "__main__":
    run_weekly_learning()
