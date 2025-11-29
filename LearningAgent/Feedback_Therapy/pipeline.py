# pipeline.py
from .db_utils import get_therapy_history_by_user
from .analyze_utils import aggregate_by_therapy, normalize_scores
from .llm_utils import score_feedback_with_llm

def compute_therapy_report_for_user(user_id: int):
    docs = get_therapy_history_by_user(user_id)
    if not docs:
        return {"user_id": user_id, "therapies": []}

    ag = aggregate_by_therapy(docs)
    therapy_ids = list(ag.keys())
    avg_durations = [ag[tid]["avg_duration"] for tid in therapy_ids]
    durations_norm = normalize_scores(avg_durations)

    therapy_rows = []
    for idx, tid in enumerate(therapy_ids):
        rec = ag[tid]
        feedbacks = rec.get("feedbacks", [])
        if feedbacks:
            fb_scores = score_feedback_with_llm(feedbacks)
            feedback_mean = sum(fb_scores) / len(fb_scores) if fb_scores else 0.5
        else:
            feedback_mean = 0.5

        norm_duration = durations_norm[idx] if idx < len(durations_norm) else 0.5

        # Composite score: weight feedback more (70%) and duration 30% by default
        composite = 0.7 * float(feedback_mean) + 0.3 * float(norm_duration)

        therapy_rows.append({
            "therapy_id": tid,
            "session_count": rec.get("session_count", 0),
            "avg_duration": round(rec.get("avg_duration", 0.0), 3),
            "total_duration": round(rec.get("total_duration", 0.0), 3),
            "feedback_mean": round(float(feedback_mean), 3),
            "norm_duration": round(float(norm_duration), 3),
            "composite_score": round(float(composite), 3),
            "sample_feedbacks": rec.get("feedbacks", [])[:5],
            "session_ids": rec.get("session_ids", [])[:10]
        })

    # sort by composite_score descending
    therapy_rows.sort(key=lambda x: x["composite_score"], reverse=True)

    return {"user_id": user_id, "therapies": therapy_rows}
