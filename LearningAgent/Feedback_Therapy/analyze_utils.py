# analyze_utils.py
from collections import defaultdict
import numpy as np

def aggregate_by_therapy(docs):
    """
    Group session docs by therapy_id and compute counts and average durations.
    Output:
      { therapy_id: {
           session_count, total_duration (sum), avg_duration, feedbacks, session_ids, raw_docs
        } }
    """
    out = defaultdict(lambda: {"session_count":0, "total_duration":0.0, "feedbacks":[], "session_ids":[], "raw_docs":[]})
    for d in docs:
        tid = d.get("therapy_id") or d.get("therapyId") or "UNKNOWN"
        rec = out[tid]
        rec["session_count"] += 1
        # duration might be stored as number or string; default 0
        try:
            dur = float(d.get("duration", 0) or 0)
        except Exception:
            dur = 0.0
        rec["total_duration"] += dur
        fb = d.get("feedback")
        if fb:
            rec["feedbacks"].append(str(fb))
        sid = d.get("session_id") or d.get("sessionId")
        if sid:
            rec["session_ids"].append(sid)
        rec["raw_docs"].append(d)

    for tid, r in out.items():
        sc = r["session_count"] or 1
        r["avg_duration"] = (r["total_duration"] / sc) if sc else 0.0

    return dict(out)

def normalize_scores(nums):
    """
    Normalizes numeric array to 0..1 range. If constant array, return 0.5s.
    """
    if not nums:
        return []
    arr = np.array(nums, dtype=float)
    mx = arr.max()
    mn = arr.min()
    if mx == mn:
        return [0.5]*len(arr)
    return ((arr - mn) / (mx - mn)).tolist()
