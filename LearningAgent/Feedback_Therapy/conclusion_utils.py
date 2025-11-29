# feedback_utils.py
from typing import Dict, Any, List

def generate_conclusion(report: Dict[str, Any], top_n: int = 5) -> str:
    therapies = report.get("therapies", [])
    user_id = report.get("user_id")
    if not therapies:
        return f"User {user_id} has no therapy history."

    # compute total_duration if missing (we already computed in pipeline, but be safe)
    for t in therapies:
        if "total_duration" not in t:
            t["total_duration"] = (t.get("avg_duration", 0.0) or 0.0) * (t.get("session_count", 0) or 0)

    total_time = sum(float(t.get("total_duration", 0.0) or 0.0) for t in therapies)

    top = therapies[0]
    suitability_pct = round(float(top.get("composite_score", 0.0)) * 100)

    if total_time > 0:
        top_usage_pct = round((float(top.get("total_duration", 0.0)) / total_time) * 100)
        usage_basis = "duration"
    else:
        total_sessions = sum(int(t.get("session_count", 0) or 0) for t in therapies) or 1
        top_usage_pct = round((int(top.get("session_count", 0) or 0) / total_sessions) * 100)
        usage_basis = "sessions"

    lines: List[str] = []
    lines.append(f"Conclusion for user {user_id}:")
    lines.append("")
    lines.append(f"- The most suitable therapy is {top.get('therapy_id')}.")
    lines.append(f"  - Suitability: {suitability_pct}% (based on combined feedback & duration).")
    lines.append(f"  - This therapy makes up ~{top_usage_pct}% of user's total therapy {usage_basis}.")
    lines.append(f"  - Details: sessions={top.get('session_count')}, avg_duration={top.get('avg_duration')}, composite_score={top.get('composite_score')}")
    lines.append("")

    if len(therapies) > 1:
        lines.append("- Other therapies and usage:")
        # list others up to top_n - 1
        for t in therapies[1:top_n]:
            if total_time > 0:
                pct = round((float(t.get("total_duration", 0.0)) / total_time) * 100)
                usage_text = f"{pct}% of total duration"
            else:
                total_sessions = sum(int(x.get("session_count", 0) or 0) for x in therapies) or 1
                pct = round((int(t.get("session_count", 0) or 0) / total_sessions) * 100)
                usage_text = f"{pct}% of sessions"
            lines.append(f"  - {t.get('therapy_id')}: sessions={t.get('session_count')}, avg_duration={t.get('avg_duration')}, total_duration={t.get('total_duration')}, {usage_text}, composite_score={t.get('composite_score')}")

    lines.append("")
    # pick next top two as secondary options if present
    secondaries = [t["therapy_id"] for t in therapies[1:3]] if len(therapies) > 1 else []
    sec_text = ", ".join(secondaries) if secondaries else "none"
    lines.append(f"Recommendation: prioritize therapy {top.get('therapy_id')} (suitability {suitability_pct}%). Consider {sec_text} as secondary options.")

    return "\n".join(lines)
