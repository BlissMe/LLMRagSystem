# utils/phq9.py
from typing import List, Dict

CHOICE_TO_SCORE = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

def score_phq9(items: List[Dict[str, str]]):
    """
    items: [{ "id": "phq9_q1", "text": "...", "answer_choice": "Several days" }, ...]
    Returns a dict ready to go into report.data
    """
    scored_items = []
    total = 0
    for it in items:
        s = CHOICE_TO_SCORE.get(it["answer_choice"], 0)
        total += s
        scored_items.append({
            "id": it["id"],
            "text": it["text"],
            "answer_choice": it["answer_choice"],
            "answer_score": s
        })
    severity = (
        "none/minimal" if total <= 4 else
        "mild" if total <= 9 else
        "moderate" if total <= 14 else
        "moderately severe" if total <= 19 else
        "severe"
    )
    suicide_item_positive = next((i for i in scored_items if i["id"] == "phq9_q9"), {"answer_score": 0})["answer_score"] >= 1
    return {
        "type": "assessment.phq9",
        "form_version": "phq9.v1",
        "items": scored_items,
        "total_score": total,
        "severity": severity,
        "flags": {"suicide_item_positive": suicide_item_positive}
    }
