from typing import Dict, Any, Optional
from .agent import analyze_text_for_depression_and_emotion

def detect_from_history_and_summary(history: str, summary: Optional[str]) -> Dict[str, Any]:
    combined = f"Conversation History:\n{history}\n\nSummary:\n{summary or 'N/A'}"
    return analyze_text_for_depression_and_emotion(combined)
