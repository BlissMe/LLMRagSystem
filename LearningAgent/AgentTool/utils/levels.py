def level_from_detection(label: str, confidence: int) -> str:
    if label == "Depression Signs Detected":
        return "moderate" if confidence >= 70 else "minimal"
    return "minimal"
