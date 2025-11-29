from .learningTool import SummarizeTool, DetectTool, TherapyChatTool, GetAggregateTool
from .utils.levels import level_from_detection  

def run_flow(user_id: str, session_id: str, history: str, user_query: str | None):
    summarize = SummarizeTool()
    detect = DetectTool()
    therapy = TherapyChatTool()
    aggregate = GetAggregateTool()

    summ = summarize.run(user_id=user_id, history=history)
    det = detect.run(user_id=user_id, history=history, summaries=[summ.get("summary","")])

    lvl = level_from_detection(
        det["depression_label"],
        det["depression_confidence_detected"]
    )

    therapy_res = None
    if user_query:
        therapy_res = therapy.run(
            user_id=user_id,
            session_id=session_id,
            user_query=user_query,
            depression_level=lvl,
            session_summaries=[summ.get("summary","")]
        )

    agg = aggregate.run(user_id=user_id)

    print({
        "summary": summ,
        "detection": det,
        "therapy": therapy_res,
        "aggregate": agg
    })
