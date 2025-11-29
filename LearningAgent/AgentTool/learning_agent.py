from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from .learningTool import SummarizeTool, DetectTool, TherapyChatTool, GetAggregateTool

SYSTEM_PROMPT = """You are the Monitoring Orchestrator.
Given user_id, session_id, and chat history (and optionally user_query),
1) Call summarize_history to produce a short summary string.
2) Call detect_depression_emotion with the same history and the summary.
3) If user_query is provided, call therapy_chat with a depression_level derived from detection:
   - If depression_label == "Depression Signs Detected":
       - if confidence >= 70 => "moderate"
       - else => "minimal"
   - If "No Depression Signs Detected": "minimal"
4) Finally, call get_user_summary to fetch the aggregate.
Return a compact JSON with {summary, detection, therapy, aggregate}.
Do not invent fields; only pass tool outputs through.
"""

def create_monitoring_agent():
    tools = [
        SummarizeTool(),
        DetectTool(),
        TherapyChatTool(),
        GetAggregateTool()
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        system_message=SYSTEM_PROMPT
    )
