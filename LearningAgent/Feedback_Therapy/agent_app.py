# agent_app.py
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from pipeline import compute_therapy_report_for_user
from conclusion_utils import generate_conclusion
from kb_writer import save_summary_to_kb
import json

# LLM used to wrap agent (for natural language reasoning)
llm = OpenAI(temperature=0)

def tool_get_report(user_input: str):
    """
    Agent calls this tool with a string containing a user id (or text with a number).
    Returns a JSON string of report + conclusions.
    """
    import re
    m = re.search(r"\d+", user_input)
    if not m:
        return "Error: please include a numeric user id."
    user_id = int(m.group())
    report = compute_therapy_report_for_user(user_id)
    conclusion = generate_conclusion(report)
    
    try:
        if report.get("therapies"):
            top_therapy = report["therapies"][0]["therapy_id"]
            save_summary_to_kb(top_therapy, conclusion)
    except Exception as e:
        print(f"[KB WARNING] Could not save summary to KB: {e}")
        
    result = {"report": report, "conclusion": conclusion}
    return json.dumps(result, indent=2)

report_tool = Tool(
    name="therapy_report_tool",
    func=tool_get_report,
    description="Get an analyzed therapy report and human conclusion for a user. Input: numeric user id or text containing id."
)

tools = [report_tool]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False  # set True for debug
)

def run_agent_query(query: str):
    return agent.run(query)



# Example usage when running as script:
if __name__ == "__main__":
    out = run_agent_query("Create a therapy suitability report for user 31")
    print(out)
