# LangAgents/monitoring_agent.py
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from .monitoringTool import GetUserSummaryTool

def create_monitoring_agent():
    tool = GetUserSummaryTool()
    tools = [tool]

    # Replace this with your OpenAI API key or other model setup
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent
