# LangAgents/orchestrator.py
from .monitoring_agent import create_monitoring_agent

def test_monitoring():
    agent = create_monitoring_agent()
    query = "Get the monitoring summary for user123"
    response = agent.run(query)
    print(response)

if __name__ == "__main__":
    test_monitoring()
