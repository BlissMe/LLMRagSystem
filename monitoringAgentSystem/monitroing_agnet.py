from langchain_openai import ChatOpenAI
from pymongo import MongoClient
import key_param

llm = ChatOpenAI(model="gpt-4o-mini")

class MonitoringAgent:

    def __init__(self):
        self.client = MongoClient(key_param.MONGO_URI)
        self.db = self.client["blissMe"]
        self.collection = self.db["agent_activity_logs"]

    def fetch_recent_events(self, user_id: int):
        events = list(
            self.collection.find({"user_id": user_id}).sort("timestamp", 1)
        )
        return events

    def analyze_events(self, events):
        """
        Turns raw event logs into intelligent AI-based monitoring insights.
        """
        text_summary = ""
        for ev in events:
            text_summary += (
                f"\nAgent: {ev['agent_name']}\n"
                f"Time: {ev['timestamp']}\n"
                f"Input: {ev.get('input_data')}\n"
                f"Output: {ev.get('output_data')}\n"
            )

        prompt = f"""
        You are a monitoring AI Agent. Analyze the following agent activities:

        {text_summary}

        Provide:
        1. Summary of conversation flow
        2. Errors or anomalies
        3. Emotional trend detection
        4. Whether therapy agent was required
        5. Any warnings or flags
        """

        return llm.invoke(prompt)

    def generate_report(self, user_id: int):
        events = self.fetch_recent_events(user_id)
        if not events:
            return {"error": "No events found"}

        analysis = self.analyze_events(events)

        return {
            "user_id": user_id,
            "events_count": len(events),
            "monitoring_summary": analysis.content
        } 