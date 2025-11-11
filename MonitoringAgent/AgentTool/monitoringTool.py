# LangAgents/monitoring_tool.py
from langchain.tools import BaseTool
import requests
import aiohttp

class GetUserSummaryTool(BaseTool):
    name = "get_user_summary"
    description = "Fetch the aggregate monitoring summary for a specific user_id from the monitoring FastAPI service."

    def _run(self, user_id: str):
        url = f"http://localhost:8000/monitor/summary/{user_id}/aggregate"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return f"Error {response.status_code}: {response.text}"

    async def _arun(self, user_id: str):
        url = f"http://localhost:8000/monitor/summary/{user_id}/aggregate"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return f"Error {resp.status}: {await resp.text()}"
