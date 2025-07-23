from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from bson import ObjectId
import key_param

# Create a standalone FastAPI app
app = FastAPI()

# 🔶 Pydantic model for request
class DepressionCheckRequest(BaseModel):
    user_id: int  # You will send userID from frontend

@app.post("/detect_depression_by_userid")
async def detect_depression(request: DepressionCheckRequest):
    try:
        # ✅ Connect to MongoDB
        client = MongoClient(key_param.MONGO_URI)
        db = client["test"]  
        summaries = db["sessionsummaries"]  

        # 🔍 Get the latest session summary for this user
        summary_doc = summaries.find_one(
            {"userID": request.user_id},
            sort=[("createdAt", -1)]  # get the most recent summary
        )

        if not summary_doc:
            return {"error": f"No summary found for user ID {request.user_id}"}

        summary_text = summary_doc["summary"]

        # 🧠 GPT Prompt
        prompt = f"""
You are a mental health assistant.

Here is a summary of a chat with a user:
\"\"\"{summary_text}\"\"\"

Does this summary show signs of depression?

Only respond with one of the following:
- Signs of depression
- No signs of depression
"""

        # 🔗 Connect to GPT (OpenAI)
        gpt = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=key_param.openai_api_key,
            temperature=0
        )

        gpt_response = gpt.invoke([
            {"role": "user", "content": prompt}
        ])
        result = gpt_response.content.strip()

        return {
            "user_id": request.user_id,
            "summary_id": str(summary_doc["_id"]),
            "depression_detection": result
        }

    except Exception as e:
        return {"error": str(e)}
