from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from .assesmentAgent import DepressionAgent
import key_param

router = APIRouter()

class SummaryRequest(BaseModel):
    history: str

@router.post("/summarize")
async def summarize_chat(data: SummaryRequest):
    paragraph = data.history.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
    return {"summary": paragraph}

class QueryRequest(BaseModel):
    user_query: str
    history: str
    summaries: List[str] = []
    asked_phq_ids: List[int] = []

AGENT = DepressionAgent(
    mongo_uri=key_param.MONGO_URI,
    db_name="Depression_Knowledge_Base",
    collection_name="depression",
    index_name="default1",
)

@router.post("/ask")
async def ask_question(data: QueryRequest):
    return AGENT.run(
        query=data.user_query,
        history=data.history,
        summaries=data.summaries,
        asked_phq_ids=data.asked_phq_ids
    )
