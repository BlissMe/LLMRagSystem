from fastapi import APIRouter
from pydantic import BaseModel
from .AssessmentAgentSinhala import DepressionAgent
import key_param

router = APIRouter()

agent_si = DepressionAgent(
    mongo_uri=key_param.MONGO_URI,
    db_name="Depression_Knowledge_Base",
    collection_name="depression",
    index_name="default1"
)

class QueryRequest(BaseModel):
    user_query: str
    history: str
    summaries: list[str] = []
    asked_phq_ids: list[int] = []

@router.post("/ask-si")
async def ask_sinhala(data: QueryRequest):
    return agent_si.run(
        query=data.user_query,
        history=data.history,
        summaries=data.summaries,
        asked_phq_ids=data.asked_phq_ids
    )
