# agent/agent.py
from pydantic import BaseModel
from typing import List, Optional, Tuple
import json
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch

from utils.phq9_questions import PHQ9_QUESTIONS
from utils.tts import generate_tts_audio
import key_param

MODEL_NAME = "gpt-3.5-turbo"  # keep pluggable

class AgentState(BaseModel):
    query: str
    history: str
    summaries: List[str] = []
    asked_phq_ids: List[int] = []
    early_stage: bool = True
    rag_context: List[str] = []

class DepressionAgent:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str, index_name: str):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.index_name = index_name

        self.llm = ChatOpenAI(model=MODEL_NAME, openai_api_key=key_param.openai_api_key, temperature=0.7)
        self.embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)

    def _user_turns_lt3(self, history: str) -> bool:
        lines = [l for l in history.splitlines() if l.lower().startswith(("you:", "user:"))]
        return len(lines) < 3

    def _vector_search(self, query: str, k: int = 3) -> List[str]:
        client = MongoClient(self.mongo_uri)
        try:
            db = client[self.db_name]
            collection = db[self.collection_name]
            vs = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=self.embedding,
                index_name=self.index_name
            )
            docs = vs.similarity_search(query, k=k)
            return [d.page_content[:500] for d in docs]
        finally:
            client.close()

    def _next_phq(self, asked_ids: List[int]) -> Optional[dict]:
        for q in PHQ9_QUESTIONS:
            if q["id"] not in asked_ids:
                return q
        return None

    def _plan(self, state: AgentState) -> dict:
        """Ask the LLM what actions to take; return a small JSON plan."""
        plan_prompt = f"""
You are a planner helping a caring mental-health chatbot decide what to do next.

Return STRICT JSON with keys:
- do_rag: boolean (should we retrieve knowledge for this turn?)
- ask_phq9: boolean (should we ask the next PHQ-9 question now?)

Guidelines:
- If user is opening up but it's the first 2 user turns, keep it light and do NOT start PHQ-9 yet.
- After the early stage, proceed through PHQ-9 sequentially, one question per turn, unless the user clearly shifts topics that need knowledge.
- Use RAG only if the user asks for info or advice that benefits from factual context.

User query: "{state.query}"
Early stage: {state.early_stage}
Asked PHQ IDs: {state.asked_phq_ids}
History (truncated): {state.history[-1500:]}
"""
        resp = self.llm.invoke([{"role": "user", "content": plan_prompt}])
        txt = resp.content.strip()
        # be defensive: try to parse JSON, fallback defaults
        try:
            # try to extract a JSON object even if there's extra text
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1:
                txt = txt[start:end+1]
            plan = json.loads(txt)
        except Exception:
            plan = {"do_rag": False, "ask_phq9": not state.early_stage}
        # Hard rule: never ask PHQ in early stage
        if state.early_stage:
            plan["ask_phq9"] = False
        return {"do_rag": bool(plan.get("do_rag")), "ask_phq9": bool(plan.get("ask_phq9"))}

    def _compose_reply(
        self,
        state: AgentState,
        next_phq: Optional[dict]
    ) -> str:
        summary_text = "\n".join(state.summaries) if state.summaries else "No previous summaries available."
        phq_instruction = ""

        if next_phq:
            if not state.asked_phq_ids:
                phq_instruction += f"""
You may gently say something like:
"To better understand how you're doing, I'd like to ask a few short questions about the past two weeks."

Then ask this question (exactly once):
- "{next_phq['question']}" (meaning: {next_phq['meaning']})

Let user respond with one of:
- not at all
- several days
- more than half the days
- nearly every day
"""
            else:
                phq_instruction += f"""
Ask the next question (exactly once):
- "{next_phq['question']}" (meaning: {next_phq['meaning']})

Let user respond with:
- not at all
- several days
- more than half the days
- nearly every day
"""

        # Final response prompt
        chat_prompt = f"""
You are a warm, caring friend. Keep replies SHORT, non-repetitive, one question per message.
NEVER mention "PHQ-9" by name. Avoid medical/crisis terms unless asked.

Past summaries:
{summary_text}

Relevant context (optional factual snippets):
{state.rag_context}

Conversation history:
{state.history}

User just said: "{state.query}"

{phq_instruction}

Now reply like a kind friend in 1–3 short sentences. If you asked a question above, do not add extra ones.
"""
        resp = self.llm.invoke([{"role": "system", "content": chat_prompt}])
        return resp.content.strip()

    def run(self, query: str, history: str, summaries: List[str], asked_phq_ids: List[int]) -> dict:
        st = AgentState(
            query=query,
            history=history,
            summaries=summaries or [],
            asked_phq_ids=asked_phq_ids or [],
            early_stage=self._user_turns_lt3(history),
            rag_context=[]
        )

        plan = self._plan(st)

        if plan["do_rag"]:
            st.rag_context = self._vector_search(st.query, k=3)

        next_q = self._next_phq(st.asked_phq_ids) if plan["ask_phq9"] else None

        reply_text = self._compose_reply(st, next_q)

        # TTS
        audio_path = generate_tts_audio(reply_text)

        return {
            "response": reply_text,
            "audio_url": f"/voice-audio?path={audio_path}",
            "phq9_questionID": (next_q["id"] if next_q else None),
            "phq9_question": (next_q["question"] if next_q else None),
        }
