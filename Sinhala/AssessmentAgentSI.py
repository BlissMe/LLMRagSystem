from pydantic import BaseModel
from typing import List, Optional
from pymongo import MongoClient
import json

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch

# 🔑 Import translations
from utils.translate import si_to_en, en_to_si
from utils.phq9_questions_si import PHQ9_QUESTIONS_SI
import key_param

MODEL_NAME = "gpt-4-turbo"  # or "gpt-3.5-turbo"


class AgentState(BaseModel):
    query: str
    history: str
    summaries: List[str] = []
    asked_phq_ids: List[int] = []
    early_stage: bool = True
    rag_context: List[str] = []


class DepressionAgentSI:
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
            vs = MongoDBAtlasVectorSearch(
                collection=client[self.db_name][self.collection_name],
                embedding=self.embedding,
                index_name=self.index_name
            )
            docs = vs.similarity_search(query, k=k)
            return [d.page_content[:500] for d in docs]
        finally:
            client.close()

    def _next_phq(self, asked_ids: List[int]) -> Optional[dict]:
        for q in PHQ9_QUESTIONS_SI:
            if q["id"] not in asked_ids:
                return q
        return None

    def _plan(self, state: AgentState) -> dict:
        # Ask GPT to decide plan (in English, since GPT is stronger in EN)
        plan_prompt = f"""
You are a planner helping a Sinhala-language chatbot.
Return STRICT JSON:
- do_rag: boolean
- ask_phq9: boolean

Guidelines:
- First 2 turns: do not ask PHQ-9
- Later: ask PHQ-9 sequentially unless knowledge is needed.
User query: "{state.query}"
Early stage: {state.early_stage}
Asked PHQ IDs: {state.asked_phq_ids}
"""
        resp = self.llm.invoke([{"role": "user", "content": plan_prompt}])
        txt = resp.content.strip()
        try:
            start, end = txt.find("{"), txt.rfind("}")
            if start != -1 and end != -1:
                txt = txt[start:end+1]
            plan = json.loads(txt)
        except Exception:
            plan = {"do_rag": False, "ask_phq9": not state.early_stage}
        if state.early_stage:
            plan["ask_phq9"] = False
        return {"do_rag": bool(plan.get("do_rag")), "ask_phq9": bool(plan.get("ask_phq9"))}

    def _compose_reply(self, state: AgentState, next_phq: Optional[dict]) -> str:
        summary_text = "\n".join(state.summaries) if state.summaries else "කෙටි සාරාංශයක් නොමැත."
        phq_instruction = ""

        if next_phq:
            if not state.asked_phq_ids:
                phq_instruction += f"""
කරුණාකර මෙසේ පටන් ගන්න:
"ඔබේ තත්වය තේරුම් ගැනීමට, මම මෙම සති දෙකේ දී ඔබේ හැඟීම් පිළිබඳ කෙටි ප්‍රශ්න කිහිපයක් අසන්නම්."

ඉන්පසු අසන්න:
- "{next_phq['question']}"
"""
            else:
                phq_instruction += f"""
ඊළඟ ප්‍රශ්නය අසන්න:
- "{next_phq['question']}"
"""

        chat_prompt = f"""
Answer in **Sinhala** clearly, kindly, and concisely.
Do not mention PHQ-9 name.
Avoid repetition.
End with a soft, caring question if possible.

Summary:
{summary_text}

Knowledge (if any):
{state.rag_context}

Past conversation:
{state.history}

User just said: "{state.query}"

{phq_instruction}
"""
        resp = self.llm.invoke([{"role": "system", "content": chat_prompt}])
        return resp.content.strip()

    def run(self, query: str, history: str, summaries: List[str], asked_phq_ids: List[int]) -> dict:
        # 🔄 Translate user input to English for planning + RAG
        query_en = si_to_en(query)
        history_en = si_to_en(history)

        st = AgentState(
            query=query_en,
            history=history_en,
            summaries=summaries or [],
            asked_phq_ids=asked_phq_ids or [],
            early_stage=self._user_turns_lt3(history_en),
            rag_context=[]
        )

        plan = self._plan(st)

        if plan["do_rag"]:
            st.rag_context = self._vector_search(st.query, k=3)

        next_q = self._next_phq(st.asked_phq_ids) if plan["ask_phq9"] else None

        # GPT generates reply (may be in Sinhala or English depending on strength)
        reply_text_en = self._compose_reply(st, next_q)

        # 🔑 Guarantee Sinhala output
        reply_text_si = en_to_si(reply_text_en)

        return {
            "response": reply_text_si,
            "audio_url": None,  # Sinhala TTS can be plugged in later
            "phq9_questionID": (next_q["id"] if next_q else None),
            "phq9_question": (next_q["question"] if next_q else None),
            "lanuage": "Sinhala"

        }
