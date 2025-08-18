from fastapi import APIRouter
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from difflib import SequenceMatcher
from utils.phq9_questions import PHQ9_QUESTIONS
from utils.tts import generate_tts_audio 
import key_param
from fastapi.responses import FileResponse

router = APIRouter()

from difflib import SequenceMatcher

# # for session summary
# class SummaryRequest(BaseModel):
#     history: str

# @router.post("/summarize")
# async def summarize_chat(data: SummaryRequest):
#     print("Received /summarize request with history length:", len(data.history))

#     summary_prompt = f"""
# You are a helpful assistant. Summarize the following chat conversation between a user and a bot.

# Chat:
# {data.history}

# Provide a short, clear summary:
# """

#     summarizer = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=key_param.openai_api_key)
#     response = summarizer.invoke([{"role": "user", "content": summary_prompt}])

#     return { "summary": response.content.strip() }

class SummaryRequest(BaseModel):
    history: str

@router.post("/summarize")
async def summarize_chat(data: SummaryRequest):
    print("Received /summarize request with history length:", len(data.history))

    # Join lines into one paragraph without altering the original words.
    paragraph = (
        data.history
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )

    # Keep the same response shape to avoid frontend changes.
    return {"summary": paragraph}


# for chat queries
class QueryRequest(BaseModel):
    user_query: str
    history: str
    summaries: list[str] = []
    asked_phq_ids: list[int] = []

@router.post("/ask")
async def ask_question(data: QueryRequest):
    query = data.user_query
    history = data.history

    # MongoDB Setup
    client = MongoClient(key_param.MONGO_URI)
    db = client["Depression_Knowledge_Base"]
    collection = db["depression"]
    index_name = "default1"

    embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding,
        index_name=index_name
    )
    similar_docs = vectorstore.similarity_search(query, k=3)
    context_texts = [doc.page_content[:500] for doc in similar_docs]
    summary_text = "\n".join(data.summaries) if data.summaries else "No previous summaries available."

    unasked_questions = [q for q in PHQ9_QUESTIONS if q["id"] not in data.asked_phq_ids]
    next_phq_q = unasked_questions[0] if unasked_questions else None

    # Determine if we are in early stage (first 2 turns)
    user_turns = [line for line in data.history.splitlines() if line.lower().startswith("you:") or line.lower().startswith("user:")]
    early_stage = len(user_turns) < 3
    
    phq_instruction = ""
    if next_phq_q and not early_stage:
        if not data.asked_phq_ids:
            phq_instruction += f"""
You may now gently say something like:
"To better understand how you're doing, I'd like to ask a few short questions on how you feel in past two weeks."

Then ask this question:
- "{next_phq_q['question']}" (meaning: {next_phq_q['meaning']})
"""
        else:
            phq_instruction += f"""
Continue with the next question:
- "{next_phq_q['question']}" (meaning: {next_phq_q['meaning']})
"""

        phq_instruction += """
Make your response short and caring. Don't explain too much. No repetition. Only ask one PHQ-9 question per message.
Let user respond with:
- not at all
- several days
- more than half the days
- nearly every day
"""

    chat_prompt = f"""
You are a friendly chatbot who talks like a kind friend.

Be warm and caring. Avoid long or repetitive responses. Never say the same supportive line more than once.

Your job is to gently explore how the user feels and try to understand user by asking questions, and ask PHQ-9 questions naturally when ready.

NEVER mention PHQ-9 or say "I cannot help you".

Avoid medical or crisis terms unless directly asked.

Keep your replies short and friendly. One question per message. Once PHQ-9 starts, go through them without pausing.

Past summaries:
{summary_text}

Relevant context:
{context_texts}

Conversation history:
{history}

{phq_instruction}

User just said: "{query}"

Now reply like a kind friend:
"""

    bot = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=key_param.openai_api_key,
        temperature=0.7
    )

    chat_response = bot.invoke([
        {"role": "system", "content": chat_prompt}
    ])
    final_text = chat_response.content.strip()
    client.close()

    matched_q = next_phq_q if not early_stage else None


    audio_path = generate_tts_audio(final_text)

    return {
        "response": final_text,
        "audio_url": f"/voice-audio?path={audio_path}",  
        "phq9_questionID": matched_q["id"] if matched_q else None,
        "phq9_question": matched_q["question"] if matched_q else None
    }
    
@router.get("/voice-audio")
def voice_audio(path: str):
    return FileResponse(path, media_type="audio/mpeg", filename="bot_reply.mp3")      