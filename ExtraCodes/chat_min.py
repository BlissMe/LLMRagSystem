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



# for session summary
class SummaryRequest(BaseModel):
    history: str

@router.post("/summarize")
async def summarize_chat(data: SummaryRequest):
    print("Received /summarize request with history length:", len(data.history))

    summary_prompt = f"""
You are a helpful assistant. Summarize the following chat conversation between a user and a bot.

Chat:
{data.history}

Provide a short, clear summary:
"""

    summarizer = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=key_param.openai_api_key)
    response = summarizer.invoke([{"role": "user", "content": summary_prompt}])

    return { "summary": response.content.strip() }

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

    phq_instruction = ""
    if unasked_questions:
        phq_instruction += """
If any of the following questions feel appropriate based on the user's mood, gently ask one of them.
Only include one question if it fits the situation. Rephrase it naturally. If none fit, skip them completely.

Here are the available questions:
"""
        for q in unasked_questions:
            phq_instruction += f'- "{q["question"]}"\n'

        phq_instruction += """
Make sure the user can answer with something like:
- not at all
- several days
- more than half the days
- nearly every day        
Do not say it's from PHQ-9.
Don't use parentheses when asking question.
Don't list choices.
Only ask if it makes sense in context.
"""

    chat_prompt = f"""
You are a friendly chatbot who talks to users like a warm and caring friend.

You are trained to help users with their feelings and thoughts, especially related to depression.
You should always respond in a kind and supportive way, making the user feel heard and understood.
NEVER say "I cannot help you".
Avoid clinical or crisis language unless directly asked.

Your only job is to respond warmly and keep the conversation going in a friendly way.

Do not ask same question again and again.
Respond in a short, kind, and caring tone.
Don't ask more than one question in a message.
If you include a PHQ-9 question, don't mix it with anything else.
Try to vary your tone and phrasing from previous messages.
Avoid sounding like a script or repeating past responses.

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
        temperature=0.8
    )

    chat_response = bot.invoke([{"role": "system", "content": chat_prompt}])
    final_text = chat_response.content.strip()
    client.close()

    matched_q = None
    for q in unasked_questions:
        similarity = SequenceMatcher(None, q["question"].lower(), final_text.lower()).ratio()
        if similarity > 0.6 or q["question"].lower() in final_text.lower():
            matched_q = q
            break

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