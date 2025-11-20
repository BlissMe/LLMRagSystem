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

    user_turns = [l for l in data.history.splitlines() if l.lower().startswith(("you:", "user:"))]
    early_stage = len(user_turns) < 3  # start PHQ only after 3 chats

    unasked = [q for q in PHQ9_QUESTIONS if q["id"] not in data.asked_phq_ids]
    next_q = unasked[0] if unasked else None

# Only enable PHQ mode if user has chatted at least 3 times
    phq_mode = False
    if not early_stage and next_q:
      phq_mode = True

    # --- Build PHQ instruction if needed ---
    phq_instruction = ""
    if phq_mode:
        if not data.asked_phq_ids:
            # Before first PHQ question
            phq_instruction = (
                "You MUST now gently say something like:\n"
                '"To better understand how you’re doing, I’d like to ask a few short questions about how you’ve felt in the past two weeks."\n'
                "Then ask this first question EXACTLY as shown (do NOT paraphrase):\n"
                f'- "{next_q["meaning"]}"\n\n'
                "After the user replies, respond with ONE SHORT caring line (eg. “Thank you for sharing.” / “I understand, that sounds tough.” / “I understand.”/ “I’m here for you.”) and move to the next PHQ-9 question in order EXACTLY as shown .\n"
                "Ask only one PHQ question per message.\n"
                "User can reply with: not at all, several days, more than half the days, nearly every day."
            )
        else:
            # For later PHQ questions
            phq_instruction = (
                "Respond with one short caring line (eg. “Thank you for sharing.” / “I understand, that sounds tough.” / “I understand.”/“I’m here for you.”) acknowledging the user’s last answer, then immediately ask the next PHQ-9 question in order and exactly as shown below (do NOT paraphrase):\n"
                f'- "{next_q["meaning"]}"\n\n'
                "Do not add unrelated or extra questions.\n"
                "Ask only one PHQ question per message.\n"
                "User can reply with: not at all, several days, more than half the days, nearly every day."
            )

    chat_prompt = f"""
You are a friendly chatbot who talks like a kind friend.

- Be warm and caring. Avoid long or repetitive responses. Never say the same supportive line more than once.

- Your job is to gently explore how the user feels and try to understand user by asking questions.

- NEVER mention PHQ-9 or say "I cannot help you".

- Avoid medical or crisis terms unless directly asked.
- Keep your replies short and friendly. One question per message. Once PHQ-9 starts, go through them without pausing.

-before starting phq-9, need to ask at least 3 normal chat turns.
-then when starting phq-9, before first question, you MUST gently inform user about the questions like "To better understand how you’re doing, I’d like to ask a few short questions about how you’ve felt in the past two weeks."
 - Ask all 9 PHQ-9 questions in order, one at a time, exactly given as in the phq_instruction.

- After finishing all 9, continue chatting normally with care and empathy.

Past summaries:
{summary_text}

Relevant context:
{context_texts}

Conversation history:
{history}

{phq_mode}
{phq_instruction}

User just said: "{query}"

Now reply like a kind friend:
"""

    bot = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=key_param.openai_api_key,f
        temperature=0.7
    )

    chat_response = bot.invoke([
        {"role": "system", "content": chat_prompt }
    ])
    final_text = chat_response.content.strip()
    client.close()

    matched_q = next_q if phq_mode else None
    if not unasked:  # all 9 done
        matched_q = None
        phq_mode = False


    audio_path = generate_tts_audio(final_text)

    return {
        "response": final_text,
        "audio_url": f"/voice-audio?path={audio_path}",  
        "phq9_questionID": matched_q["id"] if matched_q else None,
        "phq9_question": matched_q["question"] if matched_q else None,
        "lanuage": "English"
    }
    
@router.get("/voice-audio")
def voice_audio(path: str):
    return FileResponse(path, media_type="audio/mpeg", filename="bot_reply.mp3")      