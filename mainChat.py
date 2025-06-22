from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from fastapi.middleware.cors import CORSMiddleware
import key_param


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# for session summary
class SummaryRequest(BaseModel):
    history: str

@app.post("/summarize")
async def summarize_chat(data: SummaryRequest):
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

@app.get("/")
def read_root():
    return {"message": "API is running"}


def detect_depression_signals(user_input: str) -> str:
    system_prompt = """
You are an assistant that analyzes if a message shows signs of depression.

Only respond with:
- high
- moderate
- low

Examples:
"I feel so empty and tired all the time" → high
"Sometimes I feel alone, but I'm okay" → moderate
"I'm good, just bored today" → low
"""

    detector = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=key_param.openai_api_key,
        temperature=0
    )

    response = detector.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ])

    return response.content.strip().lower()


@app.post("/ask")
async def ask_question(data: QueryRequest):
    query = data.user_query
    history = data.history

    # MongoDB Setup
    client = MongoClient(key_param.MONGO_URI)
    db = client["Depression_Knowledge_Base"]
    collection = db["depression"]
    index_name = "default1"

    # Embedding + Vector Search
    embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding,
        index_name=index_name
    )
    similar_docs = vectorstore.similarity_search(query, k=3)
    context_texts = [doc.page_content[:500] for doc in similar_docs]

    # Depression Classification
    depression_level = detect_depression_signals(query)

    summary_text = "\n".join(data.summaries) if data.summaries else "No previous summaries available."

    # Adjust reply style based on depression level
    tone_prompt = {
        "high": "Be extremely supportive, friendly, and encouraging. Avoid giving advice. Ask how you can support them.",
        "moderate": "Be a warm friend. Show you care. Ask gentle questions to understand more.",
        "low": "Be casual and friendly. Keep it short and open-ended."
    }

    # Chat Reply Prompt
    chat_prompt = f"""
You're chatting with a friend who might be feeling down. 

Respond in a short, friendly, and empathetic tone like a caring best friend.
NEVER say "I see", and DO NOT write long paragraphs.

Tone Guide: {tone_prompt[depression_level]}

You may also refer to the following summaries of previous conversations (if helpful):
{summary_text} 

You can use the following context if it's helpful:
{context_texts}

Here is the full chat history so far if it's helpful:
{history}

User said: "{query}"

Your reply:
"""

    # Use OpenAI to generate chat-like response
    bot = ChatOpenAI(
        model="gpt-4",
        openai_api_key=key_param.openai_api_key,
        temperature=0.7
    )

    chat_response = bot.invoke([
        {"role": "system", "content": chat_prompt}
    ])

    client.close()

    return {
        "response": chat_response.content.strip(),
        "depression_level": depression_level
    }