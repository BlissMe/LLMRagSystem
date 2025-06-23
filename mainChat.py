from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from fastapi.middleware.cors import CORSMiddleware
import key_param
from utils.phq9_questions import PHQ9_QUESTIONS
from difflib import SequenceMatcher

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
    asked_phq_ids: list[int] = []

@app.get("/")
def read_root():
    return {"message": "API is running"}


# def detect_depression_signals(user_input: str) -> str:
#     system_prompt = """
# You are an assistant that analyzes if a message shows signs of depression.

# Only respond with:
# - high
# - moderate
# - low

# Examples:
# "I feel so empty and tired all the time" → high
# "Sometimes I feel alone, but I'm okay" → moderate
# "I'm good, just bored today" → low
# """

#     detector = ChatOpenAI(
#         model="gpt-3.5-turbo",
#         openai_api_key=key_param.openai_api_key,
#         temperature=0
#     )

#     response = detector.invoke([
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_input}
#     ])

#     return response.content.strip().lower()
@app.post("/ask")
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

    # All unasked PHQ-9 questions
    unasked_questions = [q for q in PHQ9_QUESTIONS if q["id"] not in data.asked_phq_ids]

    # Add all possible questions as soft suggestions to the model
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

    # Chat prompt
    chat_prompt = f"""
You are a friendly chatbot who talks to users like a warm and caring friend.

You are trained to help users with their feelings and thoughts, especially related to depression.
You should always respond in a kind and supportive way, making the user feel heard and understood.
NEVER say "I cannot help you".
Avoid clinical or crisis language unless directly asked.


Your only job is to respond warmly and keep the conversation going in a friendly way.

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

    # Generate response
    bot = ChatOpenAI(
        model="gpt-4",
        openai_api_key=key_param.openai_api_key,
        temperature=0.8
    )

    chat_response = bot.invoke([
        {"role": "system", "content": chat_prompt}
    ])

    final_text = chat_response.content.strip()
    client.close()

    # Soft match PHQ-9 question detection
    matched_q = None
    for q in unasked_questions:
        similarity = SequenceMatcher(None, q["question"].lower(), final_text.lower()).ratio()
        if similarity > 0.6 or q["question"].lower() in final_text.lower():
            matched_q = q
            break

    return {
        "response": final_text,
        "phq9_questionID": matched_q["id"] if matched_q else None,
        "phq9_question": matched_q["question"] if matched_q else None
    }