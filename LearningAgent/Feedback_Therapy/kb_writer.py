# kb_writer.py
import datetime
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.documents import Document  # if this import fails, use from langchain.schema import Document
import key_param

# connect to KB
client = MongoClient(key_param.MONGO_URI_KB)
db = client["Depression_Knowledge_Base"]
kb_collection = db["all_knowledge"]

embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)

# Reusable vectorstore handle for incremental writes
vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    key_param.MONGO_URI_KB,
    "Depression_Knowledge_Base.all_knowledge",
    embedding
)


def save_learning_summary_to_kb(therapy_id: str, summary_text: str):
    """Existing function for feedback-based conclusions."""
    doc = {
        "text": summary_text,
        "therapy_id": therapy_id,
        "source": "learning_agent_conclusion",
        "timestamp": datetime.datetime.utcnow()
    }
    inserted = kb_collection.insert_one(doc)

    doc_obj = Document(
        page_content=summary_text,
        metadata={
            "therapy_id": therapy_id,
            "source": "learning_agent_conclusion",
            "mongo_id": str(inserted.inserted_id),
        }
    )
    vectorstore.add_documents([doc_obj])


def save_session_summary_to_kb(session_summary: str):
    if not session_summary:
        return

    doc = {
        "text": session_summary,
        "source": "session_summary",
        "timestamp": datetime.datetime.utcnow()
    }
    inserted = kb_collection.insert_one(doc)

    doc_obj = Document(
        page_content=session_summary,
        metadata={
            "source": "session_summary",
            "mongo_id": str(inserted.inserted_id),
        }
    )
    vectorstore.add_documents([doc_obj])

