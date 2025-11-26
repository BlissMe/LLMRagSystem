# db_utils.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Please set MONGO_URI in your .env")

client = MongoClient(MONGO_URI)
db = client.get_default_database()  # will use DB from URI (blissMe)
therapy_collection = db.get_collection("TherapyHistory")  # adjust if collection name differs

def get_therapy_history_by_user(user_id: int):
    """
    Returns list of therapy history documents for the provided user_id.
    """
    if user_id is None:
        return []
    # if user_id stored as int in DB, query as int; adjust if string
    try:
        cursor = therapy_collection.find({"user_id": int(user_id)})
    except Exception:
        cursor = therapy_collection.find({"user_id": user_id})
    return list(cursor)
