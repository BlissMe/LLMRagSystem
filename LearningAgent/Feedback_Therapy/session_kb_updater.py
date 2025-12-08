# session_kb_updater.py
import os
import sys
from pymongo import MongoClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import key_param
from decrypt_utils import decrypt_summary
from kb_writer import save_session_summary_to_kb

"""
This script:
1. Reads encrypted summaries from blissMe.sessionsummaries
2. Decrypts `summary`
3. Looks up therapy_id from blissMe.TherapyHistory via sessionID
4. Writes (therapy_id + decrypted summary) into KB WITHOUT user details
"""

# connect to main app DB (blissMe)
client = MongoClient(key_param.MONGO_URI)
app_db = client["blissMe"]
sessions_collection = app_db["sessionsummaries"]
therapy_history_collection = app_db["TherapyHistory"]


def get_therapy_id_for_session(session_id: int):
    doc = therapy_history_collection.find_one({"session_id": session_id})
    if not doc:
        return None
    return doc.get("therapy_id") or doc.get("therapyId")


def process_all_session_summaries():
    cursor = sessions_collection.find({})
    count = 0
    for doc in cursor:
        session_id = doc.get("sessionID") or doc.get("sessionId")
        enc_summary = doc.get("summary")
        if not session_id or not enc_summary:
            continue

        # decrypt
        try:
            plain = decrypt_summary(enc_summary)
        except Exception as e:
            print(f"[WARN] Failed to decrypt session {session_id}: {e}")
            continue

        therapy_id = get_therapy_id_for_session(int(session_id))
        if not therapy_id:
            print(f"[INFO] No therapy_id found for session {session_id}, skipping.")
            continue

        # save into KB (WITHOUT userID)
        try:
            save_session_summary_to_kb(plain)

            count += 1
        except Exception as e:
            print(f"[ERR] Failed to save session {session_id} to KB: {e}")

    print(f" Processed and stored {count} session summaries into KB.")


if __name__ == "__main__":
    process_all_session_summaries()
