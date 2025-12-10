from datetime import datetime

def save_therapy_history(db, user_id, session_id, therapy_name, therapy_id, duration=None, feedback=None):
    history_collection = db["TherapyHistory"]
    record = {
        "user_id": user_id,
        "session_id": session_id,
        "therapy_id": therapy_id,
        "therapy_name": therapy_name,
        "date": datetime.utcnow(),
        "duration": duration,
        "feedback": feedback
    }
    history_collection.insert_one(record)


def get_user_therapy_history(db, user_id):
    history_collection = db["TherapyHistory"]
    return list(history_collection.find({"user_id": user_id}).sort("date", -1))