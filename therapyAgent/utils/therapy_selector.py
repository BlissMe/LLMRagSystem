import random

def get_therapy_recommendation(db, depression_level, history_records):
    """
    Select therapy based on depression level and past usage.
    """
    therapy_collection = db["TherapyList"]

    # filter by level
    query = {"level": {"$in": [depression_level.lower(), "general"]}}
    all_therapies = list(therapy_collection.find(query))

    if not all_therapies:
        return {"id": None, "name": "general mindfulness", "description": "simple mindfulness practice"}

    # avoid recently used therapies
    used_ids = [h["therapy_id"] for h in history_records]
    available = [t for t in all_therapies if str(t["_id"]) not in used_ids]

    selected = random.choice(available or all_therapies)
    return {"id": str(selected["_id"]), "name": selected["name"], "description": selected["description"]}
