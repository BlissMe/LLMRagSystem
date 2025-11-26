# db_handler.py
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
import os

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client["depression_monitoring"]
reports_collection = db["reports"]

async def save_report(data: dict):
    # normalize timestamp to naive UTC for Mongo or store as aware; here convert to aware
    ts = data.get("timestamp")
    if isinstance(ts, str):
        # support both Z and +00:00
        t = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        data["timestamp"] = datetime.fromisoformat(t)
    await reports_collection.insert_one(data)

async def get_reports_by_user(user_id: str):
    cursor = reports_collection.find({"user_id": user_id}).sort("timestamp", -1)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        # make timestamp ISO again for API response
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        results.append(doc)
    return results

# Aggregate over normalized fields
async def get_user_summary(user_id: str):
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"timestamp": -1}},
        {
            "$group": {
                "_id": "$user_id",
                "avg_confidence": {
                    "$avg": {
                        "$cond": [
                            {"$and": [
                                {"$eq": ["$agent_name", "classifier"]},
                                {"$ifNull": ["$data.depression_confidence_detected", False]}
                            ]},
                            "$data.depression_confidence_detected",
                            None
                        ]
                    }
                },
                "last_therapy": {
                    "$first": {
                        "$cond": [
                            {"$eq": ["$agent_name", "therapy"]},
                            "$data.response.therapy_name",
                            None
                        ]
                    }
                },
                "last_assessment": {
                    "$first": {
                        "$cond": [
                            {"$eq": ["$agent_name", "assessment"]},
                            "$timestamp",
                            None
                        ]
                    }
                },
                "total_reports": {"$sum": 1}
            }
        }
    ]
    result = await reports_collection.aggregate(pipeline).to_list(length=1)
    return result[0] if result else None
