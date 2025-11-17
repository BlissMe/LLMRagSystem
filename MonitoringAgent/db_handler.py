from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client["depression_monitoring"]
reports_collection = db["reports"]

async def save_report(data: dict):
    if isinstance(data.get("timestamp"), str):
        data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", ""))
    await reports_collection.insert_one(data)

async def get_reports_by_user(user_id: str):
    cursor = reports_collection.find({"user_id": user_id}).sort("timestamp", -1)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results

async def get_user_summary(user_id: str):
    """Compute average depression confidence, last therapy, last assessment date"""
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"timestamp": -1}},
        {
            "$group": {
                "_id": "$user_id",
                "avg_confidence": {
                    "$avg": {
                        "$cond": [
                            {"$ifNull": ["$data.depression_confidence", False]},
                            "$data.depression_confidence",
                            None
                        ]
                    }
                },
                "last_therapy": {
                    "$first": {
                        "$cond": [
                            {"$eq": ["$agent_name", "therapy"]},
                            "$data.therapy_type",
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
