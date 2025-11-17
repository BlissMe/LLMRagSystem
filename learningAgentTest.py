from LearningAgent.LearningAgent import LearningAgent
from bson import ObjectId
import key_param

# Initialize the agent
agent = LearningAgent(mongo_uri=key_param.MONGO_URI, db_name="learning_test_db")

# ---------- STEP 1: Test ingestion ----------
text = """
Hi, I've been feeling very down lately and have trouble sleeping.
Sometimes I don't want to talk to anyone. My email is user123@gmail.com
"""

result = agent.ingest_interaction(
    session_id="test_session_1",
    conversation_text=text,
    metadata={
        "source": "classifier",
        "labels": {"classifier": "Depression Signs Detected", "confidence": 85}
    }
)
print("✅ Ingestion Result:", result.dict())

# ---------- STEP 2: Verify anonymization ----------
from pymongo import MongoClient

with MongoClient(key_param.MONGO_URI) as client:
    doc = client["learning_test_db"]["knowledge_base"].find_one({"metadata.session_id": "test_session_1"})
    print("\n📄 Stored Document Content:\n", doc["content"])
    assert "[email_removed]" in doc["content"], "❌ Email was not anonymized!"


# ---------- STEP 3: Apply feedback ----------
feedback = agent.apply_feedback(
    doc_id=doc["_id"],
    corrected_label="No Depression Signs Detected",
    note="Classifier overestimated sadness"
)
print("\n✅ Feedback Result:", feedback)

# ---------- STEP 4: Export fine-tune JSONL ----------
path = agent.export_finetune_jsonl("tmp/ft_export.jsonl", limit=5)
print("\n✅ Fine-tune data exported to:", path)

# ---------- STEP 5: Distribution & summary ----------
dist = agent.get_label_distribution()
print("\n📊 Label Distribution:", dist)

plan = agent.propose_updates(sample_size=5)
print("\n🧾 Proposed Update Plan:", plan)
