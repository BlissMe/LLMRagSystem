"""
Continuous Learning Agent implementation for the Multi-Agent Depression System.

Responsibilities implemented here (synchronous, on-call):
 - ingest anonymized interaction history into the vector-backed knowledge base
 - create / export fine-tune-ready JSONL examples (instruction -> response)
 - produce a short "update plan" (human-reviewable) describing common failures
 - apply explicit feedback labels (corrections) to stored examples
 - lightweight drift / distribution checks to highlight when classifier outputs shift

This file is intended to be imported and called from your app (e.g. when a session ends,
or when the monitoring/classifier agent returns feedback). It does NOT perform any
asynchronous scheduling or background training — it prepares artifacts and updates the DB
so a human or a CI job can trigger heavy ops (fine-tune, full reindex, etc).
"""

import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from pydantic import BaseModel
from pymongo import MongoClient, ASCENDING
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import key_param

# Configurable defaults
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 800  # characters per chunk when storing KB
FT_BUFFER_COLLECTION = "fine_tune_buffer"  # collection to hold labeled examples for later export

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{6,}\d)")

# -------------------------
# Data models
# -------------------------
class IngestResult(BaseModel):
    inserted_count: int
    last_ids: List[Any]

class FeedbackRecord(BaseModel):
    interaction_id: Any
    corrected_label: str
    note: Optional[str] = None
    timestamp: float = time.time()

# -------------------------
# Utilities
# -------------------------
def _anonymize_text(s: str) -> str:
    """Remove emails and phone numbers, minimal PII masking.
    This is intentionally conservative; you may add stronger PII removal
    (names, addresses) if you have heuristics.
    """
    if not s:
        return s
    s = _EMAIL_RE.sub("[email_removed]", s)
    s = _PHONE_RE.sub("[phone_removed]", s)
    # collapse sequences of whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _chunk_text(s: str, max_len: int = CHUNK_SIZE) -> List[str]:
    """Simple character-based chunking that tries to break on sentence boundaries."""
    if not s:
        return []
    s = s.strip()
    if len(s) <= max_len:
        return [s]
    sentences = re.split(r'(?<=[\.\?\!])\s+', s)
    chunks = []
    cur = ""
    for sent in sentences:
        if len(cur) + len(sent) + 1 <= max_len:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(sent) <= max_len:
                cur = sent
            else:
                # hard split long sentence
                for i in range(0, len(sent), max_len):
                    chunks.append(sent[i:i+max_len])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks

# -------------------------
# LearningAgent
# -------------------------
class LearningAgent:
    """
    Continuous Learning Agent.

    Primary public methods:
      - ingest_interaction(session_id, conversation_text, meta) -> IngestResult
      - apply_feedback(interaction_doc_id, corrected_label, note=None)
      - export_finetune_jsonl(path, limit=None) -> path
      - propose_updates(sample_size=200) -> dict (human-readable plan)
      - get_label_distribution(window=1000) -> dict
    """

    def __init__(
        self,
        mongo_uri: str,
        db_name: str = "mas_db",
        kb_collection: str = "knowledge_base",
        index_name: str = "kb_index",
        embedding_model: str = DEFAULT_EMBED_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.kb_collection = kb_collection
        self.index_name = index_name
        self.embedding = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
        self.llm = ChatOpenAI(model=llm_model, openai_api_key=key_param.openai_api_key, temperature=0.0)
        self.embed_model_name = embedding_model

        # Ensure indexes on Mongo side for efficient queries (idempotent)
        with MongoClient(self.mongo_uri) as c:
            db = c[self.db_name]
            coll = db[self.kb_collection]
            coll.create_index([("created_at", ASCENDING)])
            coll.create_index([("metadata.session_id", ASCENDING)])

    # -------------------------
    # Core ingestion
    # -------------------------
    def ingest_interaction(
        self,
        session_id: str,
        conversation_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        keep_raw: bool = False,
    ) -> IngestResult:
        """
        Ingest a single session's conversation into the vector KB.

        - anonymizes the text
        - chunks it
        - computes embeddings (batch)
        - upserts documents into self.kb_collection with fields:
            { content, embedding, metadata, created_at }
        Returns IngestResult with inserted ids count.
        """
        metadata = metadata or {}
        anonymized = _anonymize_text(conversation_text)
        chunks = _chunk_text(anonymized, CHUNK_SIZE)
        if not chunks:
            return IngestResult(inserted_count=0, last_ids=[])

        # compute embeddings
        embeddings = self.embedding.embed_documents(chunks)

        docs = []
        ts = datetime.utcnow()
        for i, chunk in enumerate(chunks):
            doc = {
                "content": chunk,
                "embedding": embeddings[i],
                "metadata": {
                    "session_id": session_id,
                    "source": metadata.get("source", "conversation"),
                    "labels": metadata.get("labels", {}),  # classifier / monitoring metadata if any
                    "keep_raw": bool(keep_raw),
                },
                "created_at": ts,
            }
            docs.append(doc)

        client = MongoClient(self.mongo_uri)
        try:
            coll = client[self.db_name][self.kb_collection]
            res = coll.insert_many(docs)
            return IngestResult(inserted_count=len(res.inserted_ids), last_ids=res.inserted_ids)
        finally:
            client.close()

    # -------------------------
    # Feedback / labeling
    # -------------------------
    def apply_feedback(
        self,
        doc_id,
        corrected_label: str,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply explicit feedback to a KB document (e.g. classifier was wrong).
        Stores feedback in a dedicated collection for traceability and also updates
        the document's metadata 'labels.corrected'.
        """
        client = MongoClient(self.mongo_uri)
        try:
            db = client[self.db_name]
            coll = db[self.kb_collection]
            fb_coll = db[FT_BUFFER_COLLECTION]

            # update doc metadata
            upd = coll.find_one_and_update(
                {"_id": doc_id},
                {
                    "$set": {"metadata.labels.corrected": corrected_label, "metadata.labels.corrected_at": datetime.utcnow()}
                },
                return_document=True,
            )

            fb = {
                "interaction_doc_id": doc_id,
                "corrected_label": corrected_label,
                "note": note or "",
                "timestamp": datetime.utcnow(),
            }
            fb_id = fb_coll.insert_one(fb).inserted_id

            return {"updated_doc": bool(upd), "feedback_id": fb_id}
        finally:
            client.close()

    # -------------------------
    # Fine-tune exports
    # -------------------------
    def export_finetune_jsonl(self, out_path: str, limit: Optional[int] = 1000) -> str:
        """
        Export labeled examples suitable for human review / fine-tuning.

        Schema (OpenAI-style JSONL): each line is {"prompt": "...", "completion": " ..."}
        We build instruction-response pairs from KB items that have 'labels' in metadata.

        Returns the path to the written file.
        """
        client = MongoClient(self.mongo_uri)
        try:
            coll = client[self.db_name][self.kb_collection]
            # Prefer items that have corrected labels first, then items with original labels
            cursor = coll.find(
                {"$or": [{"metadata.labels.corrected": {"$exists": True}}, {"metadata.labels": {"$exists": True}}]}
            ).sort("created_at", ASCENDING).limit(limit)

            lines = []
            for doc in cursor:
                content = doc.get("content", "")
                labels = doc.get("metadata", {}).get("labels", {})
                corrected = labels.get("corrected")
                original = labels.get("classifier") or labels.get("detected_label") or labels.get("depression_label")
                label = corrected or original
                if not label:
                    continue
                prompt = (
                    "You are a supportive mental-health assistant. Given the anonymized user text below, "
                    "produce a concise empathetic reply that reflects appropriate tone (no diagnosis). "
                    "Also include a suggested 'next action' token in square brackets, one of: [ask_phq], [encourage], [provide_self_help], [escalate].\n\n"
                    "User text:\n"
                    f"{content}\n\n"
                    "Reply:"
                )
                completion = f"{label} ||| SuggestedAction: {labels.get('suggested_action','[encourage]')}"
                lines.append(json.dumps({"prompt": prompt, "completion": completion}))

            # write file
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for l in lines:
                    f.write(l + "\n")
            return out_path
        finally:
            client.close()

    # -------------------------
    # Simple analysis / proposals
    # -------------------------
    def propose_updates(self, sample_size: int = 200) -> Dict[str, Any]:
        """
        Produce a short plan (human-readable) describing:
          - top 3 recurring user concerns (from KB content)
          - suggested prompt/template changes for the therapy agent
          - count of labeled corrections awaiting review

        Returns a dict with keys: summary, suggestions, counts
        """
        client = MongoClient(self.mongo_uri)
        try:
            coll = client[self.db_name][self.kb_collection]
            # sample recent docs
            cursor = coll.find({}).sort("created_at", -1).limit(sample_size)
            texts = []
            correction_count = 0
            for d in cursor:
                texts.append(d.get("content", "")[:1200])
                if d.get("metadata", {}).get("labels", {}).get("corrected"):
                    correction_count += 1
            joined = "\n\n".join(texts[: max(1, min(30, len(texts)))])

            # Ask the LLM to summarize recurring themes and give concrete suggestions
            prompt = (
                "You are an analyst. Given anonymized conversation snippets, return a STRICT JSON with keys:\n"
                "- top_themes: list of up to 5 short theme strings\n"
                "- therapy_agent_prompt_changes: short list of suggested prompt/template edits (1-3)\n"
                "- high_priority_issues: short list (e.g. 'frequent suicidal ideation mentions')\n\n"
                "Snippets:\n" + joined
            )
            resp = self.llm.invoke([{"role": "user", "content": prompt}])
            raw = resp.content.strip()
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                # best-effort extract
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = {"error": "could not parse llm output", "raw": raw[:400]}

            return {
                "summary": parsed,
                "counts": {"sample_size": sample_size, "corrections_waiting_review": correction_count},
                "generated_at": datetime.utcnow().isoformat(),
            }
        finally:
            client.close()

    # -------------------------
    # Label distribution / drift
    # -------------------------
    def get_label_distribution(self, window: int = 1000) -> Dict[str, int]:
        """
        Compute distribution of classifier labels in recent KB docs.
        Expects metadata.labels.classifier OR metadata.labels.detected_label present.
        """
        client = MongoClient(self.mongo_uri)
        try:
            coll = client[self.db_name][self.kb_collection]
            cursor = coll.find({"metadata.labels": {"$exists": True}}).sort("created_at", -1).limit(window)
            dist = {}
            total = 0
            for d in cursor:
                labels = d.get("metadata", {}).get("labels", {})
                label = labels.get("corrected") or labels.get("classifier") or labels.get("detected_label")
                if not label:
                    label = "unspecified"
                dist[label] = dist.get(label, 0) + 1
                total += 1
            dist["__total__"] = total
            return dist
        finally:
            client.close()

    # -------------------------
    # Convenience: quick in-place re-embedding (if model changed)
    # -------------------------
    def reembed_collection(self, batch_size: int = 256, resume_after_id: Optional[Any] = None) -> Dict[str, Any]:
        """
        Recompute and update embeddings for documents in KB using current embedding model.
        This is a synchronous operation and may be slow; use with care.

        Returns summary {updated: n, skipped: m}.
        """
        client = MongoClient(self.mongo_uri)
        updated = 0
        skipped = 0
        try:
            coll = client[self.db_name][self.kb_collection]
            query = {}
            if resume_after_id is not None:
                query["_id"] = {"$gt": resume_after_id}
            cursor = coll.find(query).sort("_id", ASCENDING)
            to_process = []
            ids = []
            for doc in cursor:
                if "content" not in doc:
                    skipped += 1
                    continue
                to_process.append(doc["content"])
                ids.append(doc["_id"])
                if len(to_process) >= batch_size:
                    embs = self.embedding.embed_documents(to_process)
                    for i, _id in enumerate(ids):
                        coll.update_one({"_id": _id}, {"$set": {"embedding": embs[i], "reembed_at": datetime.utcnow()}})
                        updated += 1
                    to_process = []
                    ids = []
            # leftover
            if to_process:
                embs = self.embedding.embed_documents(to_process)
                for i, _id in enumerate(ids):
                    coll.update_one({"_id": _id}, {"$set": {"embedding": embs[i], "reembed_at": datetime.utcnow()}})
                    updated += 1
            return {"updated": updated, "skipped": skipped}
        finally:
            client.close()
