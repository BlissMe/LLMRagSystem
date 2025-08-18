# compare_models.py
# pip install requests numpy pandas scikit-learn
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import os, re, json, time, math, requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import key_param

# =========================
# ---- CONFIGURE THIS -----
# =========================

# Your ngrok/Ollama server
OLLAMA_BASE  = os.getenv("OLLAMA_BASE", "https://d53cb0fd37cb.ngrok-free.app").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-mentalhealth")
OLLAMA_USER  = os.getenv("OLLAMA_USER") or None
OLLAMA_PASS  = os.getenv("OLLAMA_PASS") or None
AUTH         = (OLLAMA_USER, OLLAMA_PASS) if (OLLAMA_USER and OLLAMA_PASS) else None

# OpenAI (for GPT-3.5)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or key_param.openai_api_key 
OPENAI_MODEL   = "gpt-3.5-turbo"    # or a compatible alias in your account

# Your system prompt and (optional) running conversation history text
SYSTEM_TEXT = """You are a friendly assistant who speaks like a kind friend.
Be warm, concise, and non-repetitive. Reply in 1–5 sentences unless asked for more.
Use context only to inform your reply—do NOT quote/echo instructions, tags, or the user.
Ask questions naturally to keep the conversation going, but exactly one gentle question per reply.
Avoid medical/crisis terms unless asked. Output only your reply (no labels)."""

HISTORY_TEXT = ""  # keep empty or dump your chat transcript if you want the same history for both

# Test set of user messages (you can replace with your own)
USER_MESSAGES = [
    "hi, i'm not feeling well lately. i keep feeling low but i don't know why.",
    "i have a heavy academic workload and several projects due, and i don’t know where to start.",
    "i haven’t been sleeping well these last two weeks—either i’m up late or i wake up a lot.",
    "i used to enjoy hanging out and gaming, but lately nothing feels fun.",
    "my appetite’s all over the place—some days i barely eat, other days i overeat.",
    "i feel tired and low-energy almost every day, even when i don’t do much.",
    "i keep blaming myself for small mistakes and feeling guilty most of the time.",
    "i can’t focus on studying; my mind drifts and i reread the same page.",
    "i feel anxious in crowds—my heart races and my thoughts spiral.",
    "i lost someone close last year and the grief still hits me randomly."
]

# Generation knobs (kept the same for both models where possible)
TEMP        = 0.7
NUM_PREDICT = 180  # Ollama-only; GPT uses max_tokens
MAX_TOKENS  = 220  # GPT-only

# =========================
# ====== CALL MODELS ======
# =========================

def call_ollama(system_text: str, user_text: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": f"{HISTORY_TEXT}\n\n{user_text}".strip()},
        ],
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "num_predict": NUM_PREDICT,
            "temperature": TEMP,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "stop": ["</s>", "<s>", "[INST]", "[/INST]", "User:", "Assistant:", "<<", "\nUser"],
        },
    }
    r = requests.post(f"{OLLAMA_BASE}/api/chat",
                      auth=AUTH,
                      headers={"Content-Type": "application/json"},
                      json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}).get("content") or data.get("response") or "").strip()

def call_gpt(system_text: str, user_text: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": f"{HISTORY_TEXT}\n\n{user_text}".strip()},
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


# =========================
# ====== SCORING ==========
# =========================

TAG_PATTERNS = re.compile(r"(</?s>|<s>|<<|<</|<[^>]+>|\[/?INST\]|User:|Assistant:)", re.I)

EMPATHY_LEX = [
    "i'm sorry", "sorry to hear", "that sounds", "i hear you", "i understand",
    "thanks for sharing", "it makes sense", "it's okay", "i'm here for you",
    "that must be", "can imagine", "take your time", "you're not alone",
    "it’s understandable", "appreciate you"
]

def strip_tags(text: str) -> str:
    t = re.sub(r"<<[^>]*>>", " ", text)
    t = re.sub(r"<[^>]*>", " ", t)
    t = re.sub(r"\[/?INST\]", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def empathy_score(text: str) -> int:
    tl = text.lower()
    hits = sum(1 for p in EMPATHY_LEX if p in tl)
    # Presence of "you" and supportive tone bonus
    you = " you " in f" {tl} "
    score = 1 + min(4, hits // 2) + (1 if you and hits >= 1 else 0)
    return max(1, min(5, score))

def clarity_score(text: str) -> int:
    sents = split_sentences(text)
    if not sents: return 1
    words = [w for s in sents for w in s.split()]
    if not words: return 1
    avg_len = len(words) / max(1, len(sents))  # words per sentence
    # 8–22 wps best; degrade outside
    if 8 <= avg_len <= 22: return 5
    if 6 <= avg_len < 8 or 22 < avg_len <= 28: return 4
    if 4 <= avg_len < 6 or 28 < avg_len <= 35: return 3
    if 3 <= avg_len < 4 or 35 < avg_len <= 45: return 2
    return 1

def relevance_score(user_msg: str, reply: str) -> int:
    # TF-IDF cosine similarity as a keyless, robust proxy (0..1)
    vect = TfidfVectorizer(min_df=1, stop_words="english")
    mats = vect.fit_transform([user_msg, reply])
    sim = cosine_similarity(mats[0:1], mats[1:2])[0,0]
    # Map to 1..5
    if sim >= 0.45: return 5
    if sim >= 0.35: return 4
    if sim >= 0.25: return 3
    if sim >= 0.15: return 2
    return 1

def followup_question_score(text: str) -> int:
    sents = split_sentences(text)
    qmarks = text.count("?")
    if qmarks == 1 and sents and sents[-1].endswith("?"):
        return 5
    if qmarks >= 1:
        return 3
    return 1

def no_tags_score(text: str) -> int:
    return 5 if not TAG_PATTERNS.search(text) else 1

def length_score(text: str) -> int:
    n = len(split_sentences(text))
    if 1 <= n <= 5: return 5
    if 6 <= n <= 7: return 2
    return 1

def score_all(user_msg: str, reply: str) -> dict:
    cleaned = strip_tags(reply)
    return {
        "empathy":      empathy_score(cleaned),
        "clarity":      clarity_score(cleaned),
        "relevance":    relevance_score(user_msg, cleaned),
        "one_question": followup_question_score(cleaned),
        "no_leaked_tags": no_tags_score(reply),
        "length_1to5":  length_score(cleaned),
        "reply_clean":  cleaned,
    }

# Optional weighting (equal by default)
WEIGHTS = {
    "empathy": 1.0,
    "clarity": 1.0,
    "relevance": 1.0,
    "one_question": 1.0,
    "no_leaked_tags": 1.0,
    "length_1to5": 1.0,
}

def total_from(parts: dict) -> float:
    return sum(parts[k]*WEIGHTS.get(k,1.0) for k in WEIGHTS)


# =========================
# ====== RUN & LOG ========
# =========================
def run_benchmark():
    rows = []
    for i, user_msg in enumerate(USER_MESSAGES, 1):
        # Call both models
        try:
            mistral = call_ollama(SYSTEM_TEXT, user_msg)
        except Exception as e:
            mistral = f"[ERROR calling Ollama: {e}]"

        try:
            gpt = call_gpt(SYSTEM_TEXT, user_msg)
        except Exception as e:
            gpt = f"[ERROR calling GPT: {e}]"

        # Score
        m_scores = score_all(user_msg, mistral)
        g_scores = score_all(user_msg, gpt)

        rows.append({
            "idx": i,
            "user_message": user_msg,
            "mistral_reply": m_scores["reply_clean"],
            "gpt_reply": g_scores["reply_clean"],
            "m_empathy": m_scores["empathy"],
            "m_clarity": m_scores["clarity"],
            "m_relevance": m_scores["relevance"],
            "m_one_question": m_scores["one_question"],
            "m_no_leaked_tags": m_scores["no_leaked_tags"],
            "m_length_1to5": m_scores["length_1to5"],
            "m_total": total_from(m_scores),
            "g_empathy": g_scores["empathy"],
            "g_clarity": g_scores["clarity"],
            "g_relevance": g_scores["relevance"],
            "g_one_question": g_scores["one_question"],
            "g_no_leaked_tags": g_scores["no_leaked_tags"],
            "g_length_1to5": g_scores["length_1to5"],
            "g_total": total_from(g_scores),
        })

    df = pd.DataFrame(rows)

    # ---- console view (optional) ----
    view_cols = [
        "idx","m_total","g_total",
        "m_empathy","g_empathy",
        "m_clarity","g_clarity",
        "m_relevance","g_relevance",
        "m_one_question","g_one_question",
        "m_no_leaked_tags","g_no_leaked_tags",
        "m_length_1to5","g_length_1to5"
    ]
    print("\n=== Score Summary (higher is better; max per dim = 5) ===\n")
    print(df[view_cols].to_string(index=False))

    # ---- Excel summary sheet ----
    summary = pd.DataFrame({
        "metric": ["empathy","clarity","relevance","one_question","no_leaked_tags","length_1to5","total"],
        "mistral_avg": [
            df["m_empathy"].mean(), df["m_clarity"].mean(), df["m_relevance"].mean(),
            df["m_one_question"].mean(), df["m_no_leaked_tags"].mean(), df["m_length_1to5"].mean(),
            df["m_total"].mean()
        ],
        "gpt_avg": [
            df["g_empathy"].mean(), df["g_clarity"].mean(), df["g_relevance"].mean(),
            df["g_one_question"].mean(), df["g_no_leaked_tags"].mean(), df["g_length_1to5"].mean(),
            df["g_total"].mean()
        ],
    }).round(2)

    # ---- save to Excel ----
    excel_path = "model_comparison_log.xlsx"
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # detailed sheet
        df.to_excel(writer, sheet_name="Detailed", index=False)
        # summary sheet
        summary.to_excel(writer, sheet_name="Summary", index=False)

        # auto-fit columns for both sheets
        for sheet_name, dataframe in {"Detailed": df, "Summary": summary}.items():
            ws = writer.sheets[sheet_name]
            for col_idx, col_name in enumerate(dataframe.columns):
                max_len = max(
                    [len(str(col_name))] +
                    [len(str(v)) for v in dataframe[col_name].astype(str).values]
                )
                ws.set_column(col_idx, col_idx, min(max_len + 2, 60))
            ws.freeze_panes(1, 0)

    # (optional) keep CSV too
    df.to_csv("model_comparison_log.csv", index=False, encoding="utf-8")

    print(f"\nSaved Excel: {excel_path}")
    return df
if __name__ == "__main__":
    print("Running model benchmark on", len(USER_MESSAGES), "messages...")
    df = run_benchmark()
    out_path_csv = os.path.abspath("model_comparison_log.csv")
    out_path_xlsx = os.path.abspath("model_comparison_log.xlsx")
    print("CSV:", out_path_csv)
    print("XLSX:", out_path_xlsx)
