# llm_utils.py
import os
import json
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your .env or environment")

# Try to use LangChain (newer versions) first, but gracefully fallback to openai if needed.
USE_LANGCHAIN = False
try:
    # langchain v0.0x+ style imports
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    # create an LLM wrapper
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # safe default
    USE_LANGCHAIN = True

    # Prepare a prompt template expecting a JSON list string
    sentiment_template = PromptTemplate(
        input_variables=["items_json"],
        template=(
            "You are given a JSON array of textual user feedback about a therapy, e.g. "
            '["Felt Good", "Not helpful", ...]. For each feedback item, rate it on a scale 0..1 '
            "where 0 = very negative/not helpful, 0.5 = neutral, 1 = extremely positive/highly helpful. "
            "Return ONLY a JSON array of numbers (one number per feedback) with values between 0 and 1. "
            "Example input: {items_json}"
        )
    )
    chain = LLMChain(llm=llm, prompt=sentiment_template)
except Exception:
    # LangChain not available / incompatible — fallback to direct openai
    USE_LANGCHAIN = False

if not USE_LANGCHAIN:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception as e:
        raise RuntimeError("LangChain is not usable and openai package import failed: " + str(e))


def _heuristic_score(fb: str) -> float:
    s = (fb or "").lower()
    if not s.strip():
        return 0.5
    positive = ["good", "great", "helpful", "relaxed", "calm", "felt better", "worked", "improved", "awesome"]
    negative = ["bad", "worse", "not helpful", "didn't", "didnt", "no help", "terrible", "awful"]
    if any(p in s for p in positive):
        return 0.8
    if any(n in s for n in negative):
        return 0.2
    return 0.5


def score_feedback_with_llm(feedback_list):
    """
    feedback_list: list[str]
    returns: list[float] between 0 and 1 (same order)
    Uses LangChain+ChatOpenAI when available, otherwise uses openai.ChatCompletion as fallback.
    """
    if not feedback_list:
        return []

    # Prepare JSON payload
    payload = json.dumps(feedback_list, ensure_ascii=False)

    # If LangChain is available, use it
    if USE_LANGCHAIN:
        try:
            resp = chain.run(items_json=payload)
            # extract the first JSON array in the response
            text = resp.strip()
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1 and end > start:
                arr_text = text[start:end+1]
                arr = json.loads(arr_text)
                scores = [max(0.0, min(1.0, float(x))) for x in arr]
                if len(scores) == len(feedback_list):
                    return scores
            # fallback to heuristic if parsing fails
        except Exception:
            pass

    # Fallback: use openai.ChatCompletion API directly
    try:
        system_prompt = (
            "You are an assistant that converts textual therapy feedback into a numeric helpfulness score between 0 and 1. "
            "Return only a JSON array of numbers (one per input), e.g. [0.8, 0.5, 0.2]."
        )
        user_prompt = f"Here are the feedback items as a JSON array: {payload}\n\nReturn the array of scores."
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=500,
        )
        text = completion.choices[0].message.content.strip()
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            arr_text = text[start:end+1]
            arr = json.loads(arr_text)
            scores = [max(0.0, min(1.0, float(x))) for x in arr]
            if len(scores) == len(feedback_list):
                return scores
    except Exception:
        # fallthrough to heuristics
        pass

    # Final fallback: heuristics per feedback
    return [_heuristic_score(fb) for fb in feedback_list]
