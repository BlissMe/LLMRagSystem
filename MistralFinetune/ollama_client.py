# services/ollama_client.py
import os, json, requests
from jsonschema import validate  # pip install jsonschema

BASE = os.getenv("OLLAMA_BASE", "https://e037d0b95762.ngrok-free.app")
AUTH = None  # or ("user","pass") if you used --basic-auth

STOPS = ["</s>", "[INST]", "User:", "\nUser"]

def mistral_generate(prompt: str, temperature: float = 0.3, num_predict: int = 120) -> str:
    r = requests.post(
        f"{BASE.rstrip('/')}/api/generate",
        auth=AUTH,
        headers={"Content-Type": "application/json"},
        json={
            "model": "mistral-mentalhealth",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "num_predict": num_predict,      # correct key for Ollama
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "stop": STOPS,
            },
        },
        timeout=120,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def mistral_generate_json(user_prompt: str, schema: dict) -> dict:
    instruct = (
        "Return strictly valid JSON only. No explanations.\n"
        f"Match this schema as closely as possible:\n{json.dumps(schema)}\n\n"
        f"{user_prompt}\n"
    )
    r = requests.post(
        f"{BASE.rstrip('/')}/api/generate",
        auth=AUTH,
        headers={"Content-Type": "application/json"},
        json={
            "model": "mistral-mentalhealth",
            "prompt": instruct,
            "stream": False,
            "format": "json",
            "options": {
                "num_ctx": 4096,
                "num_predict": 200,
                "temperature": 0.2,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "stop": STOPS,
            },
        },
        timeout=120,
    )
    r.raise_for_status()
    obj = json.loads(r.json()["response"])
    validate(instance=obj, schema=schema)
    return obj
