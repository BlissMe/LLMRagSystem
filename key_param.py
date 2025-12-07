import os
from dotenv import load_dotenv
from langchain.chains import llm_bash

load_dotenv()

openai_api_key = os.getenv("openai_api_key")
MONGO_URI = os.getenv("MONGO_URI")
assemblyai_api_key = os.getenv("assemblyai_api_key")
elevenlabs_api_key = os.getenv("elevenlabs_api_key")
elevenlabs_voice_id = os.getenv("elevenlabs_voice_id")
llm_base = os.getenv("LLM_BASE")

required_keys = {
    "openai_api_key": openai_api_key,
    "MONGO_URI": MONGO_URI,
    "assemblyai_api_key": assemblyai_api_key,
    "elevenlabs_api_key": elevenlabs_api_key,
    "elevenlabs_voice_id": elevenlabs_voice_id,
    "llm_base": llm_base,
}

missing_keys = [k for k, v in required_keys.items() if not v]
if missing_keys:
    raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")