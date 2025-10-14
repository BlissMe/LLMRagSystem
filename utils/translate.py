# utils/translate.py
from langchain_openai import ChatOpenAI
import key_param

# Use GPT to translate Sinhala→English
def si_to_en(text: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     openai_api_key=key_param.openai_api_key,
                     temperature=0)
    resp = llm.invoke([{"role": "user",
                        "content": f"Translate this Sinhala text to English:\n{text}"}])
    return resp.content.strip()

# Use GPT to translate English→Sinhala
def en_to_si(text: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     openai_api_key=key_param.openai_api_key,
                     temperature=0)
    resp = llm.invoke([{"role": "user",
                        "content": f"Translate this English text to Sinhala (keep meaning, not literal):\n{text}"}])
    return resp.content.strip()
