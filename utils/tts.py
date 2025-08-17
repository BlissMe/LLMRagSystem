from gtts import gTTS
import tempfile

def generate_tts_audio(text: str) -> str:
    if not text or not text.strip():
        return None
    tts = gTTS(text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name
