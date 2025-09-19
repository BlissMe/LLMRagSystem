from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment
from io import BytesIO
import requests
import json
import key_param
import sys
import os
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from textChatMode.chat import ask_question, QueryRequest
from .emotion_model import predict_emotion

router = APIRouter()
OPENAI_API_KEY = key_param.openai_api_key


@router.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    asked_phq_ids: str = Form("[]"),
    history: str = Form(""),
    summaries: str = Form("[]"),
    emotion_history: str = Form("[]")
):
    print("\n===== START /voice-chat DEBUG LOG =====\n")

    asked_ids = json.loads(asked_phq_ids)
    summary_list = json.loads(summaries)
    emotion_list = json.loads(emotion_history)

    # Step 2: Read audio bytes
    audio_bytes = await audio.read()

    # === CASE: No audio (e.g. PHQ-9 button press) ===
    if len(audio_bytes) == 0:
        print("No audio provided. Skipping Whisper + Emotion detection.")

        user_query = history.strip().split("\n")[-1].replace("User: ", "") if history else ""

        # Don't do emotion detection from text
        predicted_emotion = None
        overall_emotion = emotion_list[-1] if emotion_list else None

    # === CASE: Normal audio input ===
    else:
        webm_io = BytesIO(audio_bytes)

        try:
            webm_audio = AudioSegment.from_file(webm_io, format="webm")
            wav_io = BytesIO()
            webm_audio.export(wav_io, format="wav")
            wav_io.seek(0)
            predicted_emotion = predict_emotion(wav_io)
        except Exception as e:
            print("Emotion detection error:", str(e))
            predicted_emotion = "unknown"

        emotion_list.append(predicted_emotion)
        recent_emotions = emotion_list[-3:]
        overall_emotion = Counter(recent_emotions).most_common(1)[0][0]

        # Whisper transcription
        files = {
            'file': (audio.filename, BytesIO(audio_bytes), audio.content_type)
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        data = {"model": "whisper-1", "language": "en"}

        whisper_response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            data=data,
            files=files
        )

        if whisper_response.status_code != 200:
            print("Whisper Transcription Failed:", whisper_response.status_code, whisper_response.text)
            return JSONResponse(status_code=500, content={"error": "Whisper transcription failed"})

        user_query = whisper_response.json()["text"].strip()
        print("Transcribed User Query:", user_query)

    # === Chatbot call ===
    query_data = QueryRequest(
        user_query=user_query,
        history=history,
        summaries=summary_list,
        asked_phq_ids=asked_ids
    )

    ask_result = await ask_question(query_data)

    return {
        "user_query": user_query,
        "bot_response": ask_result["response"],
        "audio_url": ask_result["audio_url"],
        "phq9_questionID": ask_result["phq9_questionID"],
        "phq9_question": ask_result["phq9_question"],
        "current_emotion": predicted_emotion,
        "overall_emotion": overall_emotion,
        "emotion_history": emotion_list
    }


@router.get("/voice-audio")
def voice_audio(path: str):
    return FileResponse(path, media_type="audio/mpeg", filename="bot_reply.mp3")

