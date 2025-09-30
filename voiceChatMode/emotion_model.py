from fastapi import FastAPI, UploadFile, File
import torch
from io import BytesIO

app = FastAPI()

# Globals
model = None
processor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_LABELS = ['angry', 'sad', 'neutral', 'happy', 'fearful']

def load_model():
    global model, processor
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    from huggingface_hub import hf_hub_download
    import torch.nn as nn

    class EmotionClassifier(nn.Module):
        def __init__(self, num_labels):
            super().__init__()
            MODEL_PATH = "ImashaNawodi/my-wav2vec2-emotion"
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_PATH)
            self.dropout = nn.Dropout(0.3)
            self.fc_audio = nn.Linear(self.wav2vec2.config.hidden_size, 128)
            self.fc_prosody = nn.Linear(3, 32)
            self.fc_combined = nn.Linear(128 + 32, num_labels)

        def forward(self, input_values, attention_mask, prosody):
            outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
            audio_feat = outputs.last_hidden_state.mean(dim=1)
            audio_proj = self.fc_audio(self.dropout(audio_feat))
            prosody_proj = self.fc_prosody(self.dropout(prosody))
            combined = torch.cat([audio_proj, prosody_proj], dim=1)
            logits = self.fc_combined(combined)
            return logits

    model = EmotionClassifier(num_labels=len(EMOTION_LABELS)).to(device)

    # Load checkpoint
    checkpoint_path = hf_hub_download(
        repo_id="ImashaNawodi/my-wav2vec2-emotion",
        filename="best_model_v3.pth"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    processor = Wav2Vec2Processor.from_pretrained("ImashaNawodi/my-wav2vec2-emotion")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, processor
    if model is None or processor is None:
        load_model()  # Lazy load on first request

    from pydub import AudioSegment
    import numpy as np
    import torch
    import librosa

    wav_io = BytesIO(await file.read())
    audio = AudioSegment.from_file(wav_io, format="wav").set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.max(np.abs(samples))
    sr = audio.frame_rate
    if sr != 16000:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Extract simple prosody features
    pitch = librosa.yin(samples, fmin=80, fmax=400, sr=sr)
    pitch_mean = np.mean(pitch[np.isfinite(pitch)])
    energy = np.mean(samples ** 2)
    duration = len(samples) / sr
    prosody_feat = torch.tensor([[pitch_mean, energy, duration]], dtype=torch.float32).to(device)

    # Processor
    inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = getattr(inputs, "attention_mask", torch.ones_like(input_values)).to(device)

    # Model inference
    with torch.no_grad():
        logits = model(input_values=input_values, attention_mask=attention_mask, prosody=prosody_feat)
        pred_id = torch.argmax(logits, dim=-1).item()

    return {"emotion": EMOTION_LABELS[pred_id]}
