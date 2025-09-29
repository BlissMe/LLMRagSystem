import torch
import torch.nn as nn
import numpy as np
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from io import BytesIO
import librosa
import os
from huggingface_hub import hf_hub_download

# Define emotions
emotions = ['angry', 'sad', 'neutral', 'happy', 'fearful']
EMOTION_LABELS = emotions
LABEL2ID = {label: idx for idx, label in enumerate(emotions)}

# === Model Architecture ===
class EmotionClassifier(nn.Module):
    def __init__(self, num_labels):
        super(EmotionClassifier, self).__init__()

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

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(num_labels=len(emotions)).to(device)

# Download fine-tuned checkpoint from Hugging Face
checkpoint_path = hf_hub_download(
    repo_id="ImashaNawodi/my-wav2vec2-emotion",
    filename="best_model_v3.pth"
)

# Load fine-tuned weights
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# === Load Processor ===
processor = Wav2Vec2Processor.from_pretrained("ImashaNawodi/my-wav2vec2-emotion")

# === Predict Function ===
def predict_emotion(wav_io: BytesIO) -> str:
    try:
        print("Reading audio from BytesIO...")
        wav_io.seek(0)
        audio = AudioSegment.from_file(wav_io, format="wav").set_channels(1)
        print("Audio loaded. Frame rate:", audio.frame_rate)

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples = samples / np.max(np.abs(samples))  # Normalize
        print("Audio samples normalized. Shape:", samples.shape)

        sr = audio.frame_rate
        if sr != 16000:
            print(f"Resampling from {sr} to 16000")
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Extract prosodic features
        print("Extracting pitch using librosa.yin...")
        pitch = librosa.yin(samples, fmin=80, fmax=400, sr=sr)
        pitch_mean = np.mean(pitch[np.isfinite(pitch)])
        print("Pitch mean:", pitch_mean)

        energy = np.mean(samples ** 2)
        duration = len(samples) / sr
        print(f"Energy: {energy}, Duration: {duration}")

        prosody_feat = torch.tensor([[pitch_mean, energy, duration]], dtype=torch.float32).to(device)
        print("Prosody tensor:", prosody_feat)

        # Processor (audio to input tensor)
        print("Processing audio using Wav2Vec2 processor...")
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        if hasattr(inputs, "attention_mask"):
            attention_mask = inputs.attention_mask.to(device)
        else:
            attention_mask = torch.ones_like(input_values).to(device)
        print("Input values shape:", input_values.shape)
        print("Attention mask shape:", attention_mask.shape)

        # Model inference
        print("Running model inference...")
        with torch.no_grad():
            logits = model(input_values=input_values, attention_mask=attention_mask, prosody=prosody_feat)
            print("Logits:", logits)
            pred_id = torch.argmax(logits, dim=-1).item()
            print("Predicted label index:", pred_id)
            return EMOTION_LABELS[pred_id]

    except Exception as e:
        print("Error during prediction:")
        import traceback
        traceback.print_exc()
        return "Error"
