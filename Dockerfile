# Use slim Debian-based Python
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for dlib, face_recognition, audio/video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    ffmpeg \
    libsndfile1 \
    libatlas-base-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install dlib (build from source) and face_recognition
RUN pip install --no-cache-dir \
    dlib==19.24.2 \
    face_recognition==1.3.0

# Copy requirements.txt and install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
