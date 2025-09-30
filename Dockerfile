FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (build tools + ffmpeg for face_recognition)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir "dlib==19.24.2" face_recognition==1.3.0

# Copy app code
COPY . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
