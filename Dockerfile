# Use Python 3.10 (compatible with prebuilt wheels)
FROM python:3.10

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed for dlib and face_recognition
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgtk-3-dev \
    libboost-all-dev \
    ffmpeg \
    libsndfile1 \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, wheel, setuptools
RUN pip install --upgrade pip setuptools wheel

# Install dlib from a prebuilt wheel
RUN pip install --no-cache-dir dlib==19.24.2

# Install the rest of Python dependencies
RUN pip install --no-cache-dir face_recognition==1.3.0 numpy Pillow Click

# Copy app code
COPY . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
