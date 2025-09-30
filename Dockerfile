# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    cmake \
    git \
    python3-dev \
    libgtk-3-dev \
    libboost-all-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install latest CMake (in case default is too old)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.5/cmake-3.27.5-linux-x86_64.sh \
    && sh cmake-3.27.5-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.27.5-linux-x86_64.sh

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python packages
RUN pip install --no-cache-dir \
    dlib==19.24.2 \
    face_recognition==1.3.0 \
    numpy \
    Pillow \
    Click

# Copy app code
COPY . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
