FROM python:3.10-bullseye

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgtk-3-dev \
    ffmpeg \
    libsndfile1 \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Install dlib from a prebuilt wheel
RUN pip install dlib==19.24.2 --find-links https://github.com/davisking/dlib/releases

# Install face_recognition and other deps
RUN pip install face_recognition==1.3.0 numpy Pillow Click

COPY . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
