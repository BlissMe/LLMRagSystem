FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libatlas3-base \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install prebuilt dlib wheel + face_recognition
RUN pip install --upgrade pip
RUN pip install --no-cache-dir dlib==19.24.2 face_recognition==1.3.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
