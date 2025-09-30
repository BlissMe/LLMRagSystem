FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install dlib using prebuilt wheel
RUN pip install --no-cache-dir dlib==19.24.2-cp310-cp310-manylinux2014_x86_64.whl \
    face_recognition==1.3.0

# Copy requirements if you have other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
