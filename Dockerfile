# Use a slim Debian-based Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libatlas3-base \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install prebuilt dlib and face_recognition wheels (no compilation)
RUN pip install --no-cache-dir \
    dlib==19.24.2 --only-binary :all: \
    face_recognition==1.2.3 --only-binary :all:

# Copy requirements.txt and install the rest of dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose the port
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
