# Use Python slim image
FROM python:3.10-slim   # Important: use 3.10 for prebuilt dlib wheels

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libatlas3-base \
    curl \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Install dlib and face_recognition first (prebuilt wheel)
RUN pip install --no-cache-dir dlib==19.24.2 face_recognition==1.3.0

# Install the rest of your dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose port
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
