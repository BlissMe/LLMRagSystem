# Use official Python slim image (3.11 for scipy>=1.16)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
        gfortran \
        libatlas-base-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Start command for FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
