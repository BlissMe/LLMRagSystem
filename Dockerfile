# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only runtime dependencies (avoid build tools unless necessary)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libatlas3-base \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy only app code (ignore unnecessary files via .dockerignore)
COPY . .

# Expose Hugging Face default port
EXPOSE 7860

# Start FastAPI using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
