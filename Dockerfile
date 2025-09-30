FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (runtime only, no heavy dev tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libatlas3-base \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose Hugging Face default port
EXPOSE 7860

# Start FastAPI with uvicorn (no reload in production)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
