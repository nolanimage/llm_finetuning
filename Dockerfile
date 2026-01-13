# Dockerfile for ChatGLM3 Fine-tuning
# Supports both CPU and GPU training

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version by default, can be overridden for GPU)
# For GPU support, use: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torch torchvision torchaudio

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/logs /app/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Expose port for potential API server (if added later)
EXPOSE 8000

# Default command
CMD ["python", "train.py"]
