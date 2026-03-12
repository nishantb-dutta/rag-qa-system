# Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for ChromaDB and other libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create persistent directories
RUN mkdir -p documents chroma_store

# Pre-download the embedding model to the image (optimization)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose Gradio port
EXPOSE 7860

# Command to run the app
# Use server_name 0.0.0.0 for container accessibility
CMD ["python", "app.py"]
