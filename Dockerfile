# Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# --allow-releaseinfo-change helps with Debian version shifts (stable -> oldstable)
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Gradio port
EXPOSE 7860

# Command to run the app
# Use server_name 0.0.0.0 for container accessibility
CMD ["python", "app.py"]
