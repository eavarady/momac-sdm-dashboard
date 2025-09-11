# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Ensure output is not buffered and set default port
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONPATH=/app/src

# Set the working directory in the container
WORKDIR /app

# Install system dependencies only if needed (kept minimal for slim image)
# Uncomment if you later need build tools for heavy libs (e.g., Prophet backend compiling Stan):
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends build-essential g++ make \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port Streamlit will listen on
EXPOSE 8080

# Streamlit needs to bind to 0.0.0.0 on Cloud Run and use the PORT env var
CMD ["bash", "-lc", "streamlit run src/dashboard/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
