FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

# Use gunicorn for production — single worker, 2 threads (cost-effective)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 120 main:app
