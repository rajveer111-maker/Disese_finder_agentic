# Use public ECR gallery to bypass Docker Hub pull rate limits
FROM public.ecr.aws/docker/library/python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for compiling certain python libs (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies without cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase to container
COPY api.py .
COPY config.py .
COPY models/ ./models/
COPY utils/ ./utils/
COPY sample_data/ ./sample_data/

# Expose FastAPI port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MOCK_AWS=True
ENV PORT=8000

# Run uvicorn server
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
