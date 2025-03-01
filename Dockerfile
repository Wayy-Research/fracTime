FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy the entire application first
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# No need to copy files again as we've already copied them above

# Health check endpoint is already copied with the rest of the application

# Expose API port
EXPOSE 5000

# Environment variables
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

# Run the compute API server
CMD ["python3", "compute_server.py"]