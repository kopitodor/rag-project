FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port uvicorn will run on
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]