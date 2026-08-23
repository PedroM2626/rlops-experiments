FROM python:3.14-slim

# System dependencies for Box2D / OpenAI Gym
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment files
COPY . .

# Run orchestrator by default
CMD ["python", "run_all.py"]
