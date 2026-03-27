FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for built packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app

# Setup PYTHONPATH so modules can be imported correctly
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Define the default command to run either tests or training (can be overridden)
CMD ["python", "test_chase.py"]
