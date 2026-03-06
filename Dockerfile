# Use Python 3.11 which is more stable for these libraries
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies needed for some python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

CMD ["python", "src/server.py", "1.1"]