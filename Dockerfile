# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the app directory into the container
COPY ./app/* /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Set environment variable for logging
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV LOG_LEVEL=INFO

# Run web.py using gunicorn when the container launches
# CMD ["python","web.py"]
CMD ["gunicorn", "-c", "/app/gunicorn_config.py", "web:app"]
