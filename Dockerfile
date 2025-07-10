# Dockerfile
# Use a slim Python image to keep the container size down
FROM python:3.9-slim-buster

# Install system dependencies for vmstat and sar (part of procps and sysstat)
# apt-get update is necessary before apt-get install
RUN apt-get update && apt-get install -y \
    procps \
    sysstat \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# This copies app.py, ai_analyzer.py, and the templates/ directory
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5000

# Command to run the application when the container starts
CMD ["python", "app.py"]