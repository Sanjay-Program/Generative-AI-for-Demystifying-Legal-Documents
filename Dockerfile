# Dockerfile

# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Cloud Run will use
EXPOSE 8080

# Command to run the application using uvicorn
# This listens on all network interfaces (0.0.0.0) and on the port provided by Cloud Run
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}