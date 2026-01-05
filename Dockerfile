# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (e.g., for ODBC/SQL Server if needed)
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable
ENV MODULE_NAME="app.main"
ENV VARIABLE_NAME="app"
ENV PORT="8000"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
