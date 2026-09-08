# Use a lightweight Python image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy dependency list first.
# This allows Docker to cache the dependency installation layer.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and trained model.
COPY src ./src
COPY models ./models

# Expose the FastAPI port.
EXPOSE 8000

# Start the FastAPI application.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]