# Dockerfile

# Start a new container image
# slim: smaller image
FROM python:3.10-slim

# Set the working directory inside the container
# All subsequent commands will run from /app
WORKDIR /app

# Copy dependency list 
COPY requirements.txt /app/requirements.txt

# Install runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src /app/src
COPY api /app/api


# Make src importable inside the container
ENV PYTHONPATH=/app/src

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

