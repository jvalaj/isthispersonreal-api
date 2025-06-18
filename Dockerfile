FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers torch Pillow

# Create app directory
WORKDIR /app

# Copy everything
COPY . .

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
