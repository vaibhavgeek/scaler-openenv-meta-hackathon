FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Default: run the FastAPI server for OpenEnv
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
