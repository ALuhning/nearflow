FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential git

# Upgrade pip and install project dependencies
RUN pip install --upgrade pip

# Copy local code (from near-model branch)
COPY . .

# Install Langflow in editable mode with all extras
RUN pip install -e . --no-deps
RUN pip install gunicorn
# Expose default Langflow port
EXPOSE 7860

# Production entrypoint using Gunicorn + Uvicorn worker
CMD ["gunicorn", "langflow.main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:7860", "--workers=1"]
