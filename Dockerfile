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

RUN pip install uv

# Install backend dependencies
RUN make backend

# Install frontend dependencies and build frontend
RUN make frontend

# Expose default Langflow port
EXPOSE 7860

# Use uv to run Langflow in the container
CMD ["uv", "run", "langflow", "run", "--host", "0.0.0.0", "--port", "7860"]

