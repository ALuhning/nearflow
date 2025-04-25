FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential git

# Install psycopg2 (PostgreSQL adapter for Python)
RUN pip install psycopg[binary]

# Upgrade pip and install project dependencies
RUN pip install --upgrade pip

# Copy local code (from near-model branch)
COPY . .

# Install Langflow in editable mode with all extras
RUN pip install -e . --no-deps

RUN pip install uv

# Install system dependencies for frontend (Node.js and npm)
RUN apt-get update && apt-get install -y nodejs npm

# Install frontend dependencies and build frontend
RUN make install_frontend
RUN make build_frontend

# Expose default Langflow port
EXPOSE 7860

# Use uv to run Langflow in the container
CMD ["uv", "run", "langflow", "run", "--host", "0.0.0.0", "--port", "7860"]

