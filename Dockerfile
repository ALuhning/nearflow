FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential git

# Build docker image
RUN make docker_build



