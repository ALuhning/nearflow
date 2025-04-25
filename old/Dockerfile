FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential git

# Build docker image
RUN docker build -f docker_nearflow/build_and_push.Dockerfile -t ghcr.io/aluhning/nearflow:latest .



