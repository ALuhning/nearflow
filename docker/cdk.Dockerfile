FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install Poetry
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install gcc g++ curl build-essential postgresql-server-dev-all -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -
# # Add Poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Always copy pyproject.toml
COPY pyproject.toml ./

# Conditionally copy poetry.lock only if it exists
# Safe fallback for environments that don't support --ignore-missing
# We don't use COPY here to avoid build error
# Instead use RUN cp if it exists in the build context
RUN test -f poetry.lock && cp poetry.lock . || echo "No poetry.lock found, skipping"

# Copy the rest of the application codes
COPY ./ ./

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

RUN poetry add pymysql

COPY docker/container-cmd-cdk.sh ./container-cmd-cdk.sh
RUN chmod +x ./container-cmd-cdk.sh

CMD ["sh", "./container-cmd-cdk.sh"]
