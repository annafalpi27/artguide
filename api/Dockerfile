# Dockerfile builds your FastAPI app into a container image
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install system dependencies (including libgomp1 for the TLS issue)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /artguide

# Copy uv project files from root
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies using uv
RUN uv sync

# Copy the entire project unless .dockerignore
COPY . .

# Add the project root to PYTHONPATH so imports work correctly
ENV PYTHONPATH=/artguide

# create dir for mount volume
RUN mkdir -p /artguide/tts/models
RUN mkdir -p /tmp && chmod 1777 /tmp

# Set LD_PRELOAD to fix the TLS allocation error
ENV PYTHONPATH=/artguide
ENV LD_PRELOAD=/artguide/.venv/lib/python3.11/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0

# Expose the port
EXPOSE 7005

# Run the application using uv
CMD ["uv", "run", "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "7005"]