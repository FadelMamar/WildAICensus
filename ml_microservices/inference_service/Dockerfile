# ---------- Stage 1: Builder ----------
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

# Install build-time dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Prepare working directory
WORKDIR /build

# Copy only requirements for caching
COPY pyproject.toml requirements.txt ./

# Install Python dependencies into /install
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
RUN uv pip install --no-cache-dir -e . \
 && uv pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="fadel.seydou@gmail.com"
LABEL version="0.0.1"
LABEL description="Inference service using litserve"

# Create workspace directory
WORKDIR /workspace

# Copy installed dependencies from builder
COPY --from=builder /bin/uv /bin/uvx /bin/
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code only
COPY main.py src/ ./

# Create and mount model weights directory
RUN mkdir -p /model_weights
VOLUME ["/model_weights"]

# Expose inference port
EXPOSE 4141

ENV MLFLOW_TRACKING_URI=http://mlflow_service:5000
ENV INFERENCE_PORT=4141
ENV MODEL_NAME="labeler"
ENV MODEL_ALIAS="demo"
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000

RUN apt-get update \
 && apt-get install -y libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Entrypoint
CMD ["uv", "run", "python", "main.py"]
