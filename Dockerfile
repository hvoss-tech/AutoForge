# AutoForge web UI with NVIDIA GPU support.
#
#   docker compose up -d --build        -> http://localhost:8000
#
# or without compose:
#
#   docker build -t autoforge .
#   docker run --gpus all -p 8000:8000 -v autoforge-data:/data autoforge
#
# The CLI is still available by overriding the command:
#
#   docker run --gpus all -v "$PWD:/work" -w /work autoforge autoforge --input_image ...
#
# GPU access needs the NVIDIA Container Toolkit on the host. Without a GPU the
# container still starts and falls back to the CPU. For AMD GPUs or a CPU-only
# image see TORCH_INDEX_URL below and docker-compose.rocm.yml /
# docker-compose.cpu.yml.

# ---------------------------------------------------------------------------
# Stage 1 - build the React frontend
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /app/webui/frontend

COPY webui/frontend/package.json webui/frontend/package-lock.json ./
RUN npm ci

COPY webui/frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 - Python runtime + FastAPI server
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# PyTorch wheel index, which picks the backend the image runs on:
#   .../whl/cu128    NVIDIA, Turing and newer (default)
#   .../whl/cu126    NVIDIA, Pascal/Volta (GTX 10xx)
#   .../whl/rocm7.1  AMD GPUs via ROCm (x86_64 only)
#   .../whl/cpu      CPU only; also the one to use on Apple Silicon, since
#                    containers on macOS cannot reach Metal (MPS)
# docker-compose.rocm.yml and docker-compose.cpu.yml set this for you.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Harmless on non-NVIDIA hosts; tell the NVIDIA runtime what to expose.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    WEBUI_HOST=0.0.0.0 \
    WEBUI_PORT=8000 \
    # All persistent state lives under /data (mount a volume there): jobs and
    # project state, uploaded images, the filament library, the telemetry id
    # (~/.autoforge) and downloaded model weights (~/.cache).
    HOME=/data \
    AUTOFORGE_WEBUI_CHECKPOINTS_DIR=/data/checkpoints \
    AUTOFORGE_WEBUI_UPLOADS_DIR=/data/uploads \
    AUTOFORGE_WEBUI_LIBRARY_DIR=/data/filament_library

# build-essential stays in the runtime image: Triton compiles its CUDA
# launcher stubs with gcc the first time the fused kernels run.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch in its own layer so source changes don't re-download it.
RUN pip install --upgrade pip \
    && pip install "torch>=2.9.1" "torchvision>=0.21.0" --index-url "$TORCH_INDEX_URL"

COPY pyproject.toml README.md ./
COPY src ./src
# Editable install: server.py locates the built frontend relative to its own
# source file (<repo>/webui/frontend/dist), so the package has to run from
# /app/src rather than from site-packages.
RUN pip install -e .

COPY --from=frontend-builder /app/webui/frontend/dist ./webui/frontend/dist

RUN mkdir -p /data
VOLUME ["/data"]
WORKDIR /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${WEBUI_PORT}/api/system/health" || exit 1

CMD ["sh", "-c", "exec uvicorn autoforge.webui.server:app --host \"$WEBUI_HOST\" --port \"$WEBUI_PORT\""]
