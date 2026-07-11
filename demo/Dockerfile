# Morpheus v2 Mamba Demo
#
# Configurable CPU/GPU build via build args (run from repo root):
#   CPU (default):  docker build -f demo/Dockerfile .
#   GPU:            docker build -f demo/Dockerfile . --build-arg GGML_CUDA=ON \
#                     --build-arg BUILD_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 \
#                     --build-arg RUNTIME_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04
#
# Or use docker-compose (run from demo/ directory):
#   CPU:  docker compose up -d
#   GPU:  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
#
# The GGUF model is downloaded automatically from HuggingFace (itzune/morpheus-gguf)
# on first start and cached in a Docker volume. No local model files needed.

# ---- Build stage: compile llama.cpp ----
ARG BUILD_IMAGE=ubuntu:22.04
ARG RUNTIME_IMAGE=ubuntu:22.04
ARG GGML_CUDA=OFF
FROM ${BUILD_IMAGE} AS builder

ARG GGML_CUDA=OFF

RUN apt-get update && apt-get install -y \
    git wget cmake build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp with Mamba-2 support (requires >= commit dc2187d48, 2025-07-04).
# Earlier builds have an SSM_SCAN bug producing silently incorrect greedy
# outputs for Mamba-2 models with n_groups > 1. See docs/demo.md for details.
WORKDIR /opt
RUN git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
WORKDIR /opt/llama.cpp
RUN cmake -B build -DGGML_CUDA=${GGML_CUDA} -DGGML_BLAS=OFF
RUN cmake --build build --config Release -j$(nproc) --target llama-server

# ---- Runtime stage ----
FROM ${RUNTIME_IMAGE}

# Install Python and runtime deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    curl psmisc \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

# Copy built llama-server and its shared libraries from builder stage
COPY --from=builder /opt/llama.cpp/build/bin/ /opt/llama-server/
ENV LLAMA_SERVER_BINARY=/opt/llama-server/llama-server
ENV LD_LIBRARY_PATH=/opt/llama-server

# ---- Demo app deps ----
WORKDIR /app
RUN python3.11 -m pip install --no-cache-dir \
    fastapi==0.115.* \
    uvicorn==0.34.* \
    websockets==13.* \
    httpx==0.28.* \
    sentencepiece==0.2.* \
    huggingface_hub==0.25.*

COPY demo/server.py demo/
COPY demo/static/ demo/static/
COPY demo/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Default env (overridable via docker-compose or docker run)
# MORPHEUS_NGL: GPU layers to offload (0=CPU only, 99=all layers for GPU)
ENV MORPHEUS_MODEL=morpheus-v2-mamba.Q4_K_M.gguf
ENV MORPHEUS_NGL=0
ENV HF_REPO=itzune/morpheus-gguf

EXPOSE 9090

ENTRYPOINT ["/app/docker-entrypoint.sh"]
