#!/bin/sh
# Docker entrypoint: download GGUF model from HuggingFace if not present, then start services.
#
# The model is cached in a Docker volume (mounted at /app/models) so it only
# downloads once. Subsequent container starts use the cached file.
#
# Env vars:
#   MORPHEUS_MODEL    GGUF filename (e.g. morpheus-v2-mamba.Q4_K_M.gguf)
#   MORPHEUS_NGL      GPU layers to offload (0=CPU, 99=all GPU)
#   HF_REPO           HuggingFace repo for GGUF models (default: itzune/morpheus-gguf)
#   HF_TOKEN          Optional HF token for private repos (not needed for public)

set -e

MODEL="${MORPHEUS_MODEL:-morpheus-v2-mamba.Q4_K_M.gguf}"
NGL="${MORPHEUS_NGL:-0}"
HF_REPO="${HF_REPO:-itzune/morpheus-gguf}"
MODEL_DIR="/app/models"
MODEL_PATH="${MODEL_DIR}/${MODEL}"

echo "=== Morpheus v2 Demo ==="
echo "  Model: ${MODEL}"
echo "  Repo:  ${HF_REPO}"
echo "  NGL:   ${NGL} (0=CPU only)"

# ── Download model if not present ──
if [ ! -f "${MODEL_PATH}" ]; then
    echo ""
    echo "  Model not found locally. Downloading from HuggingFace..."
    mkdir -p "${MODEL_DIR}"

    # Try hf CLI first (if available), fall back to huggingface_hub Python
    if command -v hf >/dev/null 2>&1; then
        echo "  Using hf CLI..."
        hf download "${HF_REPO}" "${MODEL}" \
            --local-dir "${MODEL_DIR}" \
            ${HF_TOKEN:+--token "${HF_TOKEN}"}
    else
        echo "  Using Python huggingface_hub..."
        python3.11 -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id='${HF_REPO}',
    filename='${MODEL}',
    local_dir='${MODEL_DIR}',
    token=os.environ.get('HF_TOKEN')
)
print(f'Downloaded to: {path}')
"
    fi
    echo "  Download complete: $(du -h ${MODEL_PATH} | cut -f1)"
else
    echo "  Model found in cache: $(du -h ${MODEL_PATH} | cut -f1)"
fi

echo ""
echo "  Starting llama-server..."
/opt/llama-server/llama-server \
    -m "${MODEL_PATH}" \
    --host 0.0.0.0 --port 8080 -ngl "${NGL}" &

# Wait for llama-server to be ready
echo "  Waiting for llama-server..."
for i in $(seq 1 30); do
    sleep 0.5
    if curl -s http://localhost:8080/health | grep -q "ok" 2>/dev/null; then
        echo "  llama-server is ready!"
        break
    fi
done

echo ""
echo "  Starting demo server..."
exec python3.11 -u demo/server.py --no-launch-llama --port 9090 --llama-port 8080
