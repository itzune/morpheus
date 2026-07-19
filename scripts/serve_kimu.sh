#!/bin/bash
# Launch orai-nlp/Gemma-Kimu-2b-base on the L40 via llama-server + Morpheus demo.
# Kimu is a Gemma-2-2b continually pre-trained on Basque (ZelaiHandi corpus).
# Uses the llama-fim backend (BPE tokenizer, Code Llama FIM convention).
# Usage: bash scripts/serve_kimu.sh
set -e

MODEL="/root/morpheus-mamba/models/Gemma-Kimu-2b-base.Q6_K.gguf"
LLAMA_BIN="/root/llama.cpp-fresh/build/bin/llama-server"
export LD_LIBRARY_PATH="/root/llama.cpp-fresh/build/bin"

# Kill anything on 8082/9092
fuser -k 8082/tcp 9092/tcp 2>/dev/null || true
sleep 1

echo "=== Starting llama-server (Kimu 2B Q6_K, all GPU layers) ==="
nohup "$LLAMA_BIN" -m "$MODEL" --host 0.0.0.0 --port 8082 -ngl 99 -c 4096 \
    > /tmp/llama_kimu.log 2>&1 &
LLAMA_PID=$!
echo "  llama-server PID: $LLAMA_PID"

# Wait for llama-server
for i in $(seq 1 30); do
    if curl -s http://localhost:8082/health 2>/dev/null | grep -q "ok"; then
        echo "  llama-server ready!"
        break
    fi
    sleep 1
done

echo "=== Starting Morpheus demo server (backend=llama-fim) ==="
cd /root/morpheus-mamba
MORPHEUS_BACKEND=llama-fim nohup python3 -u demo/server.py \
    --no-launch-llama --host 0.0.0.0 --port 9092 --llama-port 8082 \
    --model "$MODEL" \
    > /tmp/kimu_demo.log 2>&1 &
DEMO_PID=$!
echo "  demo server PID: $DEMO_PID"

sleep 2
echo ""
echo "=== Status ==="
echo "  Kimu 2B (base): http://$(hostname -I | awk "{print \$1}"):9092"
echo "  Endpoints:"
echo "    /                  — greedy ghost-text editor (AR completion)"
echo "    /editor.html       — FIM + AR editor (OpenAI-compatible)"
echo "    /v1/completions    — OpenAI completions API"
echo "    /v1/complete       — thin-client {prefix, suffix} API"
echo "  Logs: /tmp/llama_kimu.log, /tmp/kimu_demo.log"
