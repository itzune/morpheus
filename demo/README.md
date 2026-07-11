# Morpheus v2 Mamba — Demo

Ultra-low-latency Basque autocomplete with Mamba-2 (91M params) via llama.cpp.

Two interfaces:
- **Greedy ghost-text editor** (`/`) — multi-token continuation as ghost text, Tab to accept.
- **Predictive keyboard** (`/keyboard.html`) — smartphone-style next-word chips with virtual Basque keyboard.

## Quickstart (Docker)

```bash
cd demo

# CPU (default)
docker compose up -d

# GPU
docker compose -f docker-compose.gpu.yml up -d
```

The container auto-downloads the GGUF model from HuggingFace (`itzune/morpheus-gguf`)
at startup — no local model files needed. Open **http://localhost:9090**.

See **[docs/demo.md](../docs/demo.md)** for full documentation (architecture, API,
Docker configuration, llama.cpp version requirements, inference engineering).

## Without Docker

```bash
cd demo
uv sync
uv run python server.py
```

Requires a GGUF model in `../exports/` and `llama-server` on PATH (or set
`LLAMA_SERVER_PATH`). The model defaults to `morpheus-v2-mamba.Q4_K_M.gguf`.
