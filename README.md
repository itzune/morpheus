# Morpheus — On-Device Basque Autocomplete

Morpheus is an on-device predictive autocompletion system for Basque (Euskara). It is a 91M-parameter Mamba-2 State Space Model trained from scratch on a curated ~4.62B-token Basque corpus, quantized to a 55 MB GGUF and served via `llama.cpp` on a consumer CPU — 318 tok/s decode and 97 ms end-to-end autocomplete latency on a 2017 laptop, with zero network calls.

The model uses a 4,000-token SentencePiece Unigram vocabulary, chosen so that Basque's agglutinative morphology (root + suffix chains) splits into reusable subwords rather than fusing into opaque atomic tokens.

For the full architecture, training, evaluation, and deployment details, see **[morpheus-on-device-basque-autocompletion.pdf](./morpheus-on-device-basque-autocompletion.pdf)**.

## Models on Hugging Face

| Repo | Format | Description |
|------|--------|-------------|
| [itzune/morpheus](https://huggingface.co/itzune/morpheus) | PyTorch (`safetensors`) | The 91M Mamba-2 checkpoint with the 4K SentencePiece tokenizer |
| [itzune/morpheus-gguf](https://huggingface.co/itzune/morpheus-gguf) | GGUF (Q4_K_M / Q5_K_M) | Quantized models for `llama.cpp` — the deployment format |

> ⚠️ If you serve the GGUF with `llama-server`/`llama-cli`, use **token-ID prompts**, not string prompts. `llama.cpp`'s built-in tokenizer diverges from the reference SentencePiece library on this 4K vocabulary, which drops autocomplete quality ~7×. The [GGUF model card](https://huggingface.co/itzune/morpheus-gguf) documents the correct usage.

## Demos

### Run locally with Docker

The demo bundles a FastAPI server in front of `llama-server` with two interfaces: a greedy ghost-text editor (`/`) and a smartphone-style predictive keyboard (`/keyboard.html`).

```bash
cd demo
docker compose up -d      # CPU (default); add -f docker-compose.gpu.yml for GPU
```

The container auto-downloads the Q4_K_M GGUF from Hugging Face at startup — no local model files needed. Open **http://localhost:9090**.

See [`demo/README.md`](./demo/README.md) and [`docs/demo.md`](./docs/demo.md) for configuration, the HTTP/WebSocket API, and `llama.cpp` version requirements.

### Try it in the browser (WebAssembly)

A WebAssembly build runs the model directly in your browser at **<https://itzune.eus/morpheus-wasm>**. It is slower than the native `llama.cpp` path and still has some known issues, but lets you try Morpheus without installing anything.

## What's in this repo

| Path | Contents |
|------|----------|
| `src/` | Model definition and dataset code (`mamba-ssm` based) |
| `train.py` | Training entry point |
| `eval.py` | Evaluation entry point |
| `config/small.yaml` | Model + training config (4K vocab, 1024 sequence length) |
| `tokenizer/` | Trained 4K SentencePiece model |
| `scripts/pipeline/` | Corpus cleaning, tokenizer training, pretokenization |
| `scripts/export/` | PyTorch → Hugging Face → GGUF export and quantization |
| `scripts/` | Benchmarking, baseline evaluation, cross-lingual and domain eval utilities |
| `exports/` | Exported checkpoints (HF + GGUF, multiple quantizations) |
| `demo/` | FastAPI demo server, Dockerfile, and static frontend |
| `eval/` | Benchmark results, baseline comparisons, CSR and next-word evaluations |
| `docs/` | Research notes: tokenizer fieldwork, data scaling, Android keyboard, demo API |
| `logs/` | Training logs and completion-acceptance logs (used for replay evaluation) |
| `checkpoints/` | Training checkpoints |
| `morpheus-on-device-basque-autocompletion.pdf` | The full write-up |
| `morpheus-on-device-basque-autocompletion.md` | Source markdown for the write-up |
| `pyproject.toml` | Dependencies (managed with [`uv`](https://docs.astral.sh/uv/)) |
