# Morpheus — On-Device Basque Autocomplete

**Can a Basque text-editor autocompletion system run locally on a consumer device?**

This is the question this project investigates. Basque (Euskara) is a low-resource, morphologically agglutinative language isolate for which no on-device multi-token completion system exists. The answer is worked out in three stages:

1. **Survey the autocompletion landscape.** Production systems span three paradigms, each with a distinct architecture and deployment profile: server-side multi-token completion (GitHub Copilot, Google Smart Compose), on-device next-word prediction (Gboard, a 1.4M-parameter model specialized for smartphone keyboards), and on-device multi-token continuation (a Smart Compose–equivalent for desktop editors). None of these targets Basque, and the on-device multi-token paradigm has not been attempted for an agglutinative language.

2. **Analyze architecture options for Basque.** Two strategic paths present themselves. **Adapt an existing Basque LLM** — the HiTZ center's Latxa models (Llama-3.1-8B–based) — where a critical distinction emerges: Fill-in-the-Middle (FIM), the objective needed for cursor-mid-text completion, is a pretraining-style objective that conflicts with instruction tuning, so the instruct-vs-base choice is decisive. Or **train a new architecture from scratch**, where the on-device parameter budget (≤300 MB, constant latency) rules out large Transformers with their KV-cache latency variance, favoring State Space Models (Mamba-2) with O(1) per-step inference.

3. **Train Morpheus and benchmark against Latxa.** This repo pursues the from-scratch path. **Morpheus** is a 91M-parameter Mamba-2 model trained on a curated ~4.62B-token Basque corpus, quantized to a 55 MB GGUF and served via `llama.cpp` on a consumer CPU — 318 tok/s decode and 97 ms end-to-end autocomplete latency on a 2017 laptop, with zero network calls. It is then benchmarked head-to-head against the existing general-purpose alternative, **Latxa 8B (base)** (HiTZ/Latxa-Llama-3.1-8B).

The comparison establishes a **two-tier deployment architecture, fixed by hardware rather than preference**:

- **Morpheus (91M, 55 MB)** is the only model that runs on the edge (40.7 tok/s on a 2017 laptop CPU), but its quality ceiling is visible on generative prose — its sweet spot is **formulaic completion** (email openings/closings, fixed collocations) and **domain-specialized fine-tunes** where a narrow distribution raises the hit rate on the patterns it can actually learn.
- **Latxa 8B (base)** is the server-side quality ceiling (+8.35 CSR points, cross-domain competence without specialization) but is GPU-bound, collapsing to 2.9 s/request on the same laptop. It is the candidate for FIM continued pretraining, and a practical advantage is that it rides on a **standard LLM** (Llama-3.1) with its mature ecosystem (quantization, serving, editor plugins, eval harnesses) shared with mainstream work.

The model uses a 4,000-token SentencePiece Unigram vocabulary, chosen so that Basque's agglutinative morphology (root + suffix chains) splits into reusable subwords rather than fusing into opaque atomic tokens.

See the write-up for the full architecture, training, evaluation, and deployment details: **[morpheus-on-device-basque-autocompletion.pdf](./morpheus-on-device-basque-autocompletion.pdf)** (concise, default) covers the survey, the architecture-selection decision, the two-model benchmark, and the evaluation-methodology findings (a "fertility paradox" and a "CSR paradox" that make perplexity the only reliable checkpoint-ranking metric for agglutinative languages). A longer, detailed version is available as **[morpheus-on-device-basque-autocompletion-full.pdf](./morpheus-on-device-basque-autocompletion-full.pdf)**.

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

### Use it in Obsidian (desktop editor plugin)

A ghost-text autocomplete plugin for [Obsidian](https://obsidian.md) connects to the demo server and renders inline suggestions as you type — **Tab** to accept, **Esc** to dismiss. The plugin is backend-agnostic: point it at `localhost:9090` for the on-device Mamba-2 model, or a GPU server URL for Latxa 8B — just change the Server URL setting.

```bash
cd demo/obsidian-morpheus-plugin
npm install && npm run build
# Copy main.js, manifest.json, styles.css into your vault's
# .obsidian/plugins/morpheus-autocomplete/ folder
```

See [`demo/obsidian-morpheus-plugin/README.md`](./demo/obsidian-morpheus-plugin/README.md) for install details and dev setup.

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
| `demo/obsidian-morpheus-plugin/` | Obsidian ghost-text plugin (CodeMirror 6, backend-agnostic) |
| `eval/` | Benchmark results, baseline comparisons, CSR and next-word evaluations |
| `docs/` | Research notes + [`ARCHITECTURE.md`](./docs/ARCHITECTURE.md) (component ownership), [`TRAJECTORY.md`](./docs/TRAJECTORY.md) (desktop-editor direction), tokenizer fieldwork, data scaling, eval strategies |
| `logs/` | Training logs and completion-acceptance logs (used for replay evaluation) |
| `checkpoints/` | Training checkpoints |
| `morpheus-on-device-basque-autocompletion.pdf` | The write-up (concise, default) |
| `morpheus-on-device-basque-autocompletion.md` | Source markdown for the concise write-up |
| `morpheus-on-device-basque-autocompletion-full.pdf` | The detailed write-up |
| `morpheus-on-device-basque-autocompletion-full.md` | Source markdown for the detailed write-up |
| `pyproject.toml` | Dependencies (managed with [`uv`](https://docs.astral.sh/uv/)) |
