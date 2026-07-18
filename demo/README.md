# Morpheus Mamba — Demo

Ultra-low-latency Basque autocomplete with Mamba-2 (91M params) via llama.cpp.

## Interfaces

| Path | What |
|------|------|
| [`/`](.) | Greedy ghost-text editor (Smart Compose style, append-only) |
| [`/editor.html`](editor.html) | **Ghost-text editor — FIM infill + AR append** (reference thin client, see below) |
| [`/keyboard.html`](keyboard.html) | Predictive keyboard — smartphone-style next-word chips with virtual Basque keyboard |

## Quickstart (Docker)

```bash
cd demo

# CPU (default)
docker compose up -d

# GPU
docker compose -f docker-compose.gpu.yml up -d
```

The container auto-downloads the GGUF model from HuggingFace (`itzune/morpheus-gguf`)
at startup — no local model files needed. The default model is `v3_fim.Q5_K_M.gguf`
(the FIM-capable continued-pretraining model; see [itzune/morpheus-fim](https://huggingface.co/itzune/morpheus-fim)).
Open **http://localhost:9090**.

See **[docs/demo.md](../docs/demo.md)** for full documentation (architecture, API,
Docker configuration, llama.cpp version requirements, inference engineering).

## Without Docker

```bash
cd demo
uv sync
uv run python server.py
```

Requires a GGUF model in `../exports/` and `llama-server` on PATH (or set
`LLAMA_SERVER_PATH`). The model defaults to `v3_fim.Q5_K_M.gguf`.

---

## The editor (`/editor.html`) — a reusable thin client

`editor.html` is a **single-file, dependency-free HTML page** (no framework, no
build step) that implements the full ghost-text autocomplete UX: multi-token
continuation rendered as greyed-out ghost text, with VS Code / Copilot-style
keybindings. It is the reference implementation of a *thin client* on top of the
OpenAI-compatible inference server.

**Keybindings**

| Key | Action |
|-----|--------|
| <kbd>Tab</kbd> | Accept the whole suggestion |
| <kbd>Ctrl</kbd>+<kbd>→</kbd> | Accept the next word only |
| <kbd>Alt</kbd>+<kbd>]</kbd> / <kbd>Alt</kbd>+<kbd>[</kbd> | Cycle next / previous alternative |
| <kbd>Esc</kbd> | Dismiss suggestion |

### How it works

- **Two-layer ghost rendering.** A transparent `<textarea>` (the caret + input)
  sits on top of a read-only "ghost layer" `<div>` that renders the real text
  plus the suggestion in grey. This lets the ghost text *push* the text after
  the cursor to the right (true inline-suggestion behaviour) instead of
  overlapping it.
- **Debounced + cancellable requests.** Each keystroke resets a debounce timer
  (`debounce_ms`, default 250 ms); a new request `AbortController.abort()`s any
  in-flight one, so fast typing never wastes GPU/CPU on stale prompts.
- **Cursor position selects the mode.** The editor always sends
  `{prefix: text[:cursor], suffix: text[cursor:]}`. The **server** decides:
  non-empty suffix → **FIM infill** (cursor mid-sentence); empty suffix → **AR
  append** (cursor at end). The client knows nothing about FIM tokens.
- **Confidence gate.** The server returns a `confidence` score (avg logprob).
  The client suppresses ghosts below `confidence_threshold` (default 0.2,
  calibrated to real FIM data) — "clueless mode": rather than show wrong text,
  show nothing.
- **Best-of-n + alternative cycling.** With `n > 1` the server fires `n`
  parallel samples and returns the highest-confidence one. <kbd>Alt</kbd>+<kbd>]</kbd>
  re-requests at an elevated `alt_temperature` to surface a different completion;
  <kbd>Alt</kbd>+<kbd>[</kbd> walks back through history without a request.
- **Live-tunable advanced panel.** A collapsible `<details>` panel exposes every
  `CONFIG` knob (debounce, context windows, sampling, quality gate) and persists
  preferences to `localStorage`.

### The API it speaks — OpenAI-compatible

The editor talks to exactly **one endpoint**:

```http
POST /v1/complete
Content-Type: application/json

{
  "prefix": "Kaixo, zer moduz? Ni atzo ",
  "suffix": " etorri naiz.",
  "max_tokens": 8,
  "temperature": 0.2,
  "top_k": 5,
  "n": 1
}
```

```json
{ "text": "Bilbotik", "confidence": 0.41, "finish_reason": "stop" }
```

That is the entire client contract. The server applies the FIM template
(`<PRE>{prefix}<SUF>{suffix}<MID>`) at the **token level** using the reference
SentencePiece tokenizer, forwards the token IDs to `llama-server`, runs the
always-on output strategies (garbage filter, punctuation collapse, confidence),
and returns `{text, confidence, finish_reason}`. A 50-line client just sends
`{prefix, suffix}` and renders the returned text as a ghost — no FIM-token or
tokenizer knowledge required.

The same server exposes the **full OpenAI completion face**, so any
OpenAI-API-speaking client plugs in with zero model-specific code:

| Endpoint | Shape | Used by |
|----------|-------|---------|
| `POST /v1/completions` | Standard OpenAI `text_completion` (streaming SSE + non-streaming) | Continue.dev, Cody, codecompanion.nvim, the `openai` SDK |
| `POST /v1/complete` | Convenience `{prefix, suffix}` → `{text, confidence, finish_reason}` | Thin bespoke clients: `editor.html`, Obsidian plugins, Vim, CLI |
| `GET /v1/models` | OpenAI model list | Clients that probe on startup |

`/v1/completions` accepts `prompt` as a string, a list of token IDs (the
divergent-tokenizer bypass), or a batch; returns the OpenAI schema plus an extra
`confidence` field (unknown keys are ignored by standard clients, used by thin
clients to suppress low-quality ghosts). See `test_openai_compat.py` for a
conformance smoke test against the real `openai` SDK.

### Reuse the server, swap the client

The architecture is deliberately split:

```
┌──────────────────────────┐        OpenAI-compatible HTTP        ┌──────────────────────┐
│  Thin client (swap me)   │  ─────────────────────────────────▶  │  Reusable backend    │
│  editor.html · Obsidian  │   POST /v1/complete  {prefix,suffix} │  FastAPI proxy       │
│  plugin · Continue.dev   │   POST /v1/completions (OpenAI SDK)  │   + llama-server     │
│  · Vim · CLI · …         │  ◀─────────────────────────────────  │   + GGUF model       │
└──────────────────────────┘       {text, confidence, ...}        └──────────────────────┘
```

`editor.html` exists to prove the client is trivial: ~50 lines of fetch logic,
no tokenizer, no FIM templating, no model-specific code — all of that lives in
the server. Because the wire protocol is the OpenAI completion API, the **same
FastAPI proxy + llama-server + GGUF model** is reused unchanged by:

- **[Continue.dev](https://www.continue.dev/)** — see `continue_config.json`,
  a working `~/.continue/config.json` that wires Morpheus up as the
  `tabAutocompleteModel` with the FIM template.
- **An Obsidian plugin** — a TypeScript plugin posts `{prefix, suffix}` to
  `/v1/complete` on a debounce and renders the response as a ghost decoration,
  exactly like `editor.html` does in the browser. No server changes.
- **Vim / Neovim** (`codecompanion.nvim`, custom `completefunc`), a shell CLI,
  or any other OpenAI-compatible client.

Spin up the Docker container once; point as many clients as you like at
`http://localhost:9090/v1`.
