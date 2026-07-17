# Architecture

> **Goal:** Provide Basque ghost-text autocompletion to any desktop text editor
> (Obsidian, VS Code, Neovim) via a single reusable inference server, where the
> model is a 91M Mamba-2 with FIM (Fill-in-the-Middle) capability.

This document defines **what each component owns** and **why**, so that logic
lands in exactly one place. The governing principle is simple:

> **If it touches tokens, logits, or the SentencePiece tokenizer → it lives in
> the proxy. If it touches the editor UI → it lives in the plugin. The wire
> between them is the OpenAI completions API.**

---

## 1. The three layers

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — Editor Plugin (Obsidian / VS Code / Neovim)      │
│  • cursor tracking, debounce, ghost-text render, keybinds   │
│  • sends {prefix, suffix} → receives {text, confidence}     │
└──────────────────────────┬──────────────────────────────────┘
                           │  OpenAI /v1/completions  (standard)
                           │  or /v1/complete         (convenience)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — Proxy (demo/server.py, FastAPI on :9090)         │
│  • SP tokenization (THE fix for the divergence problem)     │
│  • FIM templating, stop tokens, output cleanup              │
│  • always-on correctness strategies (digit repair, garbage  │
│    filter, confidence computation)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │  token-ID /completion  (internal)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — Engine (llama-server on :8080, GGUF model)       │
│  • raw inference, sampling, logprobs                        │
│  • tokenizer is BYPASSED — receives token IDs, not strings  │
└─────────────────────────────────────────────────────────────┘
```

Each layer has a single responsibility and a clean contract to the layer below.
No layer reaches through the one below it.

---

## 2. Layer 1 — Engine (llama-server)

**Process:** `/root/llama.cpp-fresh/build/bin/llama-server --port 8080`

**Owns:** Inference only. Loads the GGUF model, accepts `/completion` requests
with **token-ID prompts** (not strings), returns generated tokens + logprobs.

**Explicitly does NOT own:** Tokenization. This is the whole reason the proxy
exists. llama.cpp reimplements SentencePiece internally and **diverges** from the
reference tokenizer on Basque's 4K UNIGRAM vocab — the same input string
tokenizes to different IDs:

- Reference SP: `▁zer` → single token (id 515)
- llama.cpp:    `▁zer` → `▁` + `z` + `er` (ids 261, 277, 372)

Measured impact (`COMPARISON.md` §1.2): the **string path scores 0%** top-1
while the **token-ID path scores 43.8%** on the same model. The engine's
tokenizer is unusable; the proxy feeds it pre-tokenized IDs and the engine's
own tokenizer never runs.

**stdin quirk:** llama-server exits on stdin EOF. Launch with
`sleep 100000 | llama-server ...` to keep stdin open.

---

## 3. Layer 2 — Proxy (`demo/server.py`)

**Process:** FastAPI app on `:9090`, started via the demo venv
(`/tmp/demo-venv`, Python 3.14 with fastapi/uvicorn/httpx/sentencepiece).

This is the architectural keystone. It owns three categories of logic, each
detailed below.

### 3.1 Two HTTP faces

| Route | Purpose | Used by |
|-------|---------|---------|
| `POST /v1/completions` | Standard OpenAI completions API (streaming SSE + non-streaming). Accepts `prompt` as string or token-ID list. | Any OpenAI-compatible client: Continue.dev, Cody, codecompanion.nvim, `openai` SDK |
| `POST /v1/complete` | Convenience route for thin bespoke clients. Takes `{prefix, suffix}` raw text; server applies the FIM template. | Obsidian plugin, Vim plugin, CLI — ~50-line clients |
| `GET /v1/models` | OpenAI model list (some clients probe this on startup). | All clients |

The two faces exist because FIM templating is awkward to express in the standard
schema: Continue.dev ships its own FIM templates and uses `/v1/completions`,
while a bespoke Obsidian plugin wants to just send cursor context and let the
server handle templating via `/v1/complete`. Both routes converge on the same
internal `_call_llama()` path.

### 3.2 Tokenization (the fix)

`_normalize_prompt_to_ids(prompt)` accepts any OpenAI prompt form (string,
`list[int]`, or batch `list[str]`) and returns reference SentencePiece token
IDs matching training exactly. This is the single function that makes the model
usable through any client.

For FIM, the FIM tokens (`<PRE>`/`<SUF>`/`<MID>`/`<EOT>`) are atomic
`USER_DEFINED` pieces in `basque_unigram_fim.model`, so `sp.encode()` preserves
them as single tokens (IDs 4000–4003). The proxy builds the FIM string and lets
SP tokenize it as one unit — no manual token splicing.

### 3.3 Always-on correctness strategies

These are **not configurable**. They are bug fixes, not preferences, and making
them toggleable would let a misconfigured client get garbage output. They run
on every request.

| Strategy | Function | What it does |
|----------|----------|--------------|
| SP tokenization | `_normalize_prompt_to_ids` | Reference SentencePiece encode; bypasses llama.cpp divergence |
| FIM templating | `/v1/complete` route | Builds `<PRE>{prefix}<SUF>{suffix}<MID>` when suffix present; falls back to AR when absent |
| Stop tokens | both routes | `<EOT>`, `</s>` (native EOS), `\n\n` (paragraph boundary); passed to llama-server + double-checked client-side |
| Digit token repair | `_generate_with_repair` | Swaps digit-containing output tokens for their best non-digit alternative from top-k logprobs (token-level, not string-level) |
| Byte-fallback garbage filter | `_has_byte_fallback_garbage` | Detects non-Latin output (Cyrillic, U+FFFD) that signals a tokenizer trap; returns empty instead of junk |
| Suggestion cleanup | `filter_suggestion` | Strips ▁ markers, U+FFFD, collapsed punct runs, trailing junk — the proxy decodes tokens→text, so it cleans here |
| Confidence | `compute_confidence` | Average logprob excluding EOS/stop tokens; returned as a `confidence` field so clients can suppress low-quality ghosts |
| Subword-aware context (AR mode) | `smart_context` + `ghost_suffix` | Strips half-finished subword fragments and deduplicates overlap so only the non-typed suffix shows as ghost. **N/A for FIM** — the middle is purely new text |

### 3.4 What the proxy deliberately does NOT do

- **Debounce / timing** — editor concern.
- **Ghost-text rendering** — editor concern.
- **Tab/Esc keybindings** — editor concern.
- **Auto-commit** — explicitly rejected (see §6).

---

## 4. Layer 3 — Editor Plugin

A plugin is **thin**: it owns editor glue and nothing else. Its entire loop:

```
on cursor idle (debounce_ms):
    prefix = textBeforeCursor (last prefix_chars)
    suffix = textAfterCursor (next suffix_chars)
    if len(prefix) < min_prefix_chars: return

    abort previous request
    resp = POST /v1/complete  { prefix, suffix, max_tokens, temperature, top_k }
    if resp.confidence < confidence_threshold: return

    show resp.text as grey ghost-text at cursor

on Tab:  insert ghost-text
on Esc:  dismiss ghost-text
on any keystroke: abort in-flight request, restart debounce
```

### 4.1 Plugin-owned config (Layer 3 — purely local)

These cannot be server-side because they are about the editor, not the model:

| Config | Default | Why plugin-only |
|--------|---------|-----------------|
| `server_url` | `http://localhost:9090` | Where the proxy lives |
| `debounce_ms` | 250 | Editor timing |
| `confidence_threshold` | 0.6 | Client decision using the proxy's `confidence` field |
| `min_prefix_chars` | 10 | Don't request until enough context |
| `prefix_chars` | 2048 | How much context to send (client-side truncation) |
| `suffix_chars` | 1024 | How much suffix to send |
| Ghost-text styling | grey | CodeMirror 6 / Obsidian editor API |
| Keybindings | Tab/Esc | Editor keymap |

### 4.2 API params the plugin controls (Layer 1 — forwarded by proxy)

These are the standard OpenAI request-body fields. The plugin sets them
per-request; the proxy forwards them to llama-server:

| Param | Default (ghost-text) | Notes |
|-------|----------------------|-------|
| `temperature` | 0.2 | Low-but-nonzero: recovers rank-2/rank-3 tokens without wildness (P7) |
| `top_k` | 5 | Pure greedy is too brittle; the rank-2 evidence justifies a small top-k |
| `max_tokens` | 32 | Enough for a phrase, not a paragraph |
| `stop` | `["<EOT>", "\n\n"]` (FIM) | Server sets sane defaults; client can override |

---

## 5. Configuration: the three-layer rule

Confusion about "where should this config live?" is resolved by asking which of
three categories it falls into:

### Layer 1 — API params (client-controlled, server forwards)
The *official* OpenAI knobs: `temperature`, `top_k`, `max_tokens`, `stop`,
`stream`. The plugin sets these per-request; the proxy passes them through to
llama-server. This works because the OpenAI spec built them in.

### Layer 2 — Always-on strategies (server-side, NOT configurable)
Our custom strategies are **correctness fixes, not preferences**: digit repair,
garbage filter, confidence computation, FIM templating. They run on every
request and are not toggleable. A misconfigured client must not be able to get
Cyrillic garbage or digit-corrupted output.

> If a debug mode is ever needed, custom fields go on `/v1/complete` only
> (it is already bespoke with `{prefix, suffix}` — adding `repair_digits: false`
> there breaks nothing). The standard `/v1/completions` route stays pure OpenAI
> so any client works unmodified.

### Layer 3 — Editor behavior (purely plugin, server can't help)
Debounce, confidence threshold, ghost styling, keybindings, context windows.
These cannot be server-side because they are about the editor, not the model.

---

## 6. What we explicitly reject (and why)

Filtered out of the roadmap because they serve mobile-keyboard UX and are
**harmful** for desktop ghost-text. See `TODO.md` → "Explicitly not adopting".

| Strategy | Origin | Why rejected |
|----------|--------|--------------|
| Logit banning of punctuation/symbols | FUTO mobile (word-chip boundaries) | Desktop completions legitimately need periods, commas, newlines, quotes. Banning them breaks prose. |
| Capitalization-aware logit banning | FUTO mobile (typing modes) | Desktop editors handle capitalization via context. |
| Confidence-ratio auto-commit | FUTO mobile (auto-replace typed word) | Desktop uses explicit Tab-accept, not auto-replace. The *other* half — suppress low-confidence — IS useful and is covered by the plugin's `confidence_threshold` (Continue.dev already implements this). |
| Chat-completions API for autocomplete | — | Wrong shape for FIM. FIM needs raw prompt + streaming + stop tokens, not a message-array/role abstraction. |
| Bespoke per-editor plugin (reimplementing FIM) | — | Continue.dev already covers FIM templating + prefiltering + VS Code/JetBrains. Reusing it is strictly better. One server, N thin clients. |

---

## 7. Data flow — FIM completion (the primary path)

```
EDITOR                        PROXY (server.py)                    ENGINE (llama-server)
  │
  │ cursor idle, prefix="Gaur goizean " suffix="etorriko da."
  │ POST /v1/complete {prefix, suffix, max_tokens:32, temp:0.2}
  ├──────────────────────────►│
  │                            │ build FIM string:
  │                            │   <PRE>Gaur goizean <SUF>etorriko da.<MID>
  │                            │ sp.encode(fim_string) → [4000, ..., 4001, ..., 4002]
  │                            │ stops = ["<EOT>", "\n\n"]
  │                            │ POST /completion {prompt: token_ids, stop: stops, ...}
  │                            ├─────────────────────────────────────────────►│
  │                            │                                              │ generates "ni "
  │                            │                                              │ then <EOT> (id 4003)
  │                            │                                              │ → stopped_eos/word
  │                            │◄─────────────────────────────────────────────┤
  │                            │ text="ni ", strip stop, compute confidence
  │                            │ check garbage filter, filter_suggestion
  │  {text:"ni ", confidence:0.91, finish_reason:"stop"}
  │◄──────────────────────────┤
  │ 0.91 > 0.6 threshold → render "ni " as grey ghost
  │ Tab → insert
```

The editor never sees a token, a FIM token, or a logprob. It sends raw cursor
text and receives clean text + a confidence number.

---

## 8. Data flow — AR-only completion (fallback / legacy)

When no suffix is provided, `/v1/complete` falls back to prefix-only
autoregressive completion. This is the path the pre-FIM (74K) checkpoint uses
and the path Continue.dev hits via `/v1/completions` with its own FIM template
disabled.

```
EDITOR                        PROXY                                  ENGINE
  │ POST /v1/complete {prefix:"Kaixo, zer ", max_tokens:16}        
  ├──────────────────────────►│
  │                            │ sp.encode(prefix) → token IDs
  │                            │ stops = ["\n\n"]
  │                            │ smart_context: strip half-subword if needed
  │                            │ POST /completion {prompt: ids, stop: ["\n\n"]}
  │                            ├──────────────────────────────────────────────►│
  │                            │◄──────────────────────────────────────────────┤ "moduz?"
  │                            │ ghost_suffix: dedup overlap with typed text
  │                            │ filter_suggestion, confidence
  │  {text:"moduz?", confidence:0.78}
  │◄──────────────────────────┤
```

---

## 9. Why the proxy is the architecture (not a fallback)

The original question was: "test if FIM masks the tokenizer divergence; if not,
keep the proxy." The evidence made it unconditional — the divergence is
**model-agnostic** (it is in the tokenizer, not the weights), so no amount of
FIM training fixes it:

- Same AR model, same prompts: string path 0% vs token-ID path 43.8%.
- The fix is always "use reference SP tokenization", which only the proxy can do
  (a JS SentencePiece port would re-diverge — the Viterbi algorithm differs).

So the proxy is **the** architecture. Every client is thin because the proxy
centralizes the one hard part. If llama.cpp ever fixes the divergence upstream,
delete the proxy and repoint clients at llama-server directly — zero client
changes, because the wire protocol (OpenAI completions) stays the same.

---

## 10. Component inventory

| Component | Path | Role |
|-----------|------|------|
| Proxy | `demo/server.py` | FastAPI app; SP encode, FIM template, cleanup, two HTTP faces |
| OpenAI conformance test | `demo/test_openai_compat.py` | 9+2 checks against real `openai` SDK (the same SDK Continue.dev uses) |
| Continue.dev config | `demo/continue_config.json` | Points Continue at the proxy (`provider: openai`, `roles: [autocomplete]`) |
| Engine | llama-server binary | Raw inference; tokenizer bypassed |
| Model (AR) | `exports/step_0074000.Q5_K_M.gguf` | Pre-FIM, 4K vocab |
| Model (FIM) | `exports/morpheus-fim.Q5_K_M.gguf` | Post-Phase-6, 4016 vocab with FIM tokens |
| Tokenizer (AR) | `tokenizer/basque_unigram_4000.model` | 4000 pieces |
| Tokenizer (FIM) | `tokenizer/basque_unigram_fim.model` | 4004 pieces (4 FIM USER_DEFINED) |
| Training | `train.py` + `config/phase6_fim.yaml` | Continued pretraining with FIM |
| FIM dataset build | `scripts/pipeline/build_fim.py` | Char-level PSM/SPM transform of corpus |
| FIM eval | `scripts/fim_eval.py` | Exact-match, char accuracy, keystrokes-saved |
| Trajectory | `docs/TRAJECTORY.md` | Desktop-editor direction + Phase 6 roadmap |
| Runbook | `docs/PHASE6_RUNBOOK.md` | Step-by-step train/eval/export/deploy commands |

---

## 11. Launch sequence

```bash
# 1. Engine (keeps stdin open)
screen -dmS llama bash -c "
sleep 100000 | /root/llama.cpp-fresh/build/bin/llama-server \
    --model exports/morpheus-fim.Q5_K_M.gguf \
    --port 8080 --ctx-size 2048 --temp 0.2 --top-k 5
"

# 2. Proxy
screen -dmS demo bash -c "
cd /root/morpheus-mamba/demo && \
exec /tmp/demo-venv/bin/python -u -m uvicorn server:app \
    --host 0.0.0.0 --port 9090
"

# 3. Verify
/tmp/demo-venv/bin/python demo/test_openai_compat.py

# 4. Point an editor at http://localhost:9090/v1
#    - Continue.dev: use demo/continue_config.json
#    - Obsidian: the plugin sends to /v1/complete
```

---

## 12. Future extensions (where new logic goes)

| Future idea | Layer | Why |
|-------------|-------|-----|
| Request cancellation (abort in-flight) | Proxy + plugin | Plugin triggers abort; proxy cancels the llama-server request. Simplifies client debounce. |
| Context windowing (truncate prefix to ctx) | Proxy | Proxy knows the model's ctx limit; client can send raw editor contents. |
| `min_prefix_chars` / `max_prefix_chars` enforcement | Proxy | Guard against trivial requests; client still gates locally too. |
| LoRA personalization (P8) | Engine + new merge step | LoRA on in_proj/out_proj, merged + requantized to GGUF. Transparent to proxy and plugin. |
| Verb-agreement SFT (P9) | Training only | Composes with FIM; no architecture change. |
| LLM-judge FIM eval (P5 extension) | Eval scripts | Sampling-based quality bar; orthogonal to serving. |
