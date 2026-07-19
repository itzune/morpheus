"""
Morpheus v2 Mamba Demo Server — proxies to llama-server's completion API.

Usage:
    # Start llama-server first (or let this script launch it):
    uv run python demo/server.py
    uv run python demo/server.py --port 9090 --llama-port 8080

Dependencies: fastapi, uvicorn, httpx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import Union, List, Optional
from pydantic import BaseModel, Field

# ── Model backend (pluggable: tokenizer + FIM template + output cleanup) ──
# The three architecture-specific concerns live in backends.py, so the
# OpenAI-compatible routes and all clients are model-agnostic. Swapping to a
# transformer or fine-tuned Llama is `MORPHEUS_BACKEND=llama-fim` — no route
# or client changes. See backends.py and demo/README.md.
from backends import (
    get_backend,
    compute_confidence,
    filter_suggestion,
    has_byte_fallback_garbage as _has_byte_fallback_garbage,
    is_pure_punct as _is_pure_punct,
)

backend = get_backend()


def _prompt_to_ids(prompt: str):
    """Encode the prompt for llama-server (backend-specific).

    The SentencePiece backend returns token IDs (bypassing llama.cpp's
    divergent tokenizer for fidelity to training); a Llama/transformer
    backend returns the string and lets llama.cpp tokenize. Either way
    _call_llama sends the result as llama-server's `prompt` field.
    """
    return backend.encode(prompt)


def smart_context(text: str) -> str:
    """Return the token-aligned context for autocomplete.

    Strips only trailing non-▁ tokens (true subword continuations like
    "d", "ko", "t"). Keeps the last ▁-prefixed token regardless.

    The model gets the full text the user typed, minus any half-finished
    subword fragment. If the user typed a complete token (even if it's a
    short word fragment), the model sees it and predicts accordingly.
    Ghost deduplication handles the overlap in the UI.

    Examples:
      "Kaixo, zer mod" → tokens [▁Kaixo, ,, ▁zer, ▁mo, d]
        → strip d (non-▁) → "Kaixo, zer mo"
      "Kaixo, zer moduz" → tokens [▁Kaixo, ,, ▁zer, ▁moduz]
        → all ▁-prefixed → keep → "Kaixo, zer moduz"
      "Gaur nire ama berandu irit" → tokens [▁Gaur, ▁nire, ▁ama, ▁berandu, ▁iri, t]
        → strip t (non-▁) → "Gaur nire ama berandu iri"
      "Gaur nire ama berandu iri" → tokens [▁Gaur, ▁nire, ▁ama, ▁berandu, ▁iri]
        → all ▁-prefixed → keep → "Gaur nire ama berandu iri"
    """
    if not text:
        return text
    return text  # Pass full text; ghost_suffix computes the token-level overlap


def ghost_suffix(text: str, smart_ctx: str, suggestion: str) -> str:
    """Compute the ghost suffix that overlaps the user's excluded text.

    Smart Compose behavior: if user typed part of the predicted completion,
    only show the non-typed suffix as ghost.

    Example:
      text="Kaixo, zer mod", smart_ctx="Kaixo, zer", suggestion=" moduz?"
      User typed "mod", model predicted " moduz?"
      Overlap: "mod" (excluded) starts match with " mod" (prediction prefix)
      → ghost = "uz?" (the novel suffix)
    Example:
      text="Euskal He", smart_ctx="Euskal", suggestion=" Herria"
      User typed "He", model predicted " Herria" (different token)
      → no overlap, ghost = full suggestion " Herria"
    """
    excluded = text[len(smart_ctx):] if len(text) > len(smart_ctx) else ""

    PUNCT = '.!,?;:()[]{}'

    # If user's text ends with punct AND model starts with punct,
    # strip the leading punct from suggestion to avoid doubling.
    # E.g.: text="Kaixo?", suggestion="? zer" → ghost=" zer"
    # Only strip if it's the SAME punct char (user="Kaixo." suggestion="?" → keep).
    if text and suggestion and text[-1] in PUNCT and suggestion[0] == text[-1]:
        suggestion = suggestion[1:]

    if not excluded or not suggestion:
        return suggestion

    # Find longest prefix of suggestion matching a suffix of excluded
    for i in range(min(len(excluded), len(suggestion)), 0, -1):
        if suggestion[:i].lower() == excluded[-i:].lower():
            # Overlap found — only show non-overlapping suffix
            return suggestion[i:]
    return suggestion


app = FastAPI(title="Morpheus v2 Mamba Demo")

LLAMA_SERVER_URL = "http://localhost:8080"
llama_process: subprocess.Popen | None = None

# Current model path (for health/metadata)
_current_model_path: str = ""


EXPORTS_DIR = Path(os.environ.get("MORPHEUS_MODELS_DIR", Path(__file__).resolve().parent.parent / "exports"))

# Also check the Docker volume mount point for HF-downloaded models
MODELS_DIR = Path(os.environ.get("MORPHEUS_MODELS_DIR", "/app/models"))

# Default model — change this when a new checkpoint is promoted
DEFAULT_MODEL = "v3_fim.Q5_K_M.gguf"

# ── Completion logging ──────────────────────────────────
# Logs user acceptances (and sends) to a JSONL file so we can build a
# real eval dataset from actual usage and replay it against any checkpoint.
LOG_DIR = Path(os.environ.get("LOG_DIR", Path(__file__).resolve().parent.parent / "logs"))
LOG_FILE = Path(os.environ.get("LOG_FILE", LOG_DIR / "completions.jsonl"))


def _model_name() -> str:
    """Current model filename (e.g. step_0032000.Q4_K_M.gguf)."""
    return Path(_current_model_path).name if _current_model_path else "unknown"


def _log_event(event: dict) -> None:
    """Append a JSON event to the completion log (fire-and-forget, best-effort)."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                       + f".{int((time.time() % 1) * 1000):03d}Z",
            "model": _model_name(),
            **event,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[log] failed to write event: {e}")


def _find_llama_binary() -> str:
    """Find llama-server binary in common locations."""
    candidates = [
        os.environ.get("LLAMA_SERVER_BINARY", ""),
        "/opt/llama.cpp/build/bin/llama-server",   # Docker
        "/root/llama.cpp-fresh/build/bin/llama-server",  # GPU server
        "/tmp/llama.cpp/build/bin/llama-server",
        "llama-server",  # on PATH
    ]
    for c in candidates:
        if not c:
            continue
        if "/" in c and Path(c).exists():
            return c
        # Check PATH
        import shutil
        found = shutil.which(c)
        if found:
            return found
    raise FileNotFoundError("llama-server not found. Build llama.cpp first.")


def _resolve_model(model_arg: str | None) -> str:
    """Resolve model path from: CLI arg > MORPHEUS_MODEL env > default.

    Checks both the exports directory and the Docker models volume (/app/models).
    """
    search_dirs = [EXPORTS_DIR, MODELS_DIR, Path("/app/exports"), Path("/app/models")]
    if model_arg:
        return model_arg
    env_model = os.environ.get("MORPHEUS_MODEL", "")
    if env_model:
        p = Path(env_model)
        if p.exists():
            return str(p.resolve())
        # Try relative to each search dir
        for d in search_dirs:
            p = d / env_model
            if p.exists():
                return str(p.resolve())
        print(f"Warning: MORPHEUS_MODEL={env_model} not found, using default")
    # Default: explicit model, fallback to latest Q5_K_M, then any GGUF
    for d in search_dirs:
        p = d / DEFAULT_MODEL
        if p.exists():
            return str(p.resolve())
    for d in search_dirs:
        ggufs = sorted(d.glob("*.Q5_K_M.gguf"))
        if ggufs:
            return str(ggufs[-1].resolve())
    for d in search_dirs:
        ggufs = sorted(d.glob("*.gguf"))
        if ggufs:
            return str(ggufs[-1].resolve())
    raise FileNotFoundError(f"No GGUF models found in {search_dirs}")


def _discover_models() -> list[dict]:
    """List available GGUF models in exports and models directories."""
    models = []
    seen = set()
    search_dirs = [EXPORTS_DIR, MODELS_DIR, Path("/app/exports"), Path("/app/models")]
    for d in search_dirs:
        for p in sorted(d.glob("*.gguf")):
            if p.name not in seen:
                seen.add(p.name)
                models.append({
                    "name": p.name,
                    "path": str(p.resolve()),
                    "size_mb": round(p.stat().st_size / 1e6, 1),
                })
    return models


def _kill_process_on_port(port: int) -> None:
    """Kill whatever process is listening on the given port.

    Used during model hot-reload: when llama-server was started externally
    (e.g. by Docker CMD, so llama_process is None), we need to find and kill
    it by port so the new model can bind.
    """
    try:
        # fuser is available on Ubuntu/Debian (our Docker base image)
        result = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        for pid_str in pids:
            try:
                pid = int(pid_str)
                if pid > 1:  # never kill PID 1 (the container init)
                    os.kill(pid, signal.SIGTERM)
                    print(f"Killed PID {pid} on port {port}")
            except (ValueError, ProcessLookupError):
                pass
    except FileNotFoundError:
        # fuser not available — try lsof as fallback
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().split():
                try:
                    pid = int(pid_str)
                    if pid > 1:
                        os.kill(pid, signal.SIGTERM)
                        print(f"Killed PID {pid} on port {port}")
                except (ValueError, ProcessLookupError):
                    pass
        except (FileNotFoundError, Exception):
            pass
    except Exception as e:
        print(f"Warning: could not kill process on port {port}: {e}")


def start_llama_server(model_path: str, port: int) -> subprocess.Popen | None:
    """Start llama-server as a subprocess.

    If an existing llama-server is running (either started by us or
    externally, e.g. by Docker CMD), kill it first so the new model loads.
    """
    global LLAMA_SERVER_URL, _current_model_path, llama_process
    LLAMA_SERVER_URL = f"http://localhost:{port}"

    # Kill any existing llama-server on this port — whether we started it
    # (llama_process) or it was started externally (Docker CMD).
    if llama_process is not None:
        stop_llama_server()
    _kill_process_on_port(port)

    # Brief pause to let the OS release the port
    time.sleep(1)

    binary = _find_llama_binary()

    ngl = os.environ.get("MORPHEUS_NGL", "0")

    llama_cmd = [
        binary,
        "-m", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "-ngl", str(ngl),
    ]

    print(f"Starting llama-server: {' '.join(llama_cmd)}")
    proc = subprocess.Popen(
        llama_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )

    _current_model_path = model_path

    # Wait for it to be ready
    for _ in range(30):
        time.sleep(0.5)
        try:
            r = httpx.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"llama-server is ready (model: {Path(model_path).name})")
                return proc
        except Exception:
            pass

    print("Warning: llama-server didn't respond — continuing anyway")
    return proc


def stop_llama_server():
    global llama_process
    if llama_process:
        print("Stopping llama-server...")
        try:
            os.killpg(os.getpgid(llama_process.pid), signal.SIGTERM)
            llama_process.wait(timeout=5)
        except Exception:
            llama_process.kill()
        llama_process = None


# Top-k logprobs per token — powers both candidate extraction (keyboard view)
# and digit-token repair (swap digit tokens for best non-digit alternative).
N_PROBS = 5


# ── Shared HTTP client (connection pooling / keep-alive) ──────────────
# Creating an httpx.AsyncClient per request costs ~60ms of TCP setup/teardown
# per call (measured against localhost llama-server). This singleton keeps the
# connection warm so repeated requests reuse it via HTTP keep-alive. Per-request
# timeouts override the default via the `timeout=` kwarg on .post()/.get().
_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    """Return the process-wide shared httpx client (lazy singleton)."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


async def _call_llama(prompt_ids: list[int], max_tokens: int, temperature: float, greedy: bool) -> dict:
    """Call llama-server /completion with token-ID prompt and top-k logprobs.

    This is the core LLM call. Callers use _generate_with_repair() instead,
    which wraps this with digit repair and candidate extraction.
    """
    payload: dict = {
        "prompt": prompt_ids,
        "n_predict": max_tokens,
        "n_probs": N_PROBS,
        "stream": False,
    }
    if greedy:
        payload.update({"temperature": 0.0, "top_p": 1.0, "top_k": 0, "repeat_penalty": 1.1})
    else:
        payload["temperature"] = temperature
    r = await _client().post(f"{LLAMA_SERVER_URL}/completion", json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()


# ── Token-level digit repair & candidate extraction ──────

import math as _math


def _decode_token(tok: dict) -> str:
    """Decode a token from completion_probabilities (handles byte-fallback).

    llama-server returns {'bytes': [98, 121, ...], 'id': N, 'token': '...', 'logprob': X}
    The 'bytes' field is the authoritative UTF-8 representation; 'token' is a
    debug string that may be wrong for byte-fallback tokens.
    """
    raw_bytes = tok.get("bytes", [])
    if raw_bytes:
        return bytes(raw_bytes).decode("utf-8", errors="replace")
    return tok.get("token", "")


def _token_has_digit(text: str) -> bool:
    """Check if a token's text contains any digit character."""
    return any(c.isdigit() for c in text)


def _extract_candidates(result: dict, top_k: int = 3, filter_digits: bool = True) -> list[dict]:
    """Extract next-token candidates from completion_probabilities[0].

    Returns [{text, prob}, ...] for the keyboard suggestion view.
    Digit-containing tokens are skipped when filter_digits is True.
    """
    probs = result.get("completion_probabilities", [])
    if not probs:
        return []
    candidates = []
    for tok in probs[0].get("top_logprobs", []):
        text = _decode_token(tok)
        if not text.strip():
            continue
        if filter_digits and _token_has_digit(text):
            continue
        candidates.append({"text": text, "prob": round(_math.exp(tok["logprob"]), 4)})
        if len(candidates) >= top_k:
            break
    return candidates


async def _generate_with_repair(
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    greedy: bool,
    filter_digits: bool,
) -> tuple[str, float, list[dict], float]:
    """Generate completion with digit-token repair and candidate extraction.

    Returns (suggestion, confidence, candidates, latency_ms).

    Digit repair strategy (replaces post-hoc string filtering):
      1. Generate max_tokens with top-k logprobs per position.
      2. Walk the greedy path. At each position, if the chosen token contains
         digits, swap it for the best non-digit alternative from top_logprobs.
      3. If no swap was needed → return original content (zero extra cost,
         the common case).
      4. If a swap changed the path → re-generate from the swap point (one
         extra request) so subsequent tokens are correctly conditioned.
      5. If a digit token has NO non-digit alternative: if it's the first
         token, return empty; otherwise truncate at the good prefix.
    """
    t0 = time.perf_counter()
    result = await _call_llama(prompt_ids, max_tokens, temperature, greedy)

    candidates = _extract_candidates(result, filter_digits=filter_digits)
    confidence = compute_confidence(result)
    content = result.get("content", "")

    # BPE backends (Llama/transformer) pass string prompts and don't need
    # digit repair — it's a SentencePiece byte-fallback workaround. A
    # converged 8B BPE model produces clean Latin text; any digits in its
    # output are legitimate (e.g. "3 milioi"), not byte-fallback garbage.
    if not filter_digits or isinstance(prompt_ids, str):
        latency = (time.perf_counter() - t0) * 1000
        return content, confidence, candidates, latency

    # Walk the greedy path, looking for digit tokens to repair
    probs = result.get("completion_probabilities", [])
    if not probs:
        latency = (time.perf_counter() - t0) * 1000
        return content, confidence, candidates, latency

    repaired_ids: list[int] = []
    first_swap = -1

    for i, pos in enumerate(probs):
        chosen_id = pos["id"]
        chosen_text = _decode_token(pos)

        # Repair needed for: digit tokens (any position) or EOS/empty at
        # position 0 (model predicts end-of-sentence immediately, leaving
        # content empty — swap for best non-empty alternative so the user
        # sees a suggestion).  At later positions, EOS is a natural stop
        # (content already excludes it), so we don't swap there.
        needs_repair = _token_has_digit(chosen_text) or (i == 0 and not chosen_text.strip())

        if needs_repair:
            # Try to find a usable alternative at this position
            found_alt = False
            for alt in pos.get("top_logprobs", []):
                alt_text = _decode_token(alt)
                if not _token_has_digit(alt_text) and alt_text.strip() and alt["id"] != chosen_id:
                    repaired_ids.append(alt["id"])
                    found_alt = True
                    if first_swap == -1:
                        first_swap = i
                    break
            if not found_alt:
                # Can't repair this position
                if i == 0:
                    # First token unrepairable → no usable suggestion
                    latency = (time.perf_counter() - t0) * 1000
                    return "", confidence, candidates, latency
                # Later token unrepairable → truncate at good prefix
                latency = (time.perf_counter() - t0) * 1000
                return backend.decode(repaired_ids), confidence, candidates, latency
        else:
            repaired_ids.append(chosen_id)

    if first_swap == -1:
        # No digit tokens found → return original content (zero extra cost)
        latency = (time.perf_counter() - t0) * 1000
        return content, confidence, candidates, latency

    # Re-generate from the first swap point (one extra request)
    new_prompt = prompt_ids + repaired_ids[: first_swap + 1]
    remaining = max_tokens - (first_swap + 1)
    prefix_text = backend.decode(repaired_ids[: first_swap + 1])

    if remaining <= 0:
        latency = (time.perf_counter() - t0) * 1000
        return prefix_text, confidence, candidates, latency

    regen = await _call_llama(new_prompt, remaining, temperature, greedy)
    regen_content = regen.get("content", "")
    latency = (time.perf_counter() - t0) * 1000
    return prefix_text + regen_content, confidence, candidates, latency


PUNCT_CHARS = '.!,?;:()[]{}'

# ── Keyboard-mode retokenization fallback ─────────────────
# When the user types a partial word character-by-character, the tokenizer
# may segment it in a way that's incompatible with the full word's
# segmentation. Example: "Kaix" → [▁Ka, ix] but "Kaixo" → [▁Ka, i, xo].
# Once on the [▁Ka, ix] path, the model cannot predict "o" to form "Kaixo"
# because it never saw that token sequence in training.
#
# Fix: try progressively shorter prefixes ("Kai", "Ka") which may land on
# a compatible tokenization path. Generate candidates from each, decode to
# surface text, and keep only those whose full word starts with what the
# user actually typed.


def _extract_current_word(text: str) -> tuple[str, str]:
    """Split text into (text_before_word, current_word) at the cursor.

    The cursor is assumed to be at the end of `text` (standard for a keyboard
    input bar). If text ends with whitespace, current_word is empty (the user
    is between words — next-word prediction, not word completion).
    """
    if not text or text[-1].isspace():
        return text, ""
    i = len(text) - 1
    while i >= 0 and not text[i].isspace():
        i -= 1
    return text[:i + 1], text[i + 1:]


def _extract_first_word(content: str) -> str:
    """Extract the first word from generated content.

    Returns empty string if content starts with whitespace (the model thinks
    the prefix is already a complete word and is starting a new one — not
    useful for word completion).
    Strips trailing punctuation.
    """
    if not content or content[0].isspace():
        return ""
    word = ""
    for c in content:
        if c.isspace():
            break
        word += c
    return word.rstrip(PUNCT_CHARS)


async def _keyboard_candidates(
    text: str,
    max_tokens: int = 5,
    top_k: int = 3,
    filter_digits: bool = True,
) -> tuple[list[dict], float]:
    """Generate word-completion candidates using retokenization fallback.

    Returns (candidates, latency_ms) where candidates is a list of
    {text, prob} dicts. The `text` field is the FULL completed word
    (e.g. "Kaixo"), not a raw token suffix.

    Strategy:
      - If the cursor is mid-word (word completion mode):
        Try N fallback prefix lengths (0, 1, 2 chars shorter). For each:
          1. Encode the shorter prefix to tokens.
          2. Generate up to max_tokens with top-k logprobs.
          3. From the greedy content: extract the first word, prepend the
             shorter prefix string, check it starts with the typed word.
          4. From top-k alternatives at position 0: decode each token, prepend
             the shorter prefix string, check it starts with the typed word.
        Merge + dedup by word, keep highest prob, sort, return top_k.

      - If the cursor is after a space (next-word prediction mode):
        Generate from the full text, return top-k first-token words plus the
        greedy full word.
    """
    t0 = time.perf_counter()

    text_before_word, current_word = _extract_current_word(text)

    # ── Next-word prediction (cursor after space) ──
    if not current_word:
        prompt_ids = backend.encode(text)
        result = await _call_llama(prompt_ids, max_tokens, 0.0, greedy=True)
        probs = result.get("completion_probabilities", [])
        content = result.get("content", "")
        candidates: list[dict] = []
        seen: set[str] = set()

        # Greedy full word first
        if content:
            stripped = content.lstrip()
            first_word = _extract_first_word(stripped) if stripped else ""
            # _extract_first_word returns "" if starts with space, but we
            # already lstripped, so re-extract properly
            if stripped:
                fw = ""
                for c in stripped:
                    if c.isspace():
                        break
                    fw += c
                first_word = fw.rstrip(PUNCT_CHARS)
            if first_word and not (filter_digits and _token_has_digit(first_word)):
                prob = round(_math.exp(probs[0]["logprob"]), 4) if probs else 0.5
                candidates.append({"text": first_word, "prob": prob})
                seen.add(first_word)

        # Top-k first-token alternatives
        if probs:
            for tok in probs[0].get("top_logprobs", []):
                tok_text = _decode_token(tok)
                if not tok_text.strip():
                    continue
                if filter_digits and _token_has_digit(tok_text):
                    continue
                word = tok_text.strip()
                if word and word not in seen:
                    candidates.append({"text": word, "prob": round(_math.exp(tok["logprob"]), 4)})
                    seen.add(word)
                if len(candidates) >= top_k:
                    break
        latency = (time.perf_counter() - t0) * 1000
        candidates = [c for c in candidates if not _has_byte_fallback_garbage(c.get("text", ""))]
        return candidates[:top_k], latency

    # ── Word completion (cursor mid-word) ──
    candidates_map: dict[str, dict] = {}  # word -> {text, prob}

    # Build fallback paths: progressively shorter prefixes.
    # The last path (shorter_word="") predicts the NEXT WORD from scratch
    # (from text_before_word only), then filters by current_word prefix.
    # This rescues single-token words like "bezala" where the user types "b"
    # but the model can't reach ▁bezala from the [▁b] token path.
    max_fallback = min(2, len(current_word) - 1)
    fallback_paths: list[tuple[str, list[int], bool]] = []  # (shorter_word, ids, is_from_scratch)
    for fallback in range(max_fallback + 1):
        shorter_len = len(current_word) - fallback
        if shorter_len < 1:
            break
        shorter_word = current_word[:shorter_len]
        prefix = text_before_word + shorter_word
        fallback_paths.append((shorter_word, backend.encode(prefix), False))
    # Always add the from-scratch path if there's preceding context
    if text_before_word.strip():
        fallback_paths.append(("", backend.encode(text_before_word), True))

    # Fire all calls in parallel to keep latency ~1x instead of Nx
    results = await asyncio.gather(
        *[_call_llama(ids, max_tokens, 0.0, greedy=True) for _, ids, _ in fallback_paths],
        return_exceptions=True,
    )

    for (shorter_word, _, is_from_scratch), result in zip(fallback_paths, results):
        if isinstance(result, Exception):
            continue
        probs = result.get("completion_probabilities", [])
        content = result.get("content", "")

        # 1. Greedy multi-token word completion
        if content:
            # At from-scratch level, content starts with ▁ (space) — lstrip it
            raw = content.lstrip() if is_from_scratch else content
            word_completion = _extract_first_word(raw)
            if word_completion:
                full_word = shorter_word + word_completion
                # Allow >= so that when the user has typed a complete word
                # (e.g. "zer"), the word itself appears as a candidate.
                # Tapping it adds a space, then next-word predictions appear.
                if (full_word.startswith(current_word)
                        and len(full_word) >= len(current_word)
                        and not (filter_digits and _token_has_digit(full_word))):
                    prob = round(_math.exp(probs[0]["logprob"]), 4) if probs else 0.5
                    if full_word not in candidates_map or prob > candidates_map[full_word]["prob"]:
                        candidates_map[full_word] = {"text": full_word, "prob": prob}

            # 1b. If content starts with a space (at non-from-scratch levels),
            # the model thinks the current word is COMPLETE and is predicting
            # the NEXT word. Include it as a next-word candidate.
            if not is_from_scratch and content and content[0].isspace():
                next_word = _extract_first_word(content.lstrip())
                if next_word and not (filter_digits and _token_has_digit(next_word)):
                    prob = round(_math.exp(probs[0]["logprob"]), 4) if probs else 0.5
                    key = "__next__" + next_word  # separate namespace from completions
                    if key not in candidates_map or prob > candidates_map[key]["prob"]:
                        candidates_map[key] = {"text": next_word, "prob": prob, "is_next_word": True}

        # 2. Top-k single-token alternatives at position 0
        if probs:
            greedy_first_id = probs[0].get("id")
            for tok in probs[0].get("top_logprobs", []):
                # Skip the greedy first token — it's already represented by
                # the greedy multi-token word above. Including it again as
                # a single-token candidate produces subword noise (e.g. "bez"
                # when the greedy word is "bezela").
                if tok.get("id") == greedy_first_id:
                    continue
                tok_text = _decode_token(tok)
                if not tok_text.strip():
                    continue
                if filter_digits and _token_has_digit(tok_text):
                    continue
                # At from-scratch level, tokens are ▁-prefixed (start with space).
                # Strip the leading space to get the word.
                if is_from_scratch:
                    tok_text = tok_text.lstrip()
                    if not tok_text:
                        continue
                else:
                    # At continuation levels, a ▁-prefixed token (starts with
                    # space) means the model thinks the current word is complete
                    # and this is the start of a NEW word. Include it as a
                    # next-word candidate instead of skipping it.
                    if tok_text[0].isspace():
                        next_word = tok_text.strip()
                        if next_word and not (filter_digits and _token_has_digit(next_word)):
                            prob = round(_math.exp(tok["logprob"]), 4)
                            key = "__next__" + next_word
                            if key not in candidates_map or prob > candidates_map[key]["prob"]:
                                candidates_map[key] = {"text": next_word, "prob": prob, "is_next_word": True}
                        continue
                full_word = shorter_word + tok_text
                if (full_word.startswith(current_word)
                        and len(full_word) >= len(current_word)):
                    prob = round(_math.exp(tok["logprob"]), 4)
                    if full_word not in candidates_map or prob > candidates_map[full_word]["prob"]:
                        candidates_map[full_word] = {"text": full_word, "prob": prob}

    sorted_cands = sorted(candidates_map.values(), key=lambda x: x["prob"], reverse=True)
    # Filter byte-fallback garbage: shorter-prefix paths can still produce
    # non-Latin chars (Cyrillic, U+FFFD) when the tokenization is incompatible.
    sorted_cands = [c for c in sorted_cands if not _has_byte_fallback_garbage(c.get("text", ""))]
    latency = (time.perf_counter() - t0) * 1000
    return sorted_cands[:top_k], latency


# Output cleanup (compute_confidence, filter_suggestion, byte-fallback guard,
# _postprocess, _clean_chunk) now lives in backends.py — applied via
# backend.postprocess() / backend.clean_chunk(). The legacy /api/* endpoints
# still call filter_suggestion / _has_byte_fallback_garbage directly (imported
# above); both are harmless no-ops on clean BPE output.


@app.get("/")
async def index():
    static_dir = Path(__file__).parent / "static"
    html_path = static_dir / "index-greedy.html"
    return HTMLResponse(html_path.read_text())


@app.get("/index-sampling.html")
async def index_sampling():
    static_dir = Path(__file__).parent / "static"
    html_path = static_dir / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/keyboard.html")
async def index_keyboard():
    static_dir = Path(__file__).parent / "static"
    html_path = static_dir / "predictive-keyboard.html"
    return HTMLResponse(html_path.read_text())


@app.get("/editor.html")
async def index_editor():
    """Ghost-text editor demo — FIM infill + AR append modes."""
    static_dir = Path(__file__).parent / "static"
    html_path = static_dir / "editor.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/autocomplete/greedy")
async def autocomplete_greedy(text: str = "", max_tokens: int = 3, filter_punctuation: bool = True, filter_numeric: bool = True):
    if not text.strip():
        return {"suggestion": "", "ghost_suffix": "", "confidence": 1.0, "latency_ms": 0.0, "smart_context": text, "candidates": []}

    ctx = smart_context(text)
    suggestion, confidence, candidates, latency = await _generate_with_repair(
        _prompt_to_ids(ctx), max_tokens, 0.0, greedy=True, filter_digits=filter_numeric
    )
    if filter_punctuation:
        suggestion = filter_suggestion(suggestion)
        # If suggestion is pure punct and user already ends with punct, drop it
        if _is_pure_punct(suggestion) and text.strip() and text.strip()[-1] in PUNCT_CHARS:
            suggestion = ""

    # Tokenization trap fallback: if the suggestion (or candidates) contain
    # byte-fallback garbage (non-Basque chars), the typed prefix's tokenization
    # is incompatible with the desired word (e.g. "zaud" → [▁za, u, d] can't
    # reach "zaude" → [▁zaude]). Fall back to keyboard retokenization, which
    # tries shorter prefixes and from-scratch generation to rescue the word.
    garbage_in_suggestion = bool(suggestion) and _has_byte_fallback_garbage(suggestion)
    garbage_in_candidates = bool(candidates) and all(
        _has_byte_fallback_garbage(c.get("text", "")) for c in candidates
    )
    if garbage_in_suggestion or garbage_in_candidates:
        kbc, kb_latency = await _keyboard_candidates(
            text, max_tokens=5, top_k=3, filter_digits=filter_numeric
        )
        latency += kb_latency
        # _keyboard_candidates already filters byte-fallback garbage internally
        if kbc:
            top = kbc[0]
            _, current_word = _extract_current_word(text)
            if top.get("is_next_word"):
                # Model thinks current word is complete; suggest next word
                suggestion = " " + top["text"]
            elif current_word and top["text"].startswith(current_word):
                # Word completion: show only the untyped suffix as ghost
                suggestion = top["text"][len(current_word):]
            else:
                suggestion = top["text"]
            # Replace garbage candidates with clean keyboard candidates
            candidates = [{"text": c["text"], "prob": c["prob"]} for c in kbc]
            confidence = top.get("prob", confidence)
        else:
            suggestion = ""
            candidates = []

    gs = ghost_suffix(text, ctx, suggestion)
    return {
        "suggestion": suggestion,
        "ghost_suffix": gs,
        "confidence": confidence,
        "candidates": candidates,
        "latency_ms": round(latency, 2),
        "prompt_length": len(ctx),
        "smart_context": ctx,
    }


@app.get("/api/autocomplete")
async def autocomplete(text: str = "", max_tokens: int = 10, temperature: float = 0.7, filter_punctuation: bool = True, filter_numeric: bool = True):
    if not text.strip():
        return {"suggestion": "", "ghost_suffix": "", "confidence": 1.0, "latency_ms": 0.0, "smart_context": text, "candidates": []}

    ctx = smart_context(text)
    suggestion, confidence, candidates, latency = await _generate_with_repair(
        _prompt_to_ids(ctx), max_tokens, temperature, greedy=False, filter_digits=filter_numeric
    )
    if filter_punctuation:
        suggestion = filter_suggestion(suggestion)
        if _is_pure_punct(suggestion) and text.strip() and text.strip()[-1] in PUNCT_CHARS:
            suggestion = ""
    gs = ghost_suffix(text, ctx, suggestion)
    return {
        "suggestion": suggestion,
        "ghost_suffix": gs,
        "confidence": confidence,
        "candidates": candidates,
        "latency_ms": round(latency, 2),
        "prompt_length": len(ctx),
        "smart_context": ctx,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("WebSocket connected")

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            text = msg.get("text", "")
            filter_punc = msg.get("filter_punctuation", True)
            filter_num = msg.get("filter_numeric", True)
            max_tokens = int(msg.get("max_tokens", 10))
            temperature = float(msg.get("temperature", 0.7))

            if not text.strip():
                await ws.send_json({
                    "type": "suggestion",
                    "suggestion": "",
                    "confidence": 1.0,
                    "latency_ms": 0.0,
                    "prompt_length": 0,
                    "smart_context": text,
                    "candidates": [],
                })
                continue

            ctx = smart_context(text)
            suggestion, confidence, candidates, latency = await _generate_with_repair(
                _prompt_to_ids(ctx), max_tokens, temperature, greedy=False, filter_digits=filter_num
            )
            if filter_punc:
                suggestion = filter_suggestion(suggestion)
                if _is_pure_punct(suggestion) and text.strip() and text.strip()[-1] in PUNCT_CHARS:
                    suggestion = ""
            gs = ghost_suffix(text, ctx, suggestion)

            await ws.send_json({
                "type": "suggestion",
                "suggestion": suggestion,
                "ghost_suffix": gs,
                "confidence": confidence,
                "candidates": candidates,
                "latency_ms": round(latency, 2),
                "prompt_length": len(ctx),
                "smart_context": ctx,
            })
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/ws/greedy")
async def websocket_endpoint_greedy(ws: WebSocket):
    await ws.accept()
    print("WebSocket (greedy) connected")

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            text = msg.get("text", "")
            filter_punc = msg.get("filter_punctuation", True)
            filter_num = msg.get("filter_numeric", True)
            max_tokens = int(msg.get("max_tokens", 3))
            keyboard_mode = msg.get("keyboard_mode", False)

            if not text.strip():
                await ws.send_json({
                    "type": "suggestion",
                    "suggestion": "",
                    "confidence": 1.0,
                    "latency_ms": 0.0,
                    "prompt_length": 0,
                    "smart_context": text,
                    "candidates": [],
                    "keyboard_mode": keyboard_mode,
                })
                continue

            ctx = smart_context(text)

            if keyboard_mode:
                # Keyboard mode: retokenization fallback for word completion.
                # Candidates are full words (e.g. "Kaixo"), not raw token text.
                # top_k=5: the frontend shows 3 chips but stores 5 in the sticky
                # pool, so lower-ranked candidates can carry forward when the
                # user types the first letter (sticky merge in the frontend).
                kb_top_k = int(msg.get("top_k", 5))
                candidates, latency = await _keyboard_candidates(
                    ctx, max_tokens=5, top_k=kb_top_k, filter_digits=filter_num
                )
                suggestion = candidates[0]["text"] if candidates else ""
                confidence = candidates[0]["prob"] if candidates else 0.0
                await ws.send_json({
                    "type": "suggestion",
                    "suggestion": suggestion,
                    "ghost_suffix": "",
                    "confidence": confidence,
                    "candidates": candidates,
                    "latency_ms": round(latency, 2),
                    "prompt_length": len(ctx),
                    "smart_context": ctx,
                    "keyboard_mode": True,
                })
                continue

            suggestion, confidence, candidates, latency = await _generate_with_repair(
                _prompt_to_ids(ctx), max_tokens, 0.0, greedy=True, filter_digits=filter_num
            )
            if filter_punc:
                suggestion = filter_suggestion(suggestion)
                if _is_pure_punct(suggestion) and text.strip() and text.strip()[-1] in PUNCT_CHARS:
                    suggestion = ""
            gs = ghost_suffix(text, ctx, suggestion)

            await ws.send_json({
                "type": "suggestion",
                "suggestion": suggestion,
                "ghost_suffix": gs,
                "confidence": confidence,
                "candidates": candidates,
                "latency_ms": round(latency, 2),
                "prompt_length": len(ctx),
                "smart_context": ctx,
            })
    except WebSocketDisconnect:
        print("WebSocket (greedy) disconnected")
    except Exception as e:
        print(f"WebSocket (greedy) error: {e}")
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/api/autocomplete/keyboard")
async def autocomplete_keyboard(text: str = "", max_tokens: int = 5, top_k: int = 3, filter_numeric: bool = True):
    """Keyboard-mode autocomplete with retokenization fallback.

    Returns full-word candidates (not raw token text). See _keyboard_candidates().
    """
    if not text.strip():
        return {"suggestion": "", "candidates": [], "latency_ms": 0.0, "smart_context": text}

    ctx = smart_context(text)
    candidates, latency = await _keyboard_candidates(
        ctx, max_tokens=max_tokens, top_k=top_k, filter_digits=filter_numeric
    )
    return {
        "suggestion": candidates[0]["text"] if candidates else "",
        "candidates": candidates,
        "latency_ms": round(latency, 2),
        "smart_context": ctx,
    }


@app.post("/api/log")
async def log_completion(payload: dict):
    """Log a user interaction for offline eval/replay.

    Frontend sends one of:
      {"event": "accept", "context": "...", "smart_context": "...",
       "current_word": "...", "accepted": {text, prob, is_next_word, is_punct},
       "candidates": [{text, prob, is_next_word}, ...]}
      {"event": "send", "text": "..."}
    """
    _log_event(payload)
    return {"ok": True}


@app.get("/api/log")
async def get_logs(limit: int = 100):
    """Return recent log entries (for quick inspection)."""
    if not LOG_FILE.exists():
        return {"entries": [], "file": str(LOG_FILE)}
    lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return {"entries": entries, "file": str(LOG_FILE), "total": len(lines)}


@app.get("/health")
async def health():
    try:
        r = await _client().get(f"{LLAMA_SERVER_URL}/health", timeout=5.0)
        llama_ok = r.status_code == 200
    except Exception:
        llama_ok = False

    model_name = Path(_current_model_path).name if _current_model_path else "unknown"

    return {
        "status": "ok" if llama_ok else "degraded",
        "model_loaded": llama_ok,
        "model": model_name,
        "device": "cpu",
        "backend": "llama.cpp",
    }


@app.get("/api/model")
async def get_model():
    """Return current model info and available models."""
    return {
        "current": {
            "name": Path(_current_model_path).name if _current_model_path else "none",
            "path": _current_model_path,
        },
        "available": _discover_models(),
    }


@app.post("/api/model/reload")
async def reload_model(model: str = ""):
    """Hot-reload to a different model.

    Query params:
        model: Model name (e.g., 'step_0016000.Q4_K_M.gguf') or absolute path.
               If empty, reloads with the same model (useful after crash).
    """
    global llama_process, _current_model_path

    # Resolve model path
    if model:
        if Path(model).exists():
            new_path = str(Path(model).resolve())
        else:
            found = False
            for d in [EXPORTS_DIR, MODELS_DIR, Path("/app/exports"), Path("/app/models")]:
                p = d / model
                if p.exists():
                    new_path = str(p.resolve())
                    found = True
                    break
            if not found:
                return {"error": f"Model not found: {model}", "available": _discover_models()}
    elif _current_model_path:
        new_path = _current_model_path
    else:
        return {"error": "No model specified and no current model loaded"}

    if not Path(new_path).exists():
        return {"error": f"Model file not found: {new_path}"}

    print(f"Hot-reloading model: {new_path}")
    old_name = Path(_current_model_path).name if _current_model_path else "none"
    new_name = Path(new_path).name

    # Start new llama-server (this stops old one first)
    port = int(LLAMA_SERVER_URL.split(":")[-1]) if ":" in LLAMA_SERVER_URL else 8080
    proc = start_llama_server(new_path, port)
    if proc:
        llama_process = proc

    return {
        "status": "ok",
        "previous": old_name,
        "current": new_name,
        "message": f"Reloaded from {old_name} → {new_name}",
    }


@app.get("/api/tokenize")
async def tokenize(text: str = ""):
    """Return token breakdown for visualization (backend-specific)."""
    if not text:
        return {"tokens": [], "smart_context": ""}
    return {
        "tokens": backend.visual_tokens(text),
        "smart_context": smart_context(text),
    }


# ── OpenAI-compatible face (/v1/completions, /v1/complete) ──────────────
# This is the reusable Basque inference server: any client speaking the
# OpenAI completion API (Continue.dev, Cody, codecompanion.nvim, Obsidian
# plugins) connects with zero model-specific code. Tokenization, FIM
# templating, and output cleanup are delegated to `backend` (backends.py) —
# so these routes are architecture-agnostic.


class CompletionRequest(BaseModel):
    """OpenAI /v1/completions request schema (subset we support)."""
    prompt: Union[str, List[int], List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    min_p: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    seed: Optional[int] = None
    repeat_penalty: Optional[float] = 1.1


def _normalize_prompt_to_ids(prompt):
    """Normalize an OpenAI prompt (str | list[int] | list[str]) for llama-server.

    Delegates to the backend: SentencePiece encodes strings to token IDs
    (fidelity to training); Llama/transformer passes strings through (llama.cpp
    tokenizes). Token-ID lists pass through unchanged on both.
    """
    return backend.normalize_prompt(prompt)


def _completion_id() -> str:
    return f"cmpl-{int(time.time() * 1000)}"


@app.post("/v1/completions")
async def openai_completions(req: CompletionRequest):
    """OpenAI-compatible text completion — the reusable server face.

    Accepts the standard schema; the backend encodes the prompt (SP → token
    IDs, or Llama → string); forwards to llama-server;
    reformats the response to OpenAI shape. Supports streaming (SSE) and
    non-streaming. Stop sequences are passed through to llama-server (which
    handles cross-token matching natively) and double-checked in backend.postprocess.

    Always-on strategies (non-streaming): backend.postprocess applies garbage
    filter, filter_suggestion, and confidence — see docs/ARCHITECTURE.md §3.3.
    Streaming: backend.clean_chunk strips ▁/U+FFFD per chunk (full strategies need
    the complete output).
    """
    prompt_ids = _normalize_prompt_to_ids(req.prompt)
    if not prompt_ids:
        return {"error": {"message": "empty prompt", "type": "invalid_request_error"}}

    stops = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    greedy = req.temperature is None or req.temperature <= 0.0

    payload: dict = {
        "prompt": prompt_ids,
        "n_predict": req.max_tokens or 16,
        "stream": bool(req.stream),
        "temperature": 0.0 if greedy else (req.temperature if req.temperature is not None else 1.0),
        "top_p": req.top_p if req.top_p is not None else 1.0,
        "top_k": req.top_k if req.top_k is not None else 0,
        "min_p": req.min_p if req.min_p is not None else 0.0,
        "repeat_penalty": req.repeat_penalty if req.repeat_penalty is not None else 1.1,
        "n_probs": N_PROBS,
        "stop": stops if stops else [],
    }
    if req.seed is not None:
        payload["seed"] = req.seed

    if req.stream:
        return StreamingResponse(
            _stream_openai(payload, stops),
            media_type="text/event-stream",
        )

    # ── Non-streaming ──
    r = await _client().post(f"{LLAMA_SERVER_URL}/completion", json=payload, timeout=60.0)
    r.raise_for_status()
    data = r.json()

    # ── Always-on strategies (see docs/ARCHITECTURE.md §3.3) ──
    text, confidence = backend.postprocess(data.get("content", ""), data, stops)
    finish_reason = "stop" if (
        not text or data.get("stopped_eos") or data.get("stopped_word") or data.get("stop")
    ) else "length"

    prompt_tokens = len(prompt_ids) if isinstance(prompt_ids, list) else backend.count_tokens(prompt_ids)
    completion_tokens = data.get("tokens_predicted", 0) or backend.count_tokens(text)

    return {
        "id": _completion_id(),
        "object": "text_completion",
        "created": int(time.time()),
        "model": _model_name(),
        "choices": [
            {"text": text, "index": 0, "finish_reason": finish_reason, "logprobs": None}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        # Extra field (not in OpenAI spec; clients ignore unknowns).
        # Thin clients (Obsidian plugin) use this to suppress low-quality ghosts.
        "confidence": confidence,
    }


async def _stream_openai(payload: dict, stops: list[str]):
    """Translate llama-server SSE stream → OpenAI SSE stream.

    llama-server emits:  data: {"content":"tok","stop":false,...}\n\n
    OpenAI expects:       data: {"choices":[{"text":"tok",...}],...}\n\n

    Stop sequences are passed to llama-server (native cross-token handling);
    we also guard client-side as a fallback. llama.cpp truncates at the stop
    sequence (stop text excluded from content).
    """
    cmpl_id = _completion_id()
    created = int(time.time())
    model = _model_name()

    def frame(text: str, finish: Optional[str] = None) -> str:
        chunk = {
            "id": cmpl_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {"text": text, "index": 0, "finish_reason": finish, "logprobs": None}
            ],
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    stop_emitted = False
    try:
        client = _client()
        async with client.stream(
            "POST", f"{LLAMA_SERVER_URL}/completion", json=payload, timeout=None
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    d = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                text = backend.clean_chunk(d.get("content", ""))
                # Client-side stop fallback (in case llama-server missed)
                if text and stops:
                    for s in stops:
                        if s and s in text:
                            text = text[: text.index(s)]
                            stop_emitted = True
                            if text:
                                yield frame(text)
                            yield frame("", "stop")
                            break
                    if stop_emitted:
                        break
                if text:
                    yield frame(text)
                if d.get("stop") or d.get("stopped_eos"):
                    if not stop_emitted:
                        yield frame("", "stop")
                        stop_emitted = True
                    break
        if not stop_emitted:
            yield frame("", "length")
        yield "data: [DONE]\n\n"
    except httpx.HTTPStatusError as e:
        err = {"error": {"message": str(e), "type": "api_error"}}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"


class CompleteRequest(BaseModel):
    """Body for /v1/complete convenience route."""
    prefix: str = ""
    suffix: str = ""
    max_tokens: int = 16
    temperature: float = 0.2
    top_k: int = 0  # 0 = disabled (full vocabulary)
    n: int = 1  # best-of-n: fire n parallel samples, return highest-confidence


async def _complete_once(client: httpx.AsyncClient, payload: dict, stops: list[str]) -> tuple[str, float, str]:
    """Fire one completion request and post-process the result.

    Returns (text, confidence, finish_reason). Used both for single
    requests and as a parallel unit in best-of-n sampling.
    """
    r = await client.post(f"{LLAMA_SERVER_URL}/completion", json=payload)
    r.raise_for_status()
    data = r.json()
    text, confidence = backend.postprocess(data.get("content", ""), data, stops)
    finish = "stop" if (not text or data.get("stopped_eos") or data.get("stop")) else "length"
    return text, confidence, finish


def _pick_best_of_n(results: list) -> tuple[str, float, str]:
    """From a list of (text, confidence, finish) tuples (or exceptions),
    pick the highest-confidence non-empty result.

    Used by best-of-n sampling in /v1/complete. Exceptions from failed
    parallel requests are silently skipped — one bad sample shouldn't
    sink the whole request.
    """
    valid = [r for r in results if isinstance(r, tuple) and r[0]]
    if valid:
        return max(valid, key=lambda r: r[1])
    return ("", 0.0, "stop")


@app.post("/v1/complete")
async def complete_prefix_suffix(req: CompleteRequest):
    """Convenience route for thin bespoke clients (Obsidian, Vim, CLI).

    Takes raw {prefix, suffix}; the server applies the FIM template
    (<PRE>{prefix}<SUF>{suffix}<MID>) when a suffix is provided.
    If suffix is empty, falls back to prefix-only AR completion.

    A 50-line client just POSTs {prefix: buffer[:cursor], suffix: buffer[cursor:]}
    and renders the returned text as ghost — no FIM-token or tokenizer knowledge.

    Always-on strategies: backend.postprocess applies garbage filter, filter_suggestion,
    and confidence on every request — see docs/ARCHITECTURE.md §3.3.

    Response: {text, confidence, finish_reason}.
    The client uses `confidence` to suppress low-quality ghosts (threshold
    is a client-side config, not a server param).

    FIM mode requires the FIM-capable model (Phase 6 checkpoint) to be loaded
    in llama-server. With the AR-only model, suffix is ignored (prefix-only).
    """
    use_fim = bool(req.suffix.strip())

    def _build_payload(p, s):
        return {
            "prompt": p,
            "n_predict": req.max_tokens,
            "stream": False,
            "temperature": req.temperature,
            "top_k": req.top_k,
            "top_p": 1.0,
            "min_p": 0.0,
            # FIM infill: no repeat penalty (legitimately reuses context words),
            # greedy sampling for deterministic ghost-text.
            "repeat_penalty": 1.0,
            "n_probs": N_PROBS,
            "stop": s,
        }

    async def _run(p, s):
        """Fire n samples (or 1) and return (text, confidence, finish)."""
        nonlocal req, client
        pl = _build_payload(p, s)
        if req.n <= 1:
            return await _complete_once(client, pl, s)
        results = await asyncio.gather(
            *[_complete_once(client, pl, s) for _ in range(req.n)],
            return_exceptions=True,
        )
        return _pick_best_of_n(results)

    client = _client()

    if use_fim:
        # ── FIM mode: the backend builds the infill prompt + stops ──
        # (SentencePiece: token-level <PRE>/<SUF>/<MID> to preserve ▁
        # markers; Llama: a string template). Clients send {prefix,
        # suffix} and stay agnostic of the sentinel convention.
        prompt, stops = backend.fim_prompt(req.prefix, req.suffix)
        if prompt:
            text, confidence, finish = await _run(prompt, stops)
            # ── FIM fallback: base models (Kimu 2B, Latxa 8B) emit EOS
            #    when they see FIM sentinels, returning empty text. Fall
            #    back to AR (prefix-only) so the user still gets a
            #    suggestion. FIM-capable models (Morpheus) return non-empty
            #    and skip the fallback (one call, no latency penalty).
            if not text.strip():
                ar_prompt = backend.ar_prompt(req.prefix)
                ar_stops = ["\n\n"]
                text, confidence, finish = await _run(ar_prompt, ar_stops)
        else:
            text, confidence, finish = "", 0.0, "stop"
    else:
        # ── AR mode: prefix-only completion ──
        prompt = backend.ar_prompt(req.prefix)
        stops = ["\n\n"]
        if not prompt:
            return {"text": "", "finish_reason": "stop"}
        text, confidence, finish = await _run(prompt, stops)

    return {"text": text, "confidence": confidence, "finish_reason": finish}


@app.get("/v1/models")
async def list_models_v1():
    """OpenAI-compatible model list (some clients probe this on startup)."""
    return {
        "object": "list",
        "data": [
            {"id": _model_name(), "object": "model", "owned_by": "morpheus"}
        ],
    }


def main():
    global llama_process, LLAMA_SERVER_URL, _current_model_path
    parser = argparse.ArgumentParser(description="Morpheus v2 Mamba Demo Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--llama-port", type=int, default=8080,
                        help="Port for llama-server (internal)")
    parser.add_argument("--model", default=None,
                        help="Path to GGUF model (env: MORPHEUS_MODEL, default: latest in exports/)")
    parser.add_argument("--no-launch-llama", action="store_true",
                        help="Don't start llama-server (assume it's already running)")
    args = parser.parse_args()

    global llama_process, LLAMA_SERVER_URL
    LLAMA_SERVER_URL = f"http://localhost:{args.llama_port}"

    model_path = _resolve_model(args.model)
    print(f"Resolved model: {model_path}")
    _current_model_path = model_path

    if not args.no_launch_llama:
        llama_process = start_llama_server(model_path, args.llama_port)

    import uvicorn
    print(f"\n🚀 Morpheus v2 Demo running at http://{args.host}:{args.port}")
    print(f"   Backend: llama-server at {LLAMA_SERVER_URL}")
    print(f"   Model: {model_path}")
    print(f"   Endpoints:")
    print(f"     GET  /api/model         — list available models")
    print(f"     POST /api/model/reload   — hot-swap model")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        stop_llama_server()


if __name__ == "__main__":
    main()
