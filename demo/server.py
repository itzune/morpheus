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
from fastapi.responses import HTMLResponse

# Tokenize endpoint (for demo token visualization)
import sentencepiece as spm
TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "basque_unigram_4000.model"
_sp = None

def get_tokenizer():
    global _sp
    if _sp is None:
        _sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    return _sp


def _prompt_to_ids(prompt: str) -> list[int]:
    """Encode the prompt to SentencePiece token IDs, matching training exactly.

    Two reasons this is required rather than sending a string prompt:

    1. NO BOS: the model was trained without <s> (pretokenize.py uses
       sp.encode(line) + </s> separators, never <s>). The GGUF now carries
       add_bos_token=false (set by export_hf.py), so llama-server won't prepend
       <s> even for string prompts — this part is solved at the artifact level.

    2. TOKENIZATION FIDELITY (the active reason): llama.cpp's built-in
       SentencePiece tokenizer does NOT always reproduce the reference
       sentencepiece library's tokenization for this 4K unigram vocab — they
       diverge on long words (e.g. "Unibertsitate" → different subword splits),
       which shifts the argmax and costs ~10 CSR points (string 17% vs token-ID
       28% on targets.json). Sending token IDs from the SAME SentencePiece
       model used in training/pretokenize/eval guarantees the model sees the
       exact token sequence it was trained on — matching the GPU eval path
       (src/eval_utils.py::evaluate_csr uses sp.encode(text, out_type=int)).
    """
    sp = get_tokenizer()
    return sp.encode(prompt, out_type=int)  # SentencePiece default: NO BOS


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


def tokenize_visual(text: str) -> list[dict]:
    """Tokenize text for visualization. Returns [{token, start, end}, ...]"""
    if not text:
        return []
    sp = get_tokenizer()
    result = []
    offset = 0
    for tid in sp.encode(text, out_type=int):
        piece = sp.id_to_piece(tid)
        # piece starts with \u2581 (space marker) — replace with actual space
        if piece.startswith("\u2581"):
            display = piece[1:]
            result.append({"token": display, "space_before": True})
        else:
            result.append({"token": piece, "space_before": False})
    return result

app = FastAPI(title="Morpheus v2 Mamba Demo")

LLAMA_SERVER_URL = "http://localhost:8080"
llama_process: subprocess.Popen | None = None

# Current model path (for health/metadata)
_current_model_path: str = ""


EXPORTS_DIR = Path(os.environ.get("MORPHEUS_MODELS_DIR", Path(__file__).resolve().parent.parent / "exports"))

# Also check the Docker volume mount point for HF-downloaded models
MODELS_DIR = Path(os.environ.get("MORPHEUS_MODELS_DIR", "/app/models"))

# Default model — change this when a new checkpoint is promoted
DEFAULT_MODEL = "morpheus-v2-mamba.Q4_K_M.gguf"

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
    # Default: explicit model, fallback to latest Q4_K_M in any search dir
    for d in search_dirs:
        p = d / DEFAULT_MODEL
        if p.exists():
            return str(p.resolve())
    for d in search_dirs:
        ggufs = sorted(d.glob("*.Q4_K_M.gguf"))
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
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{LLAMA_SERVER_URL}/completion", json=payload)
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
    sp = get_tokenizer()
    t0 = time.perf_counter()
    result = await _call_llama(prompt_ids, max_tokens, temperature, greedy)

    candidates = _extract_candidates(result, filter_digits=filter_digits)
    confidence = compute_confidence(result)
    content = result.get("content", "")

    if not filter_digits:
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
                return sp.decode_ids(repaired_ids), confidence, candidates, latency
        else:
            repaired_ids.append(chosen_id)

    if first_swap == -1:
        # No digit tokens found → return original content (zero extra cost)
        latency = (time.perf_counter() - t0) * 1000
        return content, confidence, candidates, latency

    # Re-generate from the first swap point (one extra request)
    new_prompt = prompt_ids + repaired_ids[: first_swap + 1]
    remaining = max_tokens - (first_swap + 1)
    prefix_text = sp.decode_ids(repaired_ids[: first_swap + 1])

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
    sp = get_tokenizer()
    t0 = time.perf_counter()

    text_before_word, current_word = _extract_current_word(text)

    # ── Next-word prediction (cursor after space) ──
    if not current_word:
        prompt_ids = sp.encode(text, out_type=int)
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
        fallback_paths.append((shorter_word, sp.encode(prefix, out_type=int), False))
    # Always add the from-scratch path if there's preceding context
    if text_before_word.strip():
        fallback_paths.append(("", sp.encode(text_before_word, out_type=int), True))

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


def compute_confidence(result: dict) -> float:
    """Extract average confidence from completion_probabilities.

    Excludes EOS/stop tokens (logprob near 0.0, prob near 1.0) —
    these are not real predictions and inflate the average.
    """
    probs = result.get("completion_probabilities", [])
    if not probs:
        return 1.0
    import math
    # Exclude EOS tokens (logprob > -0.01, i.e. prob > 99%)
    real_probs = [math.exp(p["logprob"]) for p in probs if p["logprob"] < -0.01]
    if not real_probs:
        return 0.0
    return round(sum(real_probs) / len(real_probs), 4)


# ── Suggestion cleanup ──────────────────────────
# Digit filtering is now done at the token level by _generate_with_repair(),
# which swaps digit-containing tokens for their best non-digit alternative
# from top-k logprobs. The old post-hoc string filter (filter_numeric_after_punct)
# has been removed.

# Punctuation filter (filter_suggestion) is retained for decoding artifacts
# (▁ markers, collapsed punct runs, replacement chars) — these are pipeline
# issues, not data contamination.


def _has_byte_fallback_garbage(text: str) -> bool:
    """Check if text contains non-Latin characters (byte-fallback garbage).

    When the tokenizer trap occurs (e.g. "zaud" → [▁za, u, d] can't reach
    "zaude" → [▁zaude]), the model predicts byte-fallback tokens (<0xB5>,
    <0xD0>, ...) which llama-server decodes into non-Latin Unicode (Cyrillic
    'е', U+FFFD, etc.).

    Basque uses Latin script: all legitimate non-ASCII characters (ñ, ç, ü,
    á, é, í, ó, ú) fall within Latin-1 Supplement (U+0080-U+00FF). Any
    character above U+00FF can only come from byte-fallback bytes forming
    unintended UTF-8 sequences, so it's garbage.
    """
    return any(ord(c) > 0xFF for c in text)


def _is_pure_punct(s: str) -> bool:
    """Check if string is only punctuation, no word characters."""
    return bool(s.strip()) and all(c in PUNCT_CHARS for c in s.strip())


def filter_suggestion(suggestion: str) -> str:
    """Remove decoding artifacts and undertraining junk from model output.

    Two categories of cleanup:
      - Decoding pipeline artifacts (always needed, not data-related):
        ▁ SentencePiece space markers, U+FFFD replacement chars at
        byte-fallback boundaries.
      - Undertraining junk (fades as model converges): collapsed punct
        runs ("da??"), trailing space-punct ("kaixo . ,"), pure-punct
        output.  The </s>-in-loss fix (separators now included in
        training) should reduce these over time, but they persist while
        the model is partially trained.

    - Preserves leading whitespace (word boundary signal)
    - Strips replacement characters (U+FFFD)
    - Replaces lone ▁ markers with spaces
    - Collapses whitespace
    - Keeps at most ONE trailing punctuation character
    - If only punctuation+whitespace remains, returns empty
    """
    if not suggestion:
        return suggestion

    # Replace ▁ markers with spaces, strip replacement chars
    text = suggestion.replace("▁", " ").replace("\ufffd", "")

    # Preserve leading whitespace (meaningful: indicates new word)
    # but strip trailing whitespace
    text = text.rstrip()

    # Collapse internal whitespace
    import re
    text = re.sub(r'\s+', ' ', text)

    # Collapse runs of 2+ punctuation chars (same or different) to single first char.
    # "da??) eta" → "da? eta"  "LLC.," → "LLC."
    text = re.sub(r'([{}])[{}]+'.format(re.escape(PUNCT_CHARS), re.escape(PUNCT_CHARS)), r'\1', text)

    # Strip trailing space-then-punct sequences: "hello . ," → "hello"
    while True:
        tail = re.search(r' [{}]+$'.format(re.escape(PUNCT_CHARS)), text)
        if not tail:
            break
        text = text[:-len(tail.group())]

    # If the result has no word characters (only punct/whitespace),
    # reject it. Exception: a bare "." or "?" without surrounding
    # whitespace is a legitimate sentence-ending suggestion.
    stripped = text.strip()
    if stripped:
        has_word = any(c.isalpha() for c in stripped)
        if not has_word:
            # Bare single punct (no whitespace around it)? Keep it.
            if stripped in PUNCT_CHARS and text == stripped:
                return stripped
            else:
                return ""  # e.g. " ." or " ," or " .," — junk

    return text


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
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{LLAMA_SERVER_URL}/health")
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
    """Return token breakdown for visualization."""
    if not text:
        return {"tokens": [], "smart_context": ""}
    sp = get_tokenizer()
    ids = sp.encode(text, out_type=int)
    tokens = []
    for tid in ids:
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            display = piece[1:] if len(piece) > 1 else " "
            tokens.append({"token": display, "space_before": True})
        else:
            tokens.append({"token": piece, "space_before": False})
    return {
        "tokens": tokens,
        "smart_context": smart_context(text),
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
