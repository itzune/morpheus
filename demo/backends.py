"""Pluggable model backends for the Morpheus demo server.

The demo server speaks the OpenAI completion API to *clients* and the
llama-server `/completion` API to the *model*. The two ends are stable;
what varies between model architectures is the middle:

  1. **Tokenization** — does the proxy encode strings to token IDs itself
     (to bypass llama.cpp's divergent SentencePiece), or hand strings to
     llama-server and let it tokenize?
  2. **FIM template** — which infill sentinel tokens wrap the prefix/suffix?
     BigCode (`<PRE>/<SUF>/<MID>`), Code Llama (`<|fim_begin|>…`), an
     instruct chat template, …
  3. **Output cleanup** — which decoding-artifact filters apply? A custom
     SentencePiece model produces ▁ markers and byte-fallback garbage; a
     fine-tuned Llama (BPE) does not.

These three concerns are encapsulated in :class:`ModelBackend`. The
OpenAI-compatible routes (``/v1/completions``, ``/v1/complete``) and every
client (``editor.html``, Continue.dev, an Obsidian plugin, Vim, CLI) call
backend methods and are therefore **architecture-agnostic**. Swapping the
model means selecting a different backend — no client changes, no route
changes. See ``demo/README.md`` ("Reuse the server, swap the client").

Selecting a backend
-------------------
``MORPHEUS_BACKEND`` env var (or ``get_backend(name)``):

  - ``morpheus-sp-fim`` (default) — the current Morpheus Mamba-2 model with
    the custom 4K SentencePiece tokenizer + BigCode FIM tokens.
  - ``llama-fim`` — a Llama / transformer fine-tuned for FIM. Passes string
    prompts to llama-server (canonical BPE tokenizer), Code Llama FIM
    convention, minimal cleanup.

Adding a third architecture (e.g. a StarCoder-style transformer) is a new
subclass — see :class:`LlamaFIMBackend` for the shape.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Union, List

# ── Shared cleanup helpers ─────────────────────────────────────────────
# These operate on decoded text, not tokens. Backends apply whichever they
# need; the OpenAI routes go through backend.postprocess() so the choice is
# the backend's, not the route's.

PUNCT_CHARS = '.!,?;:()[]{}'


def is_pure_punct(s: str) -> bool:
    """True if `s` is only punctuation, no word characters."""
    return bool(s.strip()) and all(c in PUNCT_CHARS for c in s.strip())


def has_byte_fallback_garbage(text: str) -> bool:
    """True if `text` contains non-Latin characters (byte-fallback garbage).

    SentencePiece-specific: when the tokenizer trap occurs (e.g. "zaud" →
    [▁za, u, d] can't reach "zaude" → [▁zaude]), the model predicts
    byte-fallback tokens (<0xB5>, <0xD0>, …) which llama-server decodes into
    non-Latin Unicode (Cyrillic 'е', U+FFFD, …).

    Basque uses Latin script: all legitimate non-ASCII chars (ñ, ç, ü, á, é,
    í, ó, ú) fall within Latin-1 Supplement (U+0080–U+00FF). Anything above
    U+00FF can only come from byte-fallback bytes forming unintended UTF-8.

    Harmless on BPE models (Llama/transformer): their output is already
    clean Latin text, so this returns False.
    """
    return any(ord(c) > 0xFF for c in text)


def filter_suggestion(suggestion: str) -> str:
    """Remove decoding artifacts and undertraining junk from model output.

    Two categories of cleanup:
      - Decoding pipeline artifacts (always needed for SentencePiece):
        ▁ space markers, U+FFFD replacement chars at byte-fallback boundaries.
      - Undertraining junk (fades as the model converges): collapsed punct
        runs ("da??"), trailing space-punct ("kaixo . ,"), pure-punct output.

    Idempotent on clean BPE output (Llama/transformer): no ▁ markers or
    U+FFFD to strip, and the punct/whitespace rules only fire on actual
    junk — so it is safe to leave wired into the legacy /api/* endpoints
    when running a non-SentencePiece backend.
    """
    if not suggestion:
        return suggestion

    # Replace ▁ markers with spaces, strip replacement chars
    text = suggestion.replace("▁", " ").replace("\ufffd", "")

    # Preserve leading whitespace (word-boundary signal), strip trailing
    text = text.rstrip()

    # Collapse internal whitespace
    text = re.sub(r'\s+', ' ', text)

    # Collapse runs of 2+ punctuation chars (same or different) to the first.
    # "da??) eta" → "da? eta"   "LLC.," → "LLC."
    text = re.sub(
        r'([{}])[{}]+'.format(re.escape(PUNCT_CHARS), re.escape(PUNCT_CHARS)),
        r'\1', text)

    # Strip trailing space-then-punct sequences: "hello . ," → "hello"
    while True:
        tail = re.search(r' [{}]+$'.format(re.escape(PUNCT_CHARS)), text)
        if not tail:
            break
        text = text[:-len(tail.group())]

    # If the result has no word characters (only punct/whitespace), reject
    # it — unless it's a bare single sentence-ending punct (legit).
    stripped = text.strip()
    if stripped:
        has_word = any(c.isalpha() for c in stripped)
        if not has_word:
            if stripped in PUNCT_CHARS and text == stripped:
                return stripped
            return ""  # e.g. " ." or " ," or " .," — junk

    return text


def compute_confidence(result: dict) -> float:
    """Average per-token probability from llama-server's
    ``completion_probabilities``, excluding EOS/stop tokens (whose logprob
    is ~0.0 / prob ~1.0 and would inflate the average).

    Model-agnostic: works for any backend whose llama-server returns
    ``n_probs`` (all of them do).
    """
    probs = result.get("completion_probabilities", [])
    if not probs:
        return 1.0
    real_probs = [math.exp(p["logprob"]) for p in probs if p["logprob"] < -0.01]
    if not real_probs:
        return 0.0
    return round(sum(real_probs) / len(real_probs), 4)


# ── Backend interface ──────────────────────────────────────────────────

# What a backend hands to llama-server /completion as `prompt`: either a
# list of token IDs (to bypass llama.cpp's tokenizer) or a string (to let
# llama.cpp tokenize). The OpenAI routes accept whichever the backend
# returns and pass it through unchanged.
Prompt = Union[List[int], str]


class ModelBackend:
    """Encapsulates the three architecture-specific concerns.

    Subclasses need not touch any HTTP code — the routes call these methods.
    """

    name: str = "base"

    # ── Tokenization ──────────────────────────────────────────────────
    def encode(self, text: str) -> Prompt:
        """Encode `text` for llama-server's ``prompt`` field.

        Return ``list[int]`` to force a specific tokenization (SentencePiece
        fidelity), or ``str`` to let llama-server tokenize (Llama/transformer,
        where llama.cpp ships the canonical tokenizer).
        """
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text. Only needed by the legacy
        token-level digit-repair path (``_generate_with_repair``); backends
        without a local tokenizer may raise NotImplementedError — that path
        only triggers on digit-token spam, which converged BPE models don't
        produce.
        """
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Approximate token count (for usage stats). Backends with a local
        tokenizer return the exact count; others fall back to a heuristic.
        """
        return max(1, len(text) // 4)

    def normalize_prompt(self, prompt) -> Prompt:
        """Normalize an OpenAI ``prompt`` field (``str`` | ``list[int]`` |
        ``list[str]`` batch) to what llama-server expects.

        Default: strings → :meth:`encode`; token IDs → passthrough; batch →
        first string. Override if the backend wants strings through.
        """
        if isinstance(prompt, str):
            return self.encode(prompt)
        if isinstance(prompt, list) and prompt:
            if isinstance(prompt[0], int):
                return list(prompt)           # already token IDs
            if isinstance(prompt[0], str):
                return self.encode(prompt[0])  # batch → take first
        return []

    # ── Prompt construction ───────────────────────────────────────────
    def ar_prompt(self, prefix: str) -> Prompt:
        """Prompt for append / autoregressive completion (no suffix)."""
        return self.encode(prefix)

    def fim_prompt(self, prefix: str, suffix: str) -> tuple[Prompt, List[str]]:
        """Prompt + stop sequences for Fill-in-the-Middle infill.

        Returns ``(prompt, stops)``. The route sends ``stops`` to
        llama-server (native cross-token stop matching).
        """
        raise NotImplementedError

    # ── Output postprocessing ─────────────────────────────────────────
    def postprocess(self, content: str, data: dict, stops: List[str]) -> tuple[str, float]:
        """Clean raw llama-server output and score it.

        Returns ``(clean_text, confidence)``. ``data`` is the full
        llama-server JSON (for logprob access). Apply whatever cleanup this
        architecture needs: stop-sequence truncation, decoding-artifact
        filters, confidence.
        """
        text = content
        for s in stops:
            if s and s in text:
                text = text[: text.index(s)]
                break
        return text, compute_confidence(data)

    def clean_chunk(self, text: str) -> str:
        """Per-chunk cleanup for streaming (lightweight, no logprobs).

        Default: identity. SentencePiece backends override to strip ▁ and
        U+FFFD that would corrupt the stream mid-flight.
        """
        return text

    # ── Visualization (/api/tokenize) ─────────────────────────────────
    def visual_tokens(self, text: str) -> list[dict]:
        """Token breakdown for the demo UI's token visualizer.

        Returns ``[{"token": str, "space_before": bool}, …]``. Backends with
        a local tokenizer expose real piece boundaries; others approximate.
        """
        # Naive whitespace split — a reasonable fallback for backends
        # without a local tokenizer.
        out = []
        for i, word in enumerate(text.split(" ")):
            out.append({"token": word, "space_before": i > 0})
        return out


# ── Concrete backend: Morpheus Mamba-2 + custom SentencePiece ──────────

class SentencePieceFIMBackend(ModelBackend):
    """The current Morpheus model: Mamba-2 (91M) with a custom 4K unigram
    SentencePiece tokenizer extended with BigCode FIM tokens.

    Three architecture-specific reasons this backend exists (vs. handing
    strings to llama-server):

      1. **No BOS.** The model was trained without ``<s>``; the GGUF carries
         ``add_bos_token=false``, and we never prepend it.
      2. **Tokenization fidelity.** llama.cpp's built-in SentencePiece does
         not always reproduce the reference library's tokenization for this
         4K vocab — they diverge on long words, shifting the argmax and
         costing ~10 CSR points. Sending token IDs from the *same* model
         used in training guarantees the model sees its training-time
         token sequence.
      3. **BigCode FIM tokens.** ``<PRE>/<SUF>/<MID>/<EOT>`` are special
         pieces (IDs 4000–4003) applied at the token level so word-boundary
         ▁ markers survive across the sentinels.
    """

    name = "morpheus-sp-fim"

    # BigCode/StarCoder FIM sentinels (4 extra pieces added to the 4K vocab).
    FIM_PRE = "<PRE>"
    FIM_SUF = "<SUF>"
    FIM_MID = "<MID>"
    FIM_EOT = "<EOT>"  # the model emits this to end the infill

    def __init__(self, tokenizer_path: Path | str):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    # ── Tokenization ──
    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int)  # SentencePiece default: NO BOS

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode_ids(ids)

    def count_tokens(self, text: str) -> int:
        return len(self._sp.encode(text, out_type=int))

    # ── Prompt construction ──
    def fim_prompt(self, prefix: str, suffix: str) -> tuple[list[int], list[str]]:
        # Token-level FIM: [PRE] prefix_ids [SUF] suffix_ids [MID].
        # Encoding prefix and suffix independently then concatenating with
        # FIM token IDs as raw IDs preserves ▁ word-boundary markers
        # (BigCode/StarCoder approach) — avoids the string-level issue where
        # words after FIM tokens get character-split.
        PRE = self._sp.piece_to_id(self.FIM_PRE)
        SUF = self._sp.piece_to_id(self.FIM_SUF)
        MID = self._sp.piece_to_id(self.FIM_MID)
        prefix_ids = self._sp.encode(prefix, out_type=int)
        suffix_ids = self._sp.encode(suffix, out_type=int)
        prompt_ids = [PRE] + prefix_ids + [SUF] + suffix_ids + [MID]
        # <EOT> is a string stop because llama-server decodes token 4003 as
        # "<EOT>"; </s> (EOS) is handled natively by llama-server.
        # \n\n is the AR-style paragraph break.
        return prompt_ids, [self.FIM_EOT, "\n\n"]

    # ── Output postprocessing ──
    def postprocess(self, content: str, data: dict, stops: list[str]) -> tuple[str, float]:
        # 1. Strip stop sequences (<EOT>, </s>-EOS, \n\n, or client-provided).
        text = content
        for s in stops:
            if s and s in text:
                text = text[: text.index(s)]
                break
        # 2. Byte-fallback garbage → empty (catches the tokenizer-trap case
        #    where the model predicts byte-fallback tokens decoding to
        #    Cyrillic / U+FFFD). Confidence 0.0 so the client suppresses it.
        if has_byte_fallback_garbage(text):
            return "", 0.0
        # 3. ▁ markers, U+FFFD, collapsed punct, trailing junk.
        text = filter_suggestion(text)
        # 4. Average logprob excluding EOS/stop tokens.
        return text, compute_confidence(data)

    def clean_chunk(self, text: str) -> str:
        if not text:
            return text
        # The two artifacts that would corrupt a stream mid-flight; full
        # cleanup (garbage filter, filter_suggestion, confidence) needs the
        # complete output and is done in postprocess() for non-streaming.
        return text.replace("▁", " ").replace("\ufffd", "")

    # ── Visualization ──
    def visual_tokens(self, text: str) -> list[dict]:
        if not text:
            return []
        result = []
        for tid in self._sp.encode(text, out_type=int):
            piece = self._sp.id_to_piece(tid)
            if piece.startswith("\u2581"):  # ▁ space marker
                display = piece[1:] if len(piece) > 1 else " "
                result.append({"token": display, "space_before": True})
            else:
                result.append({"token": piece, "space_before": False})
        return result


# ── Concrete backend: fine-tuned Llama / transformer ───────────────────

class LlamaFIMBackend(ModelBackend):
    """A Llama (or other transformer) fine-tuned for FIM.

    This is the shape of the swap once Morpheus moves to a transformer
    architecture. The differences from :class:`SentencePieceFIMBackend`:

      - **No tokenizer bypass.** llama.cpp ships the canonical Llama BPE
        tokenizer, so we pass string prompts and let it tokenize. The
        SentencePiece-divergence problem doesn't exist here.
      - **Code Llama FIM convention.** ``<|fim_begin|>``/``<|fim_hole|>``/
        ``<|fim_end|>`` (Llama-3 ``<|fim_prefix|>`` etc. are equivalent —
        adjust the class attrs to match the fine-tune).
      - **No byte-fallback / ▁ cleanup.** BPE output is already clean Latin
        text; :meth:`postprocess` is just stop-strip + confidence.

    The token-ID ``decode`` path (legacy digit repair) is not supported —
    it's a SentencePiece-undertraining workaround that a converged Llama
    does not need.
    """

    name = "llama-fim"

    # FIM sentinels. Code Llama / Llama-3 convention — override per fine-tune.
    FIM_PRE = "<|fim_begin|>"   # wraps the prefix (≈ BigCode <PRE>)
    FIM_SUF = "<|fim_hole|>"    # wraps the suffix (≈ BigCode <SUF>)
    FIM_MID = "<|fim_end|>"     # marks the infill start (≈ BigCode <MID>)
    FIM_EOT = "<|eot_id|>"      # the model emits this to end the infill

    # ── Tokenization ──
    def encode(self, text: str) -> str:
        # Hand the string to llama-server; it tokenizes with the canonical
        # BPE. (Return the string, not IDs.)
        return text

    def decode(self, ids: list[int]) -> str:
        # No local tokenizer — return empty so any code path that calls
        # decode() degrades gracefully instead of crashing. The /v1/*
        # OpenAI routes never call this; the legacy /api/* greedy endpoint
        # skips digit repair for string-prompt backends (see server.py).
        return ""

    # ── Prompt construction ──
    def normalize_prompt(self, prompt) -> Prompt:
        # Strings and token-ID lists both go through to llama-server
        # unchanged; batch → first string.
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and prompt:
            if isinstance(prompt[0], int):
                return list(prompt)
            if isinstance(prompt[0], str):
                return prompt[0]
        return []

    def fim_prompt(self, prefix: str, suffix: str) -> tuple[str, list[str]]:
        # String-level FIM template. BPE tokenizes the sentinels as single
        # special tokens (they're in the vocab), so no token-ID surgery
        # needed — unlike SentencePiece, where ▁ markers must be preserved
        # across sentinels by concatenating token IDs.
        prompt = f"{self.FIM_PRE}{prefix}{self.FIM_SUF}{suffix}{self.FIM_MID}"
        return prompt, [self.FIM_EOT, "\n\n"]

    # ── Output postprocessing ──
    # Inherits the base: stop-strip + compute_confidence. No ▁/byte-fallback
    # cleanup — BPE output is clean. (filter_suggestion is still wired into
    # the legacy /api/* endpoints and is a harmless no-op on clean text.)


# ── Factory ────────────────────────────────────────────────────────────

# Default tokenizer path for the SentencePiece backend: the FIM tokenizer
# (4004 pieces = 4000 base + <PRE>/<SUF>/<MID>/<EOT>).
_DEFAULT_SP_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "basque_unigram_fim.model"

_REGISTRY: dict[str, type[ModelBackend]] = {
    SentencePieceFIMBackend.name: SentencePieceFIMBackend,
    LlamaFIMBackend.name: LlamaFIMBackend,
    # Convenience aliases
    "sp": SentencePieceFIMBackend,
    "llama": LlamaFIMBackend,
}


def get_backend(name: str | None = None) -> ModelBackend:
    """Return the configured backend.

    ``name`` overrides; otherwise read ``MORPHEUS_BACKEND`` env (default
    ``morpheus-sp-fim``). The SentencePiece backend is instantiated with
    the tokenizer at ``tokenizer/basque_unigram_fim.model`` (overridable via
    ``MORPHEUS_TOKENIZER``).
    """
    name = name or os.environ.get("MORPHEUS_BACKEND", "morpheus-sp-fim")
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown backend {name!r}. Known: {sorted(_REGISTRY)}"
        )
    if cls is SentencePieceFIMBackend:
        tok_path = os.environ.get("MORPHEUS_TOKENIZER", _DEFAULT_SP_PATH)
        return cls(tok_path)
    return cls()
