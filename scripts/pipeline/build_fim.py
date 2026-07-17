#!/usr/bin/env python3
"""
Build a FIM (Fill-in-the-Middle) + AR mixed training dataset.

Transforms the Latxa v2 corpus into a 50% FIM / 50% AR token stream,
following Bavarian et al. (2022) "Efficient Training of Language Models to
Fill in the Middle" and the morpheus TRAJECTORY §3 recipe.

For each line in the corpus:
  - 50% probability: FIM transform (char-level PSM or SPM, 50/50)
  - 50% probability: plain AR (unchanged)

FIM format:
  PSM: <PRE> prefix <SUF> suffix <MID> middle <EOT>
  SPM: <SUF> suffix <PRE> prefix <MID> middle <EOT>

The FIM string is SP-encoded as a single string — the FIM tokens are atomic
USER_DEFINED pieces, so they survive encoding intact. This matches how the
inference server will encode FIM prompts in P6 (`/v1/complete` builds the
same `<PRE>…<SUF>…<MID>` string and SP-encodes it).

Split strategy:
  - ~80% random-char splits (generalization)
  - ~20% linguistic-boundary splits (word/clause/sentence boundaries,
    mimicking where real editor cursors sit)

Two-pass (count → fill) with deterministic per-line RNG (seeded by global
line index) so both passes produce identical splits. Same memory-efficient
pattern as pretokenize.py — no list accumulation, just a counter in pass 1
and in-place array fill in pass 2.

Output: uint16 .npy (FIM token IDs 4000–4003 fit in uint16, max 65535).

Usage:
    # Full FIM training set (~9-10 GB)
    python3 scripts/pipeline/build_fim.py \
        --sp-model tokenizer/basque_unigram_fim.model \
        --input-dir data/clean-v3 \
        --output data/train_fim.npy

    # FIM validation set (from held-out Wikipedia)
    python3 scripts/pipeline/build_fim.py \
        --sp-model tokenizer/basque_unigram_fim.model \
        --input-dir data/valid \
        --output data/valid_fim.npy

    # Debug: inspect a few FIM examples
    python3 scripts/pipeline/build_fim.py \
        --sp-model tokenizer/basque_unigram_fim.model \
        --input-dir data/valid \
        --output /tmp/test_fim.npy \
        --max-lines 100 --debug
"""

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

# ── FIM token IDs (match add_fim_tokens.py: USER_DEFINED at 4000–4003) ──
# These are looked up from the tokenizer at runtime, but we assert they match.
EXPECTED_FIM_IDS = {
    "<PRE>": 4000,
    "<SUF>": 4001,
    "<MID>": 4002,
    "<EOT>": 4003,
}

# Characters that mark linguistic boundaries (for ~20% biased splits).
# After these characters is a natural cursor position: word boundary,
# clause boundary, or sentence boundary.
BOUNDARY_CHARS = set(" \t\n.,;:!?)]}\"'…—")


def verify_fim_tokens(sp: spm.SentencePieceProcessor):
    """Verify the tokenizer has the FIM tokens at expected IDs."""
    for tok, expected_id in EXPECTED_FIM_IDS.items():
        tid = sp.piece_to_id(tok)
        if tid != expected_id:
            raise ValueError(
                f"FIM token {tok!r} has id {tid}, expected {expected_id}. "
                f"Did you use --sp-model tokenizer/basque_unigram_fim.model?"
            )
    # Verify atomicity (USER_DEFINED tokens are never split)
    test = "Kaixo<PRE>mundua<SUF>ageri<MID>ra<EOT>"
    ids = sp.EncodeAsIds(test)
    assert 4000 in ids and 4001 in ids and 4002 in ids and 4003 in ids, \
        f"FIM tokens not atomic! ids={ids}"
    print(f"  ✓ FIM tokens verified: <PRE>={sp.piece_to_id('<PRE>')} "
          f"<SUF>={sp.piece_to_id('<SUF>')} <MID>={sp.piece_to_id('<MID>')} "
          f"<EOT>={sp.piece_to_id('<EOT>')}")


def find_word_boundaries(text: str) -> list:
    """Find char indices that are natural cursor positions (after boundaries).

    Returns a sorted list of positions where a cursor naturally sits:
    start of text, after spaces, after punctuation.
    """
    boundaries = [0]
    for i, c in enumerate(text):
        if c in BOUNDARY_CHARS and i + 1 < len(text):
            boundaries.append(i + 1)
    boundaries.append(len(text))
    # Deduplicate (multiple boundary chars in a row → same next position)
    return sorted(set(boundaries))


def make_fim_split(text: str, rng: random.Random,
                   boundary_bias: float = 0.2, min_mid: int = 3,
                   min_line: int = 20):
    """Split text into (prefix, middle, suffix) at char level.

    ~boundary_bias fraction of splits land on linguistic boundaries
    (word/clause/sentence); the rest are random char positions.

    Returns (prefix, middle, suffix) or None if the line is too short.
    """
    n = len(text)
    if n < min_line:
        return None

    if rng.random() < boundary_bias:
        # ── Linguistic boundary split ──
        boundaries = find_word_boundaries(text)
        if len(boundaries) >= 3:
            # Pick two boundaries with enough gap for the middle
            attempts = 0
            while attempts < 10:
                idx1 = rng.randint(0, len(boundaries) - 2)
                idx2 = rng.randint(idx1 + 1, len(boundaries) - 1)
                p1, p2 = boundaries[idx1], boundaries[idx2]
                if p2 - p1 >= min_mid:
                    break
                attempts += 1
            else:
                # Couldn't find a good boundary pair; fall back to random
                p1 = rng.randint(0, n - min_mid)
                p2 = rng.randint(p1 + min_mid, n)
        else:
            # Not enough boundaries; random split
            p1 = rng.randint(0, n - min_mid)
            p2 = rng.randint(p1 + min_mid, n)
    else:
        # ── Random char split ──
        p1 = rng.randint(0, n - min_mid)
        p2 = rng.randint(p1 + min_mid, n)

    return text[:p1], text[p1:p2], text[p2:]


def build_fim_string(prefix: str, middle: str, suffix: str, mode: str) -> str:
    """Build the FIM-formatted string for SP encoding.

    PSM (Prefix-Suffix-Middle): <PRE> prefix <SUF> suffix <MID> middle <EOT>
    SPM (Suffix-Prefix-Middle): <SUF> suffix <PRE> prefix <MID> middle <EOT>
    """
    if mode == "PSM":
        return f"<PRE>{prefix}<SUF>{suffix}<MID>{middle}<EOT>"
    else:  # SPM
        return f"<SUF>{suffix}<PRE>{prefix}<MID>{middle}<EOT>"


def process_line(line: str, line_idx: int, sp: spm.SentencePieceProcessor,
                 fim_rate: float, boundary_bias: float,
                 min_mid: int, min_line: int,
                 digit_re=None) -> list:
    """Process a single line into token IDs (FIM or AR + </s> separator).

    The FIM/AR decision and split points are deterministic (seeded by line_idx),
    so pass 1 (count) and pass 2 (fill) produce identical results.

    Returns a list of token IDs, or None if the line is filtered out.
    """
    line = line.strip()
    if not line:
        return None
    if digit_re and digit_re.search(line):
        return None

    # Deterministic RNG seeded by line index — both passes get the same result
    rng = random.Random(line_idx)

    if rng.random() < fim_rate:
        # ── FIM transform ──
        split = make_fim_split(line, rng, boundary_bias, min_mid, min_line)
        if split is None:
            # Line too short for FIM; fall back to AR
            tokens = sp.encode(line, out_type=int)
        else:
            prefix, middle, suffix = split
            mode = "PSM" if rng.random() < 0.5 else "SPM"
            fim_str = build_fim_string(prefix, middle, suffix, mode)
            tokens = sp.encode(fim_str, out_type=int)
    else:
        # ── Plain AR ──
        tokens = sp.encode(line, out_type=int)

    # </s> separator (same as pretokenize.py — included in loss, teaches doc boundaries)
    tokens.append(sp.eos_id())
    return tokens


def iterate_lines(input_dir: Path, exclude_sources: list):
    """Yield (line_idx, line) for all non-excluded .txt files, in sorted order.

    line_idx is a global counter across all files, used as the deterministic
    RNG seed so both passes produce identical FIM/AR decisions.
    """
    line_idx = 0
    for txt_file in sorted(input_dir.glob("*.txt")):
        if any(src in txt_file.name for src in exclude_sources):
            print(f"  [EXCLUDE] {txt_file.name}")
            continue
        with open(txt_file, encoding="utf-8") as f:
            for line in f:
                yield line_idx, line
                line_idx += 1


def count_tokens(input_dir, sp, fim_rate, boundary_bias, min_mid, min_line,
                 exclude_sources, digit_re):
    """Pass 1: count total tokens to determine array size."""
    total = 0
    n_fim = 0
    n_ar = 0
    n_skipped = 0
    for line_idx, line in tqdm(iterate_lines(input_dir, exclude_sources),
                               total=None, desc="Counting"):
        tokens = process_line(line, line_idx, sp, fim_rate, boundary_bias,
                              min_mid, min_line, digit_re)
        if tokens is None:
            n_skipped += 1
            continue
        total += len(tokens)
        # Count FIM vs AR (re-run the decision — deterministic, same result)
        rng = random.Random(line_idx)
        if rng.random() < fim_rate:
            split = make_fim_split(line.strip(), rng, boundary_bias, min_mid, min_line)
            if split is not None:
                n_fim += 1
            else:
                n_ar += 1  # fell back to AR
        else:
            n_ar += 1
    return total, n_fim, n_ar, n_skipped


def fill_tokens(input_dir, sp, arr, fim_rate, boundary_bias, min_mid, min_line,
                exclude_sources, digit_re):
    """Pass 2: fill pre-allocated array with tokens (no list accumulation)."""
    idx = 0
    for line_idx, line in tqdm(iterate_lines(input_dir, exclude_sources),
                               total=None, desc="Filling"):
        tokens = process_line(line, line_idx, sp, fim_rate, boundary_bias,
                              min_mid, min_line, digit_re)
        if tokens is None:
            continue
        n = len(tokens)
        arr[idx:idx + n] = tokens
        idx += n
    return idx


def debug_examples(input_dir, sp, fim_rate, boundary_bias, min_mid, min_line,
                   digit_re, n_examples=5):
    """Print a few FIM examples with their tokenization for inspection."""
    print("\n" + "=" * 70)
    print("  FIM Examples (debug)")
    print("=" * 70)
    shown = 0
    for line_idx, line in iterate_lines(input_dir, []):
        line = line.strip()
        if not line or (digit_re and digit_re.search(line)):
            continue
        rng = random.Random(line_idx)
        if rng.random() >= fim_rate:
            continue  # skip AR lines
        split = make_fim_split(line, rng, boundary_bias, min_mid, min_line)
        if split is None:
            continue
        prefix, middle, suffix = split
        mode = "PSM" if rng.random() < 0.5 else "SPM"
        fim_str = build_fim_string(prefix, middle, suffix, mode)
        tokens = sp.encode(fim_str, out_type=int)
        pieces = sp.encode(fim_str, out_type=str)

        print(f"\n  --- Example {shown + 1} (line {line_idx}, {mode}) ---")
        print(f"  Original:  {line[:100]}{'...' if len(line) > 100 else ''}")
        print(f"  Prefix:    {repr(prefix[:60])}{'...' if len(prefix) > 60 else ''}")
        print(f"  Middle:    {repr(middle[:60])}{'...' if len(middle) > 60 else ''}")
        print(f"  Suffix:    {repr(suffix[:60])}{'...' if len(suffix) > 60 else ''}")
        print(f"  FIM str:   {repr(fim_str[:80])}{'...' if len(fim_str) > 80 else ''}")
        print(f"  Tokens:    {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"  Pieces:    {pieces[:15]}{'...' if len(pieces) > 15 else ''}")
        # Verify FIM tokens are present
        fim_ids = {sp.piece_to_id("<PRE>"), sp.piece_to_id("<SUF>"),
                   sp.piece_to_id("<MID>"), sp.piece_to_id("<EOT>")}
        present = fim_ids & set(tokens)
        print(f"  FIM ids:   {sorted(present)} "
              f"({'✓ all 4' if len(present) == 4 else '⚠ missing some'})")

        shown += 1
        if shown >= n_examples:
            break
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Build FIM + AR mixed training dataset (char-level PSM/SPM)"
    )
    parser.add_argument("--sp-model", required=True,
                        help="Path to FIM SentencePiece .model (basque_unigram_fim.model)")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing .txt files")
    parser.add_argument("--output", required=True,
                        help="Output .npy file path")
    parser.add_argument("--fim-rate", type=float, default=0.5,
                        help="Fraction of lines transformed to FIM (default: 0.5)")
    parser.add_argument("--boundary-bias", type=float, default=0.2,
                        help="Fraction of FIM splits at linguistic boundaries (default: 0.2)")
    parser.add_argument("--min-mid", type=int, default=3,
                        help="Minimum middle span length in chars (default: 3)")
    parser.add_argument("--min-line", type=int, default=20,
                        help="Minimum line length for FIM (shorter → AR, default: 20)")
    parser.add_argument("--exclude-sources", default="",
                        help="Comma-separated source names to skip")
    parser.add_argument("--min-digit-run", type=int, default=0,
                        help="Skip lines with N+ consecutive digits (0=off)")
    parser.add_argument("--max-lines", type=int, default=0,
                        help="Process only N lines (0=all, for testing)")
    parser.add_argument("--debug", action="store_true",
                        help="Print FIM examples and exit (no output file)")
    args = parser.parse_args()

    # Load tokenizer and verify FIM tokens
    print(f"Loading tokenizer: {args.sp_model}")
    sp = spm.SentencePieceProcessor()
    sp.Load(args.sp_model)
    print(f"  Vocab size: {sp.get_piece_size()}")
    verify_fim_tokens(sp)

    input_dir = Path(args.input_dir)
    exclude_sources = [s.strip() for s in args.exclude_sources.split(",") if s.strip()]
    digit_re = re.compile(rf'\d{{{args.min_digit_run},}}') if args.min_digit_run > 0 else None
    eos_id = sp.eos_id()

    print(f"\nConfiguration:")
    print(f"  Input:          {input_dir}")
    print(f"  Output:         {args.output}")
    print(f"  FIM rate:       {args.fim_rate} ({args.fim_rate*100:.0f}% FIM, {(1-args.fim_rate)*100:.0f}% AR)")
    print(f"  Boundary bias:  {args.boundary_bias} ({args.boundary_bias*100:.0f}% linguistic, "
          f"{(1-args.boundary_bias)*100:.0f}% random)")
    print(f"  Min middle:     {args.min_mid} chars")
    print(f"  Min line:       {args.min_line} chars (for FIM)")
    print(f"  Separator:      </s> (id={eos_id})")
    if exclude_sources:
        print(f"  Exclude:        {exclude_sources}")
    if args.min_digit_run > 0:
        print(f"  Digit filter:   skip lines with {args.min_digit_run}+ consecutive digits")
    if args.max_lines > 0:
        print(f"  Max lines:      {args.max_lines} (testing mode)")

    # ── Debug mode: print examples and exit ──
    if args.debug:
        debug_examples(input_dir, sp, args.fim_rate, args.boundary_bias,
                       args.min_mid, args.min_line, digit_re)
        return

    # ── Pass 1: count tokens ──
    print(f"\nPass 1: counting tokens...")
    # Wrap iterate_lines to respect --max-lines
    if args.max_lines > 0:
        original_iterate = iterate_lines

        def limited_iterate(input_dir, exclude_sources):
            for i, (idx, line) in enumerate(original_iterate(input_dir, exclude_sources)):
                if i >= args.max_lines:
                    break
                yield idx, line
        # Monkey-patch for this run
        import types
        # Use the limited iterator
        total, n_fim, n_ar, n_skipped = _count_with_limit(
            input_dir, sp, args.fim_rate, args.boundary_bias,
            args.min_mid, args.min_line, exclude_sources, digit_re,
            args.max_lines, eos_id
        )
    else:
        total, n_fim, n_ar, n_skipped = count_tokens(
            input_dir, sp, args.fim_rate, args.boundary_bias,
            args.min_mid, args.min_line, exclude_sources, digit_re
        )

    print(f"\n  Total tokens:     {total:,}")
    print(f"  FIM lines:        {n_fim:,}")
    print(f"  AR lines:         {n_ar:,}")
    print(f"  Skipped (empty):  {n_skipped:,}")
    print(f"  Array size:       {total * 2 / 1e9:.2f} GB (uint16)")

    # ── Pre-allocate array ──
    print(f"\nPass 2: filling token array...")
    arr = np.zeros(total, dtype=np.uint16)

    if args.max_lines > 0:
        written = _fill_with_limit(
            input_dir, sp, arr, args.fim_rate, args.boundary_bias,
            args.min_mid, args.min_line, exclude_sources, digit_re,
            args.max_lines
        )
    else:
        written = fill_tokens(
            input_dir, sp, arr, args.fim_rate, args.boundary_bias,
            args.min_mid, args.min_line, exclude_sources, digit_re
        )

    print(f"\n  Tokens written: {written:,}")

    # ── Validate ──
    max_token = int(arr.max())
    print(f"  Max token ID:    {max_token}")
    if max_token > 65535:
        raise ValueError(f"Max token ID {max_token} exceeds uint16 range!")
    print(f"  Max token ID ≤ 4003 (FIM range)? {'✓' if max_token <= 4003 else '⚠ unexpected'}")

    # Count FIM token occurrences
    for tok_name, tok_id in [("PRE", 4000), ("SUF", 4001), ("MID", 4002), ("EOT", 4003)]:
        count = int((arr == tok_id).sum())
        print(f"  <{tok_name}> (id {tok_id}): {count:,} occurrences")

    # ── Save ──
    np.save(args.output, arr)
    print(f"\n  Saved: {args.output}")
    print(f"  Size:  {arr.nbytes / 1e9:.2f} GB")
    print(f"\n{'=' * 50}")
    print(f"✓ FIM dataset built: {args.output}")
    print(f"  {total:,} tokens ({n_fim:,} FIM + {n_ar:,} AR lines)")
    print(f"  Next: Phase 6 continued pretraining (P4)")
    print(f"{'=' * 50}")


def _count_with_limit(input_dir, sp, fim_rate, boundary_bias, min_mid, min_line,
                      exclude_sources, digit_re, max_lines, eos_id):
    """Count tokens with a line limit (for --max-lines testing)."""
    total = 0
    n_fim = 0
    n_ar = 0
    n_skipped = 0
    for i, (line_idx, line) in enumerate(iterate_lines(input_dir, exclude_sources)):
        if i >= max_lines:
            break
        tokens = process_line(line, line_idx, sp, fim_rate, boundary_bias,
                              min_mid, min_line, digit_re)
        if tokens is None:
            n_skipped += 1
            continue
        total += len(tokens)
        rng = random.Random(line_idx)
        if rng.random() < fim_rate:
            split = make_fim_split(line.strip(), rng, boundary_bias, min_mid, min_line)
            if split is not None:
                n_fim += 1
            else:
                n_ar += 1
        else:
            n_ar += 1
    return total, n_fim, n_ar, n_skipped


def _fill_with_limit(input_dir, sp, arr, fim_rate, boundary_bias, min_mid, min_line,
                     exclude_sources, digit_re, max_lines):
    """Fill array with a line limit (for --max-lines testing)."""
    idx = 0
    for i, (line_idx, line) in enumerate(iterate_lines(input_dir, exclude_sources)):
        if i >= max_lines:
            break
        tokens = process_line(line, line_idx, sp, fim_rate, boundary_bias,
                              min_mid, min_line, digit_re)
        if tokens is None:
            continue
        n = len(tokens)
        arr[idx:idx + n] = tokens
        idx += n
    return idx


if __name__ == "__main__":
    main()
