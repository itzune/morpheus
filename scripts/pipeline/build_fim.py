#!/usr/bin/env python3
"""
Build a FIM (Fill-in-the-Middle) + AR mixed training dataset.

Transforms the Latxa v2 corpus into a 50% FIM / 50% AR token stream,
following Bavarian et al. (2022) "Efficient Training of Language Models to
Fill in the Middle" and the BigCode/StarCoder implementation.

KEY DESIGN: Token-level splitting (not string-level).
  1. Tokenize the full line first → token IDs with proper ▁ word markers
  2. Split at TOKEN boundaries (not char boundaries)
  3. Insert FIM token IDs between segments as raw IDs

This matches the BigCode/StarCoder `permute()` implementation and avoids
the string-level tokenization issue where words after FIM special tokens
lose their ▁ word-boundary marker and get character-split.

FIM format (token IDs, not strings):
  PSM: [PRE] prefix_ids [SUF] suffix_ids [MID] middle_ids [EOT]
  SPM: [SUF] suffix_ids [PRE] prefix_ids [MID] middle_ids [EOT]

Split strategy:
  - ~80% random token positions (generalization)
  - ~20% word-boundary positions (before ▁-prefixed tokens, mimicking
    where real editor cursors sit)

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
EXPECTED_FIM_IDS = {
    "<PRE>": 4000,
    "<SUF>": 4001,
    "<MID>": 4002,
    "<EOT>": 4003,
}


def verify_fim_tokens(sp: spm.SentencePieceProcessor):
    """Verify the tokenizer has the FIM tokens at expected IDs."""
    for tok, expected_id in EXPECTED_FIM_IDS.items():
        tid = sp.piece_to_id(tok)
        if tid != expected_id:
            raise ValueError(
                f"FIM token {tok!r} has id {tid}, expected {expected_id}. "
                f"Did you use --sp-model tokenizer/basque_unigram_fim.model?"
            )
    print(f"  ✓ FIM tokens verified: <PRE>={sp.piece_to_id('<PRE>')} "
          f"<SUF>={sp.piece_to_id('<SUF>')} <MID>={sp.piece_to_id('<MID>')} "
          f"<EOT>={sp.piece_to_id('<EOT>')}")


def find_word_boundary_positions(tokens: list, sp: spm.SentencePieceProcessor) -> list:
    """Find token indices that are natural cursor positions (word starts).

    In SentencePiece UNIGRAM, word-initial tokens start with ▁ (U+2581).
    A word boundary in token space is the position BEFORE such a token.
    Also includes position 0 (start of text) and len(tokens) (end).
    """
    boundaries = [0]
    for i in range(1, len(tokens)):
        piece = sp.id_to_piece(tokens[i])
        if piece.startswith("▁"):
            boundaries.append(i)
    boundaries.append(len(tokens))
    return sorted(set(boundaries))


def make_fim_split_tokens(tokens: list, sp: spm.SentencePieceProcessor,
                          rng: random.Random, boundary_bias: float = 0.2,
                          min_mid: int = 3):
    """Split token list into (prefix, middle, suffix) at token boundaries.

    ~boundary_bias fraction of splits land at word boundaries (before ▁ tokens).
    The rest are random token positions.

    This is the BigCode/StarCoder approach: tokenize first, split at token
    level. Each segment retains proper ▁ word markers from the original
    tokenization — no re-encoding needed.

    Returns (prefix_ids, middle_ids, suffix_ids) or None if too short.
    """
    n = len(tokens)
    if n < min_mid + 2:
        return None

    if rng.random() < boundary_bias:
        # ── Word-boundary split ──
        boundaries = find_word_boundary_positions(tokens, sp)
        if len(boundaries) >= 3:
            attempts = 0
            while attempts < 10:
                idx1 = rng.randint(0, len(boundaries) - 2)
                idx2 = rng.randint(idx1 + 1, len(boundaries) - 1)
                p1, p2 = boundaries[idx1], boundaries[idx2]
                if p2 - p1 >= min_mid:
                    break
                attempts += 1
            else:
                p1 = rng.randint(0, n - min_mid)
                p2 = rng.randint(p1 + min_mid, n)
        else:
            p1 = rng.randint(0, n - min_mid)
            p2 = rng.randint(p1 + min_mid, n)
    else:
        # ── Random token split ──
        p1 = rng.randint(0, n - min_mid)
        p2 = rng.randint(p1 + min_mid, n)

    return tokens[:p1], tokens[p1:p2], tokens[p2:]


def process_line(line: str, line_idx: int, sp: spm.SentencePieceProcessor,
                 fim_rate: float, boundary_bias: float,
                 min_mid: int, min_line: int,
                 digit_re=None) -> tuple:
    """Process a single line into token IDs (FIM or AR + </s> separator).

    Token-level FIM: tokenize the full line first (preserving ▁ word markers),
    then split at token boundaries and insert FIM token IDs as raw IDs.
    This matches the BigCode/StarCoder approach and avoids the string-level
    tokenization issue where words after FIM tokens lose their ▁ marker.

    The FIM/AR decision and split points are deterministic (seeded by line_idx),
    so pass 1 (count) and pass 2 (fill) produce identical results.

    Returns (token_ids, is_fim) or (None, False) if the line is filtered out.
    """
    line = line.strip()
    if not line:
        return None, False
    if digit_re and digit_re.search(line):
        return None, False

    # Tokenize the FULL line first — proper ▁ markers everywhere
    tokens = sp.encode(line, out_type=int)

    # Deterministic RNG seeded by line index
    rng = random.Random(line_idx)

    if rng.random() < fim_rate and len(tokens) >= min_line:
        # ── FIM transform at TOKEN level ──
        split = make_fim_split_tokens(tokens, sp, rng, boundary_bias, min_mid)
        if split is None:
            tokens.append(sp.eos_id())
            return tokens, False

        prefix_ids, middle_ids, suffix_ids = split
        mode = "PSM" if rng.random() < 0.5 else "SPM"

        PRE = sp.piece_to_id("<PRE>")
        SUF = sp.piece_to_id("<SUF>")
        MID = sp.piece_to_id("<MID>")
        EOT = sp.piece_to_id("<EOT>")

        if mode == "PSM":
            fim_ids = [PRE] + prefix_ids + [SUF] + suffix_ids + [MID] + middle_ids + [EOT]
        else:  # SPM
            fim_ids = [SUF] + suffix_ids + [PRE] + prefix_ids + [MID] + middle_ids + [EOT]

        fim_ids.append(sp.eos_id())
        return fim_ids, True
    else:
        # ── Plain AR ──
        tokens.append(sp.eos_id())
        return tokens, False


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
        tokens, is_fim = process_line(line, line_idx, sp, fim_rate, boundary_bias,
                                      min_mid, min_line, digit_re)
        if tokens is None:
            n_skipped += 1
            continue
        total += len(tokens)
        if is_fim:
            n_fim += 1
        else:
            n_ar += 1
    return total, n_fim, n_ar, n_skipped


def fill_tokens(input_dir, sp, arr, fim_rate, boundary_bias, min_mid, min_line,
                exclude_sources, digit_re):
    """Pass 2: fill pre-allocated array with tokens (no list accumulation)."""
    idx = 0
    for line_idx, line in tqdm(iterate_lines(input_dir, exclude_sources),
                               total=None, desc="Filling"):
        tokens, is_fim = process_line(line, line_idx, sp, fim_rate, boundary_bias,
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
    print("  FIM Examples (debug — token-level splitting)")
    print("=" * 70)
    shown = 0
    for line_idx, line in iterate_lines(input_dir, []):
        line = line.strip()
        if not line or (digit_re and digit_re.search(line)):
            continue
        # Tokenize full line first
        tokens = sp.encode(line, out_type=int)
        rng = random.Random(line_idx)
        if rng.random() >= fim_rate or len(tokens) < min_line:
            continue  # skip AR lines and short lines
        split = make_fim_split_tokens(tokens, sp, rng, boundary_bias, min_mid)
        if split is None:
            continue
        prefix_ids, middle_ids, suffix_ids = split
        mode = "PSM" if rng.random() < 0.5 else "SPM"

        PRE = sp.piece_to_id("<PRE>")
        SUF = sp.piece_to_id("<SUF>")
        MID = sp.piece_to_id("<MID>")
        EOT = sp.piece_to_id("<EOT>")

        if mode == "PSM":
            fim_ids = [PRE] + prefix_ids + [SUF] + suffix_ids + [MID] + middle_ids + [EOT]
        else:
            fim_ids = [SUF] + suffix_ids + [PRE] + prefix_ids + [MID] + middle_ids + [EOT]

        fim_pieces = [sp.id_to_piece(i) for i in fim_ids]

        # Decode segments for display
        prefix_text = sp.decode(prefix_ids) if prefix_ids else ""
        middle_text = sp.decode(middle_ids) if middle_ids else ""
        suffix_text = sp.decode(suffix_ids) if suffix_ids else ""

        print(f"\n  --- Example {shown + 1} (line {line_idx}, {mode}) ---")
        print(f"  Original:  {line[:100]}{'...' if len(line) > 100 else ''}")
        print(f"  Prefix:    {repr(prefix_text[:60])} ({len(prefix_ids)} tokens)")
        print(f"  Middle:    {repr(middle_text[:60])} ({len(middle_ids)} tokens)")
        print(f"  Suffix:    {repr(suffix_text[:60])} ({len(suffix_ids)} tokens)")
        print(f"  FIM ids:   {fim_ids[:20]}{'...' if len(fim_ids) > 20 else ''}")
        print(f"  Pieces:    {fim_pieces[:15]}{'...' if len(fim_pieces) > 15 else ''}")
        # Verify word markers preserved
        word_tokens = sum(1 for p in fim_pieces if p.startswith("▁"))
        print(f"  ▁-tokens:  {word_tokens}/{len(fim_pieces)} have word marker")

        shown += 1
        if shown >= n_examples:
            break
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Build FIM + AR mixed training dataset (token-level PSM/SPM)"
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
                        help="Fraction of FIM splits at word boundaries (default: 0.2)")
    parser.add_argument("--min-mid", type=int, default=3,
                        help="Minimum middle span length in tokens (default: 3)")
    parser.add_argument("--min-line", type=int, default=10,
                        help="Minimum line length in tokens for FIM (default: 10)")
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
    print(f"  Boundary bias:  {args.boundary_bias} ({args.boundary_bias*100:.0f}% word, "
          f"{(1-args.boundary_bias)*100:.0f}% random)")
    print(f"  Min middle:     {args.min_mid} tokens")
    print(f"  Min line:       {args.min_line} tokens (for FIM)")
    print(f"  Split method:   TOKEN-LEVEL (BigCode/StarCoder approach)")
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
    if args.max_lines > 0:
        total, n_fim, n_ar, n_skipped = _count_with_limit(
            input_dir, sp, args.fim_rate, args.boundary_bias,
            args.min_mid, args.min_line, exclude_sources, digit_re,
            args.max_lines
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
                      exclude_sources, digit_re, max_lines):
    """Count tokens with a line limit (for --max-lines testing)."""
    total = 0
    n_fim = 0
    n_ar = 0
    n_skipped = 0
    for i, (line_idx, line) in enumerate(iterate_lines(input_dir, exclude_sources)):
        if i >= max_lines:
            break
        tokens, is_fim = process_line(line, line_idx, sp, fim_rate, boundary_bias,
                                      min_mid, min_line, digit_re)
        if tokens is None:
            n_skipped += 1
            continue
        total += len(tokens)
        if is_fim:
            n_fim += 1
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
        tokens, is_fim = process_line(line, line_idx, sp, fim_rate, boundary_bias,
                                      min_mid, min_line, digit_re)
        if tokens is None:
            continue
        n = len(tokens)
        arr[idx:idx + n] = tokens
        idx += n
    return idx


if __name__ == "__main__":
    main()
