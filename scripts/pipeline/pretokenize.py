"""
Pre-tokenize corpus into a memory-mapped numpy array for Mamba training.

Reads raw text files from a directory, tokenizes them with a SentencePiece
Unigram model, and stores the token sequence as a uint16 .npy file for
efficient random access during training.

Uses a two-pass approach to avoid OOM on large corpora:
  Pass 1: count total tokens → pre-allocate numpy array on disk
  Pass 2: fill array incrementally (does not accumulate in RAM)

Uses </s> (eos_id, typically id=2) as the line separator between documents.
This separator IS included in the training loss (train.py uses ignore_index=0
for <unk> only), so the model learns to predict sentence boundaries.

Options:
    --exclude-sources  Comma-separated source names to skip (e.g. 'bopv,botha')
    --min-digit-run    Skip lines with N+ consecutive digits (0=off, 8=phone/ID)

Usage:
    python scripts/pipeline/pretokenize.py \\
        --sp-model tokenizer/basque_unigram_4000.model \\
        --input-dir data/clean-v3 \\
        --output data/train_tokens_4k.npy \\
        --exclude-sources bopv,botha \\
        --min-digit-run 8
"""

import argparse
import re
import numpy as np
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm


def _load_exclude_lines(path):
    """Load lines to exclude (for train/test split) into a set."""
    if not path:
        return None
    exclude = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                exclude.add(line)
    print(f"Loaded {len(exclude):,} lines to exclude (train/test split)")
    return exclude


def count_tokens(input_dir, sp, seq_sep_token, exclude_sources=None, min_digit_run=0,
                 exclude_lines=None):
    """Pass 1: count total tokens to determine array size."""
    exclude_sources = exclude_sources or []
    digit_re = re.compile(rf'\d{{{min_digit_run},}}') if min_digit_run > 0 else None
    total = 0
    skipped_leakage = 0
    for txt_file in sorted(input_dir.glob("*.txt")):
        if any(src in txt_file.name for src in exclude_sources):
            print(f"  [EXCLUDE] {txt_file.name}")
            continue
        with open(txt_file) as f:
            for line in tqdm(f, desc=f"Counting: {txt_file.name}"):
                line = line.strip()
                if not line:
                    continue
                if exclude_lines is not None and line in exclude_lines:
                    skipped_leakage += 1
                    continue
                if digit_re and digit_re.search(line):
                    continue
                total += len(sp.encode(line, out_type=int)) + 1  # +1 for sep
    if skipped_leakage:
        print(f"  [LEAKAGE FIX] Skipped {skipped_leakage:,} validation lines from training")
    return total


def fill_tokens(input_dir, sp, arr, seq_sep_token, exclude_sources=None, min_digit_run=0,
                exclude_lines=None):
    """Pass 2: fill pre-allocated array with tokens (no list accumulation)."""
    exclude_sources = exclude_sources or []
    digit_re = re.compile(rf'\d{{{min_digit_run},}}') if min_digit_run > 0 else None
    idx = 0
    for txt_file in sorted(input_dir.glob("*.txt")):
        if any(src in txt_file.name for src in exclude_sources):
            continue
        with open(txt_file) as f:
            for line in tqdm(f, desc=f"Filling: {txt_file.name}"):
                line = line.strip()
                if not line:
                    continue
                if exclude_lines is not None and line in exclude_lines:
                    continue
                if digit_re and digit_re.search(line):
                    continue
                tokens = sp.encode(line, out_type=int)
                n = len(tokens)
                arr[idx:idx+n] = tokens
                idx += n
                arr[idx] = seq_sep_token
                idx += 1
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize corpus into numpy array for Mamba training"
    )
    parser.add_argument("--sp-model", required=True, help="Path to SentencePiece .model file")
    parser.add_argument("--input-dir", required=True, help="Directory containing .txt files to tokenize")
    parser.add_argument("--output", required=True, help="Output .npy file path")
    parser.add_argument("--exclude-sources", default="",
                        help="Comma-separated source names to skip (e.g. 'bopv,botha')")
    parser.add_argument("--min-digit-run", type=int, default=0,
                        help="Skip lines with N+ consecutive digits (0=off, 8=phone/ID noise)")
    parser.add_argument("--exclude-lines-file", default="",
                        help="File of lines to exclude from training (train/test split). "
                             "Any line matching is skipped — fixes validation leakage.")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    input_dir = Path(args.input_dir)
    seq_sep_token = sp.eos_id()  # </s> (id=2) — included in loss, teaches sentence boundaries
    exclude_sources = [s.strip() for s in args.exclude_sources.split(",") if s.strip()]
    exclude_lines = _load_exclude_lines(args.exclude_lines_file)

    print(f"Separator token: id={seq_sep_token} ({sp.id_to_piece(seq_sep_token)})")
    if exclude_sources:
        print(f"Excluding sources: {exclude_sources}")
    if args.min_digit_run > 0:
        print(f"Filtering lines with {args.min_digit_run}+ consecutive digits")

    # Pass 1: count total tokens
    print("Pass 1: counting total tokens...")
    total_tokens = count_tokens(input_dir, sp, seq_sep_token,
                                exclude_sources=exclude_sources,
                                min_digit_run=args.min_digit_run,
                                exclude_lines=exclude_lines)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Array size:   {total_tokens * 2 / 1e9:.2f} GB (uint16)")

    # Pre-allocate array (not accumulated in RAM)
    print("Pass 2: filling token array...")
    arr = np.zeros(total_tokens, dtype=np.uint16)
    written = fill_tokens(input_dir, sp, arr, seq_sep_token,
                          exclude_sources=exclude_sources,
                          min_digit_run=args.min_digit_run,
                          exclude_lines=exclude_lines)
    print(f"Tokens written: {written:,}")

    # Validate
    unique_tokens = len(np.unique(arr[:1000000]))  # sample first 1M
    max_token = arr.max()
    print(f"  Max token ID (first 1M sample): {max_token}")
    if max_token > 65535:
        raise ValueError(
            f"Max token ID {max_token} exceeds uint16 range (65535). "
            f"Use uint32 dtype instead."
        )

    np.save(args.output, arr)
    print(f"Saved {written:,} tokens to {args.output}")
    print(f"Size: {arr.nbytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
