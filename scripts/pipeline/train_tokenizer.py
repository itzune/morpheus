#!/usr/bin/env python3
"""
Train a 4K SentencePiece Unigram tokenizer on the full Basque corpus.

Per the tokenizer fieldwork conclusions (docs/tokenizer-fieldwork.md):
  - 4K Unigram achieves MorphAcc consistency of 66.7% (vs 28.6% at 32K)
  - The drop is monotonic: 4K > 8K > 16K > 32K
  - This confirms QuechuaTok (Contreras, 2026) for Basque
  - Fertility penalty: 2.58 vs 1.85 (39% more tokens per word)
  - Mitigated by increasing seq_len from 512 to 1024

Usage:
    python3 scripts/pipeline/train_tokenizer.py \
        --input-dir /root/morpheus-mamba/data/clean \
        --output-dir tokenizer \
        --vocab-size 4000
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import sentencepiece as spm


def concatenate_corpus(input_dir: str, output_file: str, exclude_sources=None):
    """Concatenate all .txt files from input_dir into a single training file.

    Args:
        exclude_sources: List of source name substrings to skip (e.g. ['bopv', 'botha']).
    """
    exclude_sources = exclude_sources or []
    input_path = Path(input_dir)
    all_files = sorted(input_path.glob("*.txt"))
    txt_files = [f for f in all_files
                 if not any(src in f.name for src in exclude_sources)]

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    excluded = [f for f in all_files if any(src in f.name for src in exclude_sources)]
    print(f"Found {len(all_files)} source files, using {len(txt_files)}, excluding {len(excluded)}:")
    total_size = 0
    for f in txt_files:
        size_mb = f.stat().st_size / 1e6
        total_size += f.stat().st_size
        print(f"  [KEEP]  {f.name}: {size_mb:.1f} MB")
    for f in excluded:
        print(f"  [SKIP]  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    print(f"  Total kept: {total_size / 1e6:.1f} MB ({total_size / 1e9:.2f} GB)")

    # Create a temporary concatenated file for SentencePiece training
    # SentencePiece reads the entire file, so we need one input
    print(f"\nConcatenating into {output_file}...")
    with open(output_file, "wb") as out:
        for txt_file in txt_files:
            with open(txt_file, "rb") as inp:
                while True:
                    chunk = inp.read(64 * 1024 * 1024)  # 64 MB chunks
                    if not chunk:
                        break
                    out.write(chunk)
                # Add newline between files
                out.write(b"\n")

    out_size = os.path.getsize(output_file) / 1e6
    print(f"  Written: {out_size:.1f} MB")


def train_tokenizer(
    input_file: str,
    output_prefix: str,
    vocab_size: int = 4000,
    character_coverage: float = 0.9995,
    model_type: str = "unigram",
):
    """
    Train a SentencePiece Unigram tokenizer.

    Parameters match the fieldwork experiment that achieved 66.7% MorphAcc.
    Key: smaller vocab (4K) forces morpheme-aligned splits.
    """
    start = time.time()

    print(f"\nTraining Unigram tokenizer...")
    print(f"  Input:            {input_file}")
    print(f"  Vocab size:       {vocab_size}")
    print(f"  Char coverage:    {character_coverage}")
    print(f"  Model type:       {model_type}")

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        max_sentence_length=16777216,  # no truncation
        input_sentence_size=3_000_000,  # 3M sentences — sufficient for 4K vocab
        shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 4,
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        user_defined_symbols=[],
    )

    elapsed = time.time() - start
    print(f"  Training time:    {elapsed:.1f}s")

    # Load and validate
    sp = spm.SentencePieceProcessor()
    sp.load(f"{output_prefix}.model")

    actual_vocab = sp.get_piece_size()
    print(f"  Actual vocab:     {actual_vocab}")
    print(f"  Model saved:      {output_prefix}.model")
    print(f"  Vocab saved:      {output_prefix}.vocab")

    # Quick fertility check on a sample
    print("\nComputing fertility on sample...")
    total_words = 0
    total_tokens = 0
    with open(input_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 500_000:
                break
            words = line.split()
            if not words:
                continue
            total_words += len(words)
            for word in words:
                tokens = sp.encode(word, out_type=str)
                total_tokens += len(tokens)

    fertility = total_tokens / total_words if total_words > 0 else 0
    print(f"  Sample words:     {total_words:,}")
    print(f"  Sample tokens:    {total_tokens:,}")
    print(f"  Fertility:        {fertility:.2f} (expected ~2.58)")

    return fertility


def show_sample_tokenization(sp_path: str):
    """Show how the tokenizer segments a few Basque words."""
    sp = spm.SentencePieceProcessor()
    sp.load(sp_path)

    test_words = [
        "etxea",
        "etxera",
        "etxetik",
        "lagunarekin",
        "mendira",
        "gizonarentzat",
        "kalean",
        "etxe", "etxea", "etxeak", "etxearekin", "etxearentzat",
    ]

    print("\nTokenization samples:")
    for word in test_words:
        tokens = sp.encode(word, out_type=str)
        print(f"  {word:20s} → {' + '.join(tokens)}")

    # Full sentence sample
    sentence = "etxetik etxera joan naiz eta lagunarekin hitz egin dut"
    tokens = sp.encode(sentence, out_type=str)
    print(f"\n  Sentence: {sentence}")
    print(f"  Tokens:   {' '.join(tokens)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train 4K Unigram tokenizer for Morpheus v2.1"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing cleaned .txt files (data/clean-v3/)"
    )
    parser.add_argument(
        "--output-dir", default="tokenizer",
        help="Output directory for tokenizer files"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=4000,
        help="Vocabulary size (default: 4000, per fieldwork conclusions)"
    )
    parser.add_argument(
        "--character-coverage", type=float, default=0.9995,
        help="Character coverage (default: 0.9995)"
    )
    parser.add_argument(
        "--exclude-sources", default="",
        help="Comma-separated source names to skip (e.g. 'bopv,botha'). "
             "Tokenizer and model should train on the same data distribution."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exclude_sources = [s.strip() for s in args.exclude_sources.split(",") if s.strip()]

    # Concatenate corpus for SentencePiece
    merged_file = os.path.join(args.output_dir, "_training_corpus.txt")
    print("=" * 60)
    print("  Morpheus v2.1 — 4K Unigram Tokenizer Training")
    print("=" * 60)

    concatenate_corpus(args.input_dir, merged_file, exclude_sources=exclude_sources)

    # Train
    model_prefix = os.path.join(args.output_dir, f"basque_unigram_{args.vocab_size}")
    fertility = train_tokenizer(
        input_file=merged_file,
        output_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )

    # Sample tokenizations
    show_sample_tokenization(f"{model_prefix}.model")

    # Clean up merged file (it's huge)
    print(f"\nCleaning up {merged_file}...")
    os.remove(merged_file)

    print(f"\n{'=' * 60}")
    print(f"  Tokenizer ready: {model_prefix}.model")
    print(f"  Fertility:       {fertility:.2f} (expect 2.5–2.6)")
    print(f"  MorphAcc target: 66–67% (vs 28.6% at baseline 32K)")
    print(f"\n  Next step: pre-tokenize corpus with 4K tokenizer")
    print(f"    python scripts/pipeline/pretokenize.py \\")
    print(f"      --sp-model {model_prefix}.model \\")
    print(f"      --input-dir {args.input_dir} \\")
    print(f"      --output data/train_tokens_4k.npy")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
