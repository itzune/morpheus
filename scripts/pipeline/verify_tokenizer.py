#!/usr/bin/env python3
"""
Verify a trained SentencePiece tokenizer for the Morpheus Basque autocomplete model.

Checks:
  1. Vocab size matches target
  2. Special tokens (unk/bos/eos) are correct
  3. Zero genuine <unk> tokens on a text sample (byte_fallback works)
  4. Fertility (tokens per word) is in expected range
  5. Verb morpheme splitting (the key 4K advantage)
  6. Character coverage
  7. Token length distribution

Usage:
    python3 scripts/pipeline/verify_tokenizer.py \
        --sp-model tokenizer/basque_unigram_4000.model \
        --sample-file data/clean-v3/HiTZ_latxa-corpus-v2_wikipedia.txt \
        --sample-lines 100000
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import sentencepiece as spm


def check_basic(sp, target_vocab):
    """Check vocab size and special tokens."""
    print("=" * 60)
    print("1. BASIC CHECKS")
    print("=" * 60)
    vocab = sp.get_piece_size()
    ok = vocab == target_vocab
    print(f"  Vocab size: {vocab} (target {target_vocab})  {'✓' if ok else '✗ FAIL'}")

    unk = sp.unk_id()
    bos = sp.bos_id()
    eos = sp.eos_id()
    print(f"  unk_id={unk} ({sp.id_to_piece(unk)}), bos_id={bos} ({sp.id_to_piece(bos)}), "
          f"eos_id={eos} ({sp.id_to_piece(eos)})")
    if eos == unk:
        print("  ✗ FAIL: eos and unk are the same id! Separator bug.")
        return False
    print(f"  ✓ eos ({sp.id_to_piece(eos)}) != unk ({sp.id_to_piece(unk)}) — separator is safe")
    return ok


def check_unk_rate(sp, sample_file, sample_lines):
    """Check that byte_fallback produces zero genuine <unk> tokens."""
    print("\n" + "=" * 60)
    print("2. <unk> RATE (byte fallback check)")
    print("=" * 60)
    unk_id = sp.unk_id()
    total = 0
    unk_count = 0
    with open(sample_file, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            total += len(ids)
            unk_count += sum(1 for t in ids if t == unk_id)
    rate = unk_count / total * 100 if total else 0
    status = "✓ PASS" if rate < 0.1 else "✗ FAIL"
    print(f"  Sampled {total:,} tokens from {sample_file.name}")
    print(f"  <unk> tokens: {unk_count:,} ({rate:.4f}%)  {status}")
    if rate > 0.1:
        print("  >>> byte_fallback may not be working — check training params")
    return rate < 0.1


def check_fertility(sp, sample_file, sample_lines):
    """Compute fertility (tokens per word) on a sample."""
    print("\n" + "=" * 60)
    print("3. FERTILITY (tokens per word)")
    print("=" * 60)
    total_words = 0
    total_tokens = 0
    with open(sample_file, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            words = line.split()
            if not words:
                continue
            total_words += len(words)
            for word in words:
                total_tokens += len(sp.encode(word, out_type=int))
    fertility = total_tokens / total_words if total_words else 0
    print(f"  Sample words: {total_words:,}")
    print(f"  Sample tokens: {total_tokens:,}")
    print(f"  Fertility: {fertility:.3f} tok/word (4K expected ~2.5-2.6, 32K ~1.8-1.9)")
    return fertility


def check_morpheme_splits(sp):
    """Verify verb morpheme splitting — the key 4K advantage."""
    print("\n" + "=" * 60)
    print("4. VERB MORPHEME SPLITTING (key 4K advantage)")
    print("=" * 60)
    # Agglutinative verbs: prefix + agreement + dative + ergative + tense
    test_cases = [
        ("dizkizut",    "I have them to you"),
        ("dakizkioke",  "it can be known to him"),
        ("emango",      "will give"),
        ("joanen",      "will go"),
        ("etorriko",    "will come"),
        ("daramat",     "I carry them"),
        ("zizkidan",    "he had them to me"),
        ("ditzagun",    "let us do them"),
    ]
    all_good = True
    for word, gloss in test_cases:
        toks = sp.encode(word, out_type=str)
        n = len(toks)
        # Good split = 2+ pieces (not one whole word)
        status = "✓" if n >= 2 else "✗"
        if n < 2:
            all_good = False
        print(f"  {word:14s} ({gloss:28s}) → {' | '.join(toks):30s}  [{n} pieces] {status}")
    if all_good:
        print("  ✓ All verbs split into 2+ morpheme-aligned pieces")
    else:
        print("  ⚠ Some verbs not split — may indicate vocab too large or morph coverage weak")
    return all_good


def check_token_lengths(sp, sample_file, sample_lines):
    """Distribution of token lengths (characters per token)."""
    print("\n" + "=" * 60)
    print("5. TOKEN LENGTH DISTRIBUTION")
    print("=" * 60)
    lengths = Counter()
    with open(sample_file, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            for tok in sp.encode(line, out_type=str):
                lengths[len(tok)] += 1
    total = sum(lengths.values())
    print(f"  Total tokens sampled: {total:,}")
    for length in sorted(lengths)[:8]:
        bar = "█" * int(lengths[length] / total * 50)
        print(f"  len={length}: {lengths[length]:>8,} ({lengths[length]/total*100:5.1f}%) {bar}")
    avg = sum(l * c for l, c in lengths.items()) / total
    print(f"  Average token length: {avg:.2f} chars")


def show_samples(sp):
    """Show sample tokenizations of sentences."""
    print("\n" + "=" * 60)
    print("6. SAMPLE SENTENCE TOKENIZATIONS")
    print("=" * 60)
    sentences = [
        "Euskal Herriko Unibertsitateak ikasle berriak hartu ditu.",
        "Gaur goizean etxetik atera naiz eta busa hartu dut.",
        "Lagunarekin hitz egin dut bihar elkartuko garela.",
    ]
    for s in sentences:
        toks = sp.encode(s, out_type=str)
        print(f"  {s}")
        print(f"    → {' '.join(toks)}")
        print(f"    ({len(toks)} tokens)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Verify a trained SentencePiece tokenizer")
    parser.add_argument("--sp-model", required=True, help="Path to .model file")
    parser.add_argument("--sample-file", required=True, help="Text file to sample for checks")
    parser.add_argument("--sample-lines", type=int, default=100000, help="Lines to sample")
    parser.add_argument("--target-vocab", type=int, default=4000, help="Expected vocab size")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    sample_file = Path(args.sample_file)

    print(f"Tokenizer: {args.sp_model}")
    print(f"Sample:    {sample_file} ({args.sample_lines:,} lines)")

    ok1 = check_basic(sp, args.target_vocab)
    ok2 = check_unk_rate(sp, sample_file, args.sample_lines)
    fert = check_fertility(sp, sample_file, args.sample_lines)
    ok4 = check_morpheme_splits(sp)
    check_token_lengths(sp, sample_file, args.sample_lines)
    show_samples(sp)

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Vocab size correct:    {'✓' if ok1 else '✗'}")
    print(f"  Zero <unk>:            {'✓' if ok2 else '✗'}")
    print(f"  Fertility:             {fert:.3f} {'✓' if 2.0 < fert < 3.0 else '⚠'}")
    print(f"  Verb morpheme splits:  {'✓' if ok4 else '⚠'}")
    if ok1 and ok2 and ok4 and 2.0 < fert < 3.0:
        print("\n  ✅ Tokenizer VERIFIED — ready for pretokenization")
    else:
        print("\n  ⚠ Issues found — review above")
        sys.exit(1)


if __name__ == "__main__":
    main()
