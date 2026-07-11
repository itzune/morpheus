#!/usr/bin/env python3
"""
Build a FROZEN, HELD-OUT CSR test set from the validation text.

Source: data/valid/wiki_valid.txt — these lines were EXCLUDED from training
via the pretokenize --exclude-lines-file leakage fix. This is genuinely
unseen text, so CSR here measures generalization, not memorization.

Output: eval/csr_heldout.json — same format as eval/targets.json
  {"version": "v-heldout", "strategies": [{"name": "csr", "tests": [...]}]}

Each test: {"category": ..., "input": prompt, "target_completion": target}
  - One cut per sentence (mid-to-late), leaving a 4-12 word target
  - 300 sentences for strong statistical power (vs 30 in targets.json)
  - Fixed seed → reproducible / frozen

Usage:
    python3 scripts/build_csr_heldout.py
    python3 scripts/build_csr_heldout.py --count 500 --seed 42
"""
import argparse
import json
import random
import re
from pathlib import Path

VALID_FILE = Path("data/valid/wiki_valid.txt")
OUT_FILE = Path("eval/csr_heldout.json")

SPLIT = re.compile(r"(?<=[.!?])\s+")
ALLOWED = re.compile(r"^[A-Za-zÀ-ÿ''\- .,;:!?]+$")
REF = re.compile(r"\[[A-Za-z0-9]+\]")


def is_clean(s: str) -> bool:
    s = s.strip()
    if not (8 <= len(s.split()) <= 30):
        return False
    if not s[0].isupper():
        return False
    if s[-1] not in ".!?":
        return False
    if s.isupper():
        return False
    if not ALLOWED.match(s):
        return False
    if sum(c.isdigit() for c in s) > 2:
        return False
    if s.count(".") > 1 or s.count("!") > 1 or s.count("?") > 1:
        return False
    if "." in s[:-1]:
        return False
    return True


def load_clean_sentences(path):
    """Extract clean unique sentences from validation text."""
    text = path.read_text(encoding="utf-8", errors="replace")
    text = REF.sub("", text)
    sents = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sent in SPLIT.split(line):
            sent = sent.strip()
            if sent and is_clean(sent) and sent not in seen:
                seen.add(sent)
                sents.append(sent)
    return sents


def build_pairs(sentences, count, seed):
    """Cut each sentence into (prompt, target) pair at mid-to-late point."""
    rng = random.Random(seed)
    eligible = [s for s in sentences if len(s.split()) >= 8]
    rng.shuffle(eligible)

    pairs = []
    for s in eligible:
        words = s.split()
        # Cut mid-to-late: leave 4-12 word target
        lo = max(2, len(words) - 12)
        hi = len(words) - 4
        if hi <= lo:
            continue
        cut = rng.randint(lo, hi)
        prompt = " ".join(words[:cut])
        target = " ".join(words[cut:])
        # category by target word count
        tw = len(target.split())
        if tw <= 6:
            cat = "heldout_short"
        elif tw <= 9:
            cat = "heldout_medium"
        else:
            cat = "heldout_long"
        pairs.append({
            "category": cat,
            "input": prompt,
            "target_completion": target,
            "full_sentence": s,  # for reference, not used by eval
        })
        if len(pairs) >= count:
            break
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build frozen held-out CSR test set")
    parser.add_argument("--valid-file", default=str(VALID_FILE))
    parser.add_argument("--out-file", default=str(OUT_FILE))
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    valid_path = Path(args.valid_file)
    print(f"Loading validation text from {valid_path} ...")
    sentences = load_clean_sentences(valid_path)
    print(f"  Clean unique sentences (8+ words): {len(sentences)}")

    pairs = build_pairs(sentences, args.count, args.seed)
    print(f"  Built {len(pairs)} (prompt, target) pairs (seed={args.seed})")

    # Stats
    from collections import Counter
    cats = Counter(p["category"] for p in pairs)
    print(f"  Categories: {dict(cats)}")
    tws = [len(p["target_completion"].split()) for p in pairs]
    tcs = [len(p["target_completion"]) for p in pairs]
    import statistics
    print(f"  Target words:  min={min(tws)}, median={statistics.median(tws)}, max={max(tws)}")
    print(f"  Target chars:  min={min(tcs)}, median={statistics.median(tcs)}, max={max(tcs)}")
    total_chars = sum(tcs)
    print(f"  Total target chars: {total_chars:,}")

    # Show samples
    print(f"\n  Sample pairs:")
    for p in pairs[:5]:
        print(f"    [{p['category']}] '{p['input']}' → '{p['target_completion']}'")

    # Write in targets.json format
    out = {
        "version": "v-heldout-300",
        "description": (
            "Frozen held-out CSR test set. Source: data/valid/wiki_valid.txt "
            "(excluded from training via leakage fix). 300 sentences, seed=20260710. "
            "For bootstrap CI eval — see scripts/ppl_eval.py and eval.py."
        ),
        "strategies": [
            {
                "name": "csr",
                "description": f"CSR — {len(pairs)} held-out Wikipedia sentences, one cut each",
                "tests": pairs,
            }
        ],
    }
    out_path = Path(args.out_file)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓ Wrote {out_path} ({len(pairs)} tests)")


if __name__ == "__main__":
    main()
