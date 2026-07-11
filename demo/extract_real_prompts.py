#!/usr/bin/env python3
"""
Extract clean, well-formed Basque sentences from REAL prose sources
(Wikipedia + Berria) and turn them into autocomplete eval prompts.

Why this instead of hand-written prompts or the noisy corpus_sample:
  - Authentic, well-formed journalistic/encyclopedic prose (clean SOV
    sentences, proper morphology) — exactly the register an autocomplete
    should serve.
  - Reproducible (fixed seed + fetched source files in eval/real_corpus/).
  - We keep the GOLD continuation (the actual rest of the sentence), so the
    expert can judge whether the model's suggestion matches or is at least
    a plausible alternative. This is a real signal, not a guess.

Sources are fetched by eval/fetch_real_corpus.sh into eval/real_corpus/.
Output: demo/real_prompts.json  (list of {source, prompt, gold, full}).
"""
import json
import random
import re
from pathlib import Path

SRC = Path("eval/real_corpus")
OUT = Path("demo/real_prompts.json")
SEED = 20260709

# Allowed chars: Basque/Latin letters (incl. accents, ñ, ç, ü), spaces,
# apostrophe, hyphen, and ordinary punctuation.
ALLOWED = re.compile(r"^[A-Za-zÀ-ÿ''\- .,;:!?]+$")
SPLIT = re.compile(r"(?<=[.!?])\s+")
# Wikipedia extract reference markers like [12], [a]
REF = re.compile(r"\[[A-Za-z0-9]+\]")
# parenthetical asides often break sentence cleanliness
PAREN = re.compile(r"\([^)]*\)")


def is_clean(s: str) -> bool:
    s = s.strip()
    if not (6 <= len(s.split()) <= 30):
        return False
    if not s[0].isupper():
        return False
    if s[-1] not in ".!?":
        return False
    if s.isupper():
        return False
    if any(len(w) >= 4 and w.isupper() for w in s.split()):
        return False
    if not ALLOWED.match(s):
        return False
    if sum(c.isdigit() for c in s) > 2:
        return False
    # single terminal sentence only
    if s.count(".") > 1 or s.count("!") > 1 or s.count("?") > 1:
        return False
    if "." in s[:-1]:
        return False
    # reject if any token looks like a section header remnant (Title Case
    # chain of >=5 words with no verb is hard to detect cheaply; skip)
    return True


def load_sentences():
    sents = []
    for f in sorted(SRC.glob("*.txt")):
        src = f.stem  # e.g. wiki_Euskara, berria_0
        text = f.read_text(encoding="utf-8", errors="replace")
        text = REF.sub("", text)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sent in SPLIT.split(line):
                sent = sent.strip()
                if not sent:
                    continue
                # drop parentheticals then re-check
                sent = PAREN.sub("", sent).strip()
                if is_clean(sent):
                    sents.append((src, sent))
    # dedupe by sentence text, keep first source
    seen = {}
    for src, s in sents:
        if s not in seen:
            seen[s] = src
    return [(src, s) for s, src in seen.items()]


def main():
    random.seed(SEED)
    sents = load_sentences()
    print(f"Clean unique sentences from {SRC}: {len(sents)}")

    buckets = {"short(6-9)": [], "mid(10-15)": [], "long(16-30)": []}
    for src, s in sents:
        n = len(s.split())
        if n <= 9:
            buckets["short(6-9)"].append((src, s))
        elif n <= 15:
            buckets["mid(10-15)"].append((src, s))
        else:
            buckets["long(16-30)"].append((src, s))
    for k, v in buckets.items():
        print(f"  {k}: {len(v)}")

    # source diversity: prefer spreading across sources. Build per-source
    # pools, then round-robin sample to maximize source variety.
    target = {"short(6-9)": 8, "mid(10-15)": 20, "long(16-30)": 12}
    records = []
    for name, pool in buckets.items():
        random.shuffle(pool)
        # group by source for round-robin
        by_src = {}
        for src, s in pool:
            by_src.setdefault(src, []).append((src, s))
        src_order = list(by_src.keys())
        random.shuffle(src_order)
        added = 0
        idx = 0
        want = target[name]
        while added < want and any(by_src[s] for s in src_order):
            src = src_order[idx % len(src_order)]
            if by_src[src]:
                _, s = by_src[src].pop()
                words = s.split()
                # cut middle-to-late, leave >=3 words gold
                lo = max(2, len(words) - 9)
                hi = len(words) - 3
                if hi <= lo:
                    idx += 1
                    continue
                cut = random.randint(lo, hi)
                records.append({
                    "bucket": name,
                    "source": src,
                    "prompt": " ".join(words[:cut]),
                    "gold": " ".join(words[cut:]),
                    "full": s,
                })
                added += 1
            idx += 1

    print(f"\nExtracted {len(records)} prompts (seed={SEED}).")
    # source distribution
    from collections import Counter
    sd = Counter(r["source"] for r in records)
    print("Source distribution:", dict(sd))
    print("\nSample prompts (prompt ||| GOLD):")
    for r in records[:10]:
        print(f"  [{r['source']}] {r['prompt']} ||| {r['gold']}")

    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT} ({len(records)} prompts)")


if __name__ == "__main__":
    main()
