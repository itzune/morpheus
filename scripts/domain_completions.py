#!/usr/bin/env python3
"""
domain_completions.py — Generate greedy completions from domain-specific prompts.

Extracts real prose sentences (skipping navigation, HTML, CSS, boilerplate)
from each domain file, generates greedy continuations.
"""
import sys
import os
import re
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT = "checkpoints/best.pt"
TOKENIZER = "tokenizer/basque_unigram_4000.model"
CORPUS_DIR = "eval/domain_corpus"
OUTPUT = "eval/domain_eval/domain_completions.txt"
MAX_NEW_TOKENS = 25

# Boilerplate patterns to skip
SKIP_PATTERNS = [
    r"Edukira salto",
    r"Salto egin nabigazio",
    r"Fitxategiak deskargatzeko",
    r"Sartu Erabilera",
    r"\.c-mainarticle",
    r"@\s*media",
    r"display:\s*(block|none)",
    r"<!--",
    r"document\.all",
    r"loadMenus",
    r"^\s*\}+\s*$",
    r"^\s*\{+.*\}+\s*$",
    r"^\d+\.\s*bideoa",
    r"^\d+\.\s*gaia",
    r"^\d+\s*Unitate",
    r"^\d+\s*Blokea",
    r"^•\s",
    r"EFE\s*$",
    r"^\s*/\s*Bideoen",
    r"^\s*/\s*Oharrak",
]
SKIP_RE = re.compile("|".join(SKIP_PATTERNS), re.IGNORECASE)


def extract_prompts(text, domain, max_prompts=3):
    """Extract real prose sentences from text, skipping boilerplate."""
    lines = text.split("\n")
    prompts = []
    seen = set()

    for line in lines:
        line = line.strip()
        if len(line) < 40:
            continue
        if SKIP_RE.search(line):
            continue
        # Skip lines that are mostly punctuation or numbers
        alpha_count = sum(1 for c in line if c.isalpha())
        if alpha_count < len(line) * 0.5:
            continue
        # Skip duplicate content
        if line[:50] in seen:
            continue
        seen.add(line[:50])

        # Take first ~9-10 words as prompt
        words = line.split()
        if len(words) < 5:
            continue
        prompt = " ".join(words[:10])
        prompts.append(prompt)
        if len(prompts) >= max_prompts:
            break

    return prompts


def main():
    from eval_baselines import MambaModelWrapper

    wrapper = MambaModelWrapper(CHECKPOINT, TOKENIZER, device="cuda")

    results = defaultdict(list)
    seen_prompts = set()  # Deduplicate across files within same domain

    for txt_file in sorted(Path(CORPUS_DIR).glob("*.txt")):
        domain = txt_file.name.split("_")[0]
        text = txt_file.read_text(encoding="utf-8").strip()
        if len(text) < 100:
            continue

        prompts = extract_prompts(text, domain, max_prompts=2)

        for prompt in prompts:
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)

            prompt_tokens = wrapper.encode(prompt)
            if len(prompt_tokens) < 2:
                continue

            generated, logprobs = wrapper.generate_greedy(prompt_tokens, max_tokens=MAX_NEW_TOKENS)
            completion = wrapper.decode(generated)

            results[domain].append({
                "file": txt_file.name,
                "prompt": prompt,
                "completion": completion,
                "avg_logprob": sum(logprobs) / len(logprobs) if logprobs else 0,
            })

    # Write results
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("Domain-Stratified Greedy Completions (step 74K, greedy)\n")
        f.write("=" * 70 + "\n\n")

        for domain in ["legal", "wiki", "news", "education", "literature"]:
            entries = results.get(domain, [])
            if not entries:
                continue
            f.write(f"\n{'='*70}\n")
            f.write(f"DOMAIN: {domain.upper()} ({len(entries)} samples)\n")
            f.write(f"{'='*70}\n\n")

            for e in entries:
                f.write(f"--- {e['file']} ---\n")
                f.write(f"PROMPT:     {e['prompt']}\n")
                f.write(f"PREDICTION: {e['completion']}\n")
                f.write(f"AVG LOGPROB: {e['avg_logprob']:.3f}\n\n")

    print(f"\nResults saved to {OUTPUT}")

    # Print summary
    print(f"\n{'Domain':<14} {'Samples':>7} {'Avg LogProb':>12}")
    print("-" * 35)
    for domain in ["legal", "wiki", "news", "education", "literature"]:
        entries = results.get(domain, [])
        if entries:
            avg_lp = sum(e["avg_logprob"] for e in entries) / len(entries)
            print(f"{domain:<14} {len(entries):>7} {avg_lp:>12.3f}")


if __name__ == "__main__":
    main()
