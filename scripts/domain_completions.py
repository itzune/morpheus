#!/usr/bin/env python3
"""
domain_completions.py — Generate greedy completions from domain-specific prompts.

Takes the first ~8 words from each domain file as a prompt,
generates a greedy continuation, and saves for qualitative analysis.
"""
import sys
import os
import math
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT = "checkpoints/best.pt"
TOKENIZER = "tokenizer/basque_unigram_4000.model"
CORPUS_DIR = "eval/domain_corpus"
OUTPUT = "eval/domain_eval/domain_completions.txt"
MAX_NEW_TOKENS = 25


def main():
    from eval_baselines import MambaModelWrapper

    wrapper = MambaModelWrapper(CHECKPOINT, TOKENIZER, device="cuda")

    results = defaultdict(list)

    for txt_file in sorted(Path(CORPUS_DIR).glob("*.txt")):
        domain = txt_file.name.split("_")[0]
        text = txt_file.read_text(encoding="utf-8").strip()
        if len(text) < 100:
            continue

        # Get prompt (first 8-10 words)
        words = text.split()
        prompt = " ".join(words[:9])
        if len(prompt) < 10:
            continue

        # Tokenize and generate
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

            for e in entries[:5]:  # max 5 per domain
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
