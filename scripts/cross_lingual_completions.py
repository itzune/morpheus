#!/usr/bin/env python3
"""
cross_lingual_completions.py — Generate greedy completions from English and Spanish prompts.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT = "checkpoints/best.pt"
TOKENIZER = "tokenizer/basque_unigram_4000.model"
OUTPUT = "eval/domain_eval/cross_lingual_completions.txt"
MAX_NEW_TOKENS = 25

PROMPTS = {
    "English": [
        "The weather is nice today and I will go for a",
        "I am learning a new language but it is still",
        "Tomorrow morning I have to go to work",
        "Having dinner with friends this weekend sounds",
        "I like to listen to music while I am",
        "Thank you very",
        "The United States of America is a country in",
        "In recent years, the development of technology has",
    ],
    "Spanish": [
        "El tiempo está bueno hoy y voy a salir a",
        "Estoy aprendiendo un idioma nuevo pero todavía es",
        "Mañana por la mañana tengo que ir a trabajar",
        "Cenar con amigos este fin de semana suena",
        "Me gusta escuchar música mientras estoy",
        "Los Estados",
        "La República Francesa es un país de",
        "En los últimos años, el desarrollo de la tecnología ha",
    ],
}


def main():
    from eval_baselines import MambaModelWrapper

    wrapper = MambaModelWrapper(CHECKPOINT, TOKENIZER, device="cuda")

    results = {}

    for lang, prompts in PROMPTS.items():
        results[lang] = []
        for prompt in prompts:
            prompt_tokens = wrapper.encode(prompt)
            if len(prompt_tokens) < 2:
                continue
            generated, logprobs = wrapper.generate_greedy(prompt_tokens, max_tokens=MAX_NEW_TOKENS)
            completion = wrapper.decode(generated)
            results[lang].append({
                "prompt": prompt,
                "completion": completion,
                "avg_logprob": sum(logprobs) / len(logprobs) if logprobs else 0,
            })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("Cross-Lingual Greedy Completions (step 74K, greedy)\n")
        f.write("=" * 70 + "\n\n")

        for lang in ["English", "Spanish"]:
            entries = results[lang]
            f.write(f"\n{'='*70}\n")
            f.write(f"LANGUAGE: {lang.upper()} ({len(entries)} samples)\n")
            f.write(f"{'='*70}\n\n")
            for e in entries:
                f.write(f"PROMPT:     {e['prompt']}\n")
                f.write(f"PREDICTION: {e['completion']}\n")
                f.write(f"AVG LOGPROB: {e['avg_logprob']:.3f}\n\n")

    print(f"\nResults saved to {OUTPUT}")

    for lang in ["English", "Spanish"]:
        entries = results[lang]
        print(f"\n{'='*70}")
        print(f"  {lang}")
        print(f"{'='*70}")
        for e in entries:
            print(f"  {e['prompt']}")
            print(f"  → {e['completion']}")
            print()


if __name__ == "__main__":
    main()
