#!/usr/bin/env python3
"""
Blinded A/B evaluation: 32K vs 54K completions on fresh held-out prompts.

R10 from docs/eval-reform-proposal.md — the gold-standard human evaluation.
The Basque expert judges which completion is better, WITHOUT knowing which
checkpoint produced which. This is the only metric that captures "valid Basque
alternative continuation" with authority (the agglutinative multiple-valid-forms
problem that defeats exact-match CSR).

Process:
  1. Generate 20 fresh prompts from held-out validation text (seed=20260711,
     guaranteed non-overlapping with csr_heldout.json seed=20260710)
  2. For EACH checkpoint (32K, 54K), generate greedy completions (15 tokens)
     on the GPU (f16, reference sentencepiece — no quantization confound)
  3. Randomly assign A/B per prompt (seed=42)
  4. Output TWO files:
     - eval/ab_eval/blinded.md      → the expert reads this (A/B only, no labels)
     - eval/ab_eval/key.json        → the decoding key (DO NOT show the expert)
  5. Expert reads blinded.md, judges each pair (A better / B better / tie)
  6. Run scripts/ab_eval_reveal.py with the judgments → reveals winner

Generation semantics (matches training exactly):
  - No BOS (model trained without BOS)
  - Greedy decoding (temperature=0 → argmax), matching the autocomplete use case
  - Reference sentencepiece for encode/decode (NOT llama.cpp SP)
  - Stop at </s> (eos_id=2) or max_tokens

Usage:
    python3 scripts/ab_eval_generate.py
    python3 scripts/ab_eval_generate.py --count 20 --max-tokens 15
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

VALID_FILE = Path("data/valid/wiki_valid.txt")
CSR_HELDOUT_FILE = Path("eval/csr_heldout.json")
OUT_DIR = Path("eval/ab_eval")

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


def load_existing_prompts(csr_file):
    """Load full sentences already used in csr_heldout.json to avoid overlap."""
    if not csr_file.exists():
        return set()
    data = json.loads(csr_file.read_text())
    existing = set()
    for strat in data.get("strategies", []):
        if strat["name"] == "csr":
            for t in strat["tests"]:
                existing.add(t.get("full_sentence", ""))
    return existing


def load_prior_run(prior_dir):
    """Load full sentences used in a prior A/B run to avoid overlap.

    Reconstructs full sentences from blinded.json (prompts) + key.json (golds).
    Returns a set of full sentences.
    """
    prior_dir = Path(prior_dir)
    blinded_path = prior_dir / "blinded.json"
    key_path = prior_dir / "key.json"
    if not blinded_path.exists() or not key_path.exists():
        return set()
    blinded = json.loads(blinded_path.read_text())
    key = json.loads(key_path.read_text())
    # prompts from blinded.json, golds from key.json assignments (same order)
    prompts = [p["prompt"] for p in blinded["pairs"]]
    golds = [a["gold"] for a in key["assignments"]]
    existing = set()
    for p, g in zip(prompts, golds):
        existing.add(f"{p} {g}")
    return existing


def build_fresh_prompts(sentences, count, seed, exclude):
    """Build fresh (prompt, gold) pairs, excluding already-used sentences."""
    rng = random.Random(seed)
    eligible = [s for s in sentences if len(s.split()) >= 8 and s not in exclude]
    rng.shuffle(eligible)

    prompts = []
    for s in eligible:
        words = s.split()
        # Cut mid-to-late, leave 4-10 word gold
        lo = max(2, len(words) - 10)
        hi = len(words) - 4
        if hi <= lo:
            continue
        cut = rng.randint(lo, hi)
        prompt = " ".join(words[:cut])
        gold = " ".join(words[cut:])
        prompts.append({"prompt": prompt, "gold": gold, "full": s})
        if len(prompts) >= count:
            break
    return prompts


def _pad_vocab(vocab_size, multiple):
    return ((vocab_size + multiple - 1) // multiple) * multiple


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]
    pad_multiple = raw_cfg.get("pad_vocab_size_multiple", 16)
    vocab_size = raw_cfg.get("padded_vocab_size",
                             _pad_vocab(raw_cfg["vocab_size"], pad_multiple))
    config = MambaConfig(
        d_model=raw_cfg["d_model"],
        n_layer=raw_cfg["n_layer"],
        vocab_size=vocab_size,
        ssm_cfg={
            "layer": raw_cfg.get("ssm_layer", "Mamba2"),
            "d_state": raw_cfg["d_state"],
            "d_conv": raw_cfg["d_conv"],
            "expand": raw_cfg["expand"],
            "headdim": raw_cfg["headdim"],
            "chunk_size": raw_cfg.get("chunk_size", 256),
        },
        residual_in_fp32=raw_cfg.get("residual_in_fp32", True),
        fused_add_norm=raw_cfg.get("fused_add_norm", True),
        rms_norm=True,
    )
    model = MambaLMHeadModel(config, device=device, dtype=torch.bfloat16)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    step = ckpt.get("step", "?")
    del ckpt
    return model, step


@torch.no_grad()
def generate_greedy(model, sp, prompt, max_new_tokens=15, device="cuda"):
    """Generate greedy completion from prompt (no BOS, matches training)."""
    tokens = sp.encode(prompt, out_type=int)
    eos_id = sp.eos_id()  # </s> = 2
    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids).logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1).item()
        if next_id == 0:  # <unk> — stop
            break
        if next_id == eos_id:  # </s> — stop
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    # Decode only the generated tokens (not the prompt)
    completion = sp.decode(generated) if generated else ""
    return completion


def main():
    parser = argparse.ArgumentParser(description="Blinded A/B eval generation")
    parser.add_argument("--ckpt-a", default="checkpoints/step_0032000.pt",
                        help="Checkpoint A (will be randomly assigned as A or B)")
    parser.add_argument("--ckpt-b", default="checkpoints/step_0054000.pt",
                        help="Checkpoint B")
    parser.add_argument("--tokenizer", default="tokenizer/basque_unigram_4000.model")
    parser.add_argument("--valid-file", default=str(VALID_FILE))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260711, help="Prompt sampling seed")
    parser.add_argument("--blind-seed", type=int, default=42, help="A/B randomization seed")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="Output directory (use different dir for each batch)")
    parser.add_argument("--exclude-prior-dir", default=None,
                        help="Prior A/B run dir to exclude (avoid prompt reuse across batches)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    # Load fresh prompts (non-overlapping with csr_heldout)
    print(f"Loading validation text from {args.valid_file} ...")
    sentences = load_clean_sentences(Path(args.valid_file))
    print(f"  Clean sentences (8+ words): {len(sentences)}")
    exclude = load_existing_prompts(CSR_HELDOUT_FILE)
    print(f"  Excluding {len(exclude)} sentences already in csr_heldout.json")
    if args.exclude_prior_dir:
        prior = load_prior_run(args.exclude_prior_dir)
        print(f"  Excluding {len(prior)} sentences from prior A/B run ({args.exclude_prior_dir})")
        exclude |= prior
    prompts = build_fresh_prompts(sentences, args.count, args.seed, exclude)
    print(f"  Built {len(prompts)} fresh prompts (seed={args.seed})")
    for i, p in enumerate(prompts[:3], 1):
        print(f"    {i}. '{p['prompt']}' → [gold: {p['gold']}]")

    # Generate completions from checkpoint A
    print(f"\nLoading checkpoint A: {args.ckpt_a}")
    model_a, step_a = load_model(args.ckpt_a, device)
    print(f"  step={step_a}")
    print(f"  Generating {len(prompts)} completions...")
    t0 = time.time()
    completions_a = []
    for i, p in enumerate(prompts):
        comp = generate_greedy(model_a, sp, p["prompt"], args.max_tokens, device)
        completions_a.append(comp)
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    del model_a
    torch.cuda.empty_cache()

    # Generate completions from checkpoint B
    print(f"\nLoading checkpoint B: {args.ckpt_b}")
    model_b, step_b = load_model(args.ckpt_b, device)
    print(f"  step={step_b}")
    print(f"  Generating {len(prompts)} completions...")
    t0 = time.time()
    completions_b = []
    for i, p in enumerate(prompts):
        comp = generate_greedy(model_b, sp, p["prompt"], args.max_tokens, device)
        completions_b.append(comp)
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    del model_b
    torch.cuda.empty_cache()

    # Randomly assign A/B per prompt
    rng = random.Random(args.blind_seed)
    # True = ckpt_a is shown as "A"; False = ckpt_a is shown as "B"
    assignments = [rng.random() < 0.5 for _ in prompts]

    # Build blinded pairs and key
    blinded_pairs = []
    key = []
    for i, p in enumerate(prompts):
        if assignments[i]:
            shown_a = completions_a[i]
            shown_b = completions_b[i]
            a_is = step_a
            b_is = step_b
        else:
            shown_a = completions_b[i]
            shown_b = completions_a[i]
            a_is = step_b
            b_is = step_a
        blinded_pairs.append({
            "n": i + 1,
            "prompt": p["prompt"],
            "completion_A": shown_a,
            "completion_B": shown_b,
        })
        key.append({
            "n": i + 1,
            "A_is_step": a_is,
            "B_is_step": b_is,
            "gold": p["gold"],
        })

    # Write key (secret)
    key_data = {
        "ckpt_a_file": args.ckpt_a,
        "ckpt_b_file": args.ckpt_b,
        "ckpt_a_step": step_a,
        "ckpt_b_step": step_b,
        "blind_seed": args.blind_seed,
        "max_tokens": args.max_tokens,
        "assignments": key,
    }
    key_path = out_dir / "key.json"
    key_path.write_text(json.dumps(key_data, indent=2, ensure_ascii=False))
    print(f"\n  ✓ Key written to {key_path} (DO NOT show the expert)")

    # Write blinded presentation
    md = [
        "# Blinded A/B Evaluation — Basque Autocomplete",
        "",
        f"**{len(prompts)} prompts** from held-out Basque text (Wikipedia, never seen in training).",
        f"Each prompt has two completions: **A** and **B**. One is from an earlier checkpoint,",
        f"one from a later checkpoint — randomly assigned per prompt.",
        "",
        "## Instructions for the expert",
        "",
        "For each pair, judge which completion is **better Basque** — more grammatical,",
        "more natural, more useful as an autocomplete suggestion.",
        "",
        "- **A** = completion A is better",
        "- **B** = completion B is better",
        "- **T** = tie (both equally good, or both equally bad)",
        "",
        "Judge on **Basque quality**, NOT on whether it matches the original text.",
        "Both completions may differ from the original but still be valid Basque.",
        "If both are garbage, mark **T**.",
        "",
        "Record your judgments as a list, e.g.: `1:A 2:B 3:T 4:A ...`",
        "",
        "---",
        "",
    ]
    for pair in blinded_pairs:
        md += [
            f"### {pair['n']}",
            f"**Prompt:** `{pair['prompt']}`",
            f"**A:** `{pair['completion_A'] or '(empty)'}`",
            f"**B:** `{pair['completion_B'] or '(empty)'}`",
            f"**Your judgment:** ___",
            "",
        ]
    blinded_path = out_dir / "blinded.md"
    blinded_path.write_text("\n".join(md))
    print(f"  ✓ Blinded presentation written to {blinded_path}")

    # Also write a JSON with just the blinded data (for programmatic use)
    blinded_json = {
        "n_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "pairs": blinded_pairs,
    }
    (out_dir / "blinded.json").write_text(json.dumps(blinded_json, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"  DONE — {len(prompts)} blinded pairs ready.")
    print(f"  Expert reads: {blinded_path}")
    print(f"  Key (secret): {key_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
