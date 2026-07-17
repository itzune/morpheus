#!/usr/bin/env python3
"""
FIM (Fill-in-the-Middle) evaluation for morpheus Phase 6.

Measures the model's ability to infill text at arbitrary cursor positions:
  1. Exact-match: does the generated <MID> reconstruct the original span?
  2. Character accuracy: Levenshtein-based similarity.
  3. Keystrokes-saved-in-the-middle: chars saved by accepting the completion.

For each evaluation sentence:
  - Pick a random char-level span (the "middle")
  - Build FIM prompt: <PRE>{prefix}<SUF>{suffix}<MID>
  - SP-encode (FIM tokenizer, atomic special tokens)
  - Greedy-decode until <EOT> or max_tokens
  - Compare generated middle to original

Usage:
    # Evaluate a Phase 6 checkpoint
    python3 scripts/fim_eval.py \
        --checkpoint checkpoints/phase6/step_00001000.pt \
        --sp-model tokenizer/basque_unigram_fim.model \
        --valid-file data/valid/wiki_valid.txt \
        --n-examples 200

    # Compare AR baseline (no FIM training) — should be poor
    python3 scripts/fim_eval.py \
        --checkpoint checkpoints/step_0074000_fim.pt \
        --sp-model tokenizer/basque_unigram_fim.model \
        --valid-file data/valid/wiki_valid.txt \
        --n-examples 200
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

# Reuse the same split logic as build_fim.py
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from build_fim import make_fim_split, build_fim_string, find_word_boundaries

# ── Constants ──
FIM_TOKENS = {"<PRE>": 4000, "<SUF>": 4001, "<MID>": 4002, "<EOT>": 4003}
EOS_ID = 2  # </s>
MAX_GEN_TOKENS = 64  # generous; most middle spans are < 30 tokens


# ── Model ──

def build_model(vocab_size: int, device) -> MambaLMHeadModel:
    config = MambaConfig(
        d_model=768,
        n_layer=24,
        vocab_size=vocab_size,
        ssm_cfg={
            "layer": "Mamba2",
            "d_state": 64,
            "d_conv": 4,
            "expand": 2,
            "headdim": 64,
            "chunk_size": 256,
        },
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
    )
    return MambaLMHeadModel(config, device=device, dtype=torch.bfloat16)


def load_checkpoint(checkpoint_path: str, device) -> tuple:
    """Load checkpoint, return (model, vocab_size)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab_size = cfg.get("padded_vocab_size", cfg.get("vocab_size", 4016))
    model = build_model(vocab_size, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    del ckpt
    torch.cuda.empty_cache()
    return model, vocab_size


# ── Generation ──

@torch.no_grad()
def generate_middle(model, sp, prompt_ids: list, device,
                    max_tokens: int = MAX_GEN_TOKENS,
                    temperature: float = 0.0, top_k: int = 0) -> list:
    """Generate tokens after <MID> until <EOT> or </s> or max_tokens.

    Args:
        prompt_ids: Token IDs of <PRE>...<SUF>...<MID> (the FIM prompt).
        temperature: 0 = greedy, >0 = sampled.
        top_k: If >0, use top-k sampling (only when temperature > 0).

    Returns: list of generated token IDs (excluding the prompt, excluding stop tokens).
    """
    ids = list(prompt_ids)
    generated = []
    stop_ids = {FIM_TOKENS["<EOT>"], EOS_ID, 0}  # <EOT>, </s>, <unk>

    for _ in range(max_tokens):
        ctx = torch.tensor([ids], device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ctx).logits[0, -1, :].float()

        if temperature == 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

        if next_id in stop_ids:
            break

        ids.append(next_id)
        generated.append(next_id)

    return generated


# ── Metrics ──

def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def char_accuracy(generated: str, reference: str) -> float:
    """Character-level accuracy = 1 - edit_distance / max_len."""
    if not reference:
        return 1.0 if not generated else 0.0
    dist = levenshtein(generated, reference)
    return max(0.0, 1.0 - dist / max(len(reference), len(generated)))


def keystrokes_saved(generated: str, reference: str) -> int:
    """Chars saved by accepting the completion vs typing the reference.

    = len(reference) - edit_distance(generated, reference)
    Positive = saved keystrokes, negative = completion was worse than typing.
    """
    return len(reference) - levenshtein(generated, reference)


# ── Evaluation ──

def load_valid_lines(valid_file: str, min_line: int = 20, max_lines: int = 0) -> list:
    """Load validation lines, filtered by minimum length."""
    lines = []
    with open(valid_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) >= min_line:
                lines.append(line)
                if max_lines > 0 and len(lines) >= max_lines:
                    break
    return lines


def evaluate_fim(model, sp, lines: list, device, n_examples: int = 200,
                 boundary_bias: float = 0.2, min_mid: int = 3, min_line: int = 20,
                 seed: int = 42):
    """Run FIM evaluation on n_examples sentences.

    Returns dict with metrics and per-example details.
    """
    rng = random.Random(seed)
    examples = []

    exact_matches = 0
    char_accs = []
    keystrokes = []
    gen_lengths = []
    ref_lengths = []

    # Pick n_examples lines (deterministic)
    indices = list(range(len(lines)))
    rng.shuffle(indices)
    indices = indices[:n_examples]

    for i, idx in enumerate(indices):
        text = lines[idx]
        # Deterministic split for this example
        ex_rng = random.Random(seed + idx)
        split = make_fim_split(text, ex_rng, boundary_bias, min_mid, min_line)
        if split is None:
            continue

        prefix, middle, suffix = split
        mode = "PSM" if ex_rng.random() < 0.5 else "SPM"
        fim_prompt = build_fim_string(prefix, middle, suffix, mode)
        # For eval, we only send the prompt (without the middle and <EOT>)
        # The model should generate the middle and then <EOT>
        if mode == "PSM":
            prompt_str = f"<PRE>{prefix}<SUF>{suffix}<MID>"
        else:
            prompt_str = f"<SUF>{suffix}<PRE>{prefix}<MID>"

        prompt_ids = sp.encode(prompt_str, out_type=int)
        gen_ids = generate_middle(model, sp, prompt_ids, device)
        gen_text = sp.decode(gen_ids) if gen_ids else ""

        # Strip leading space (SentencePiece often adds ▁ at start of generation)
        gen_text = gen_text.strip()
        ref_text = middle.strip()

        em = gen_text == ref_text
        ca = char_accuracy(gen_text, ref_text)
        ks = keystrokes_saved(gen_text, ref_text)

        if em:
            exact_matches += 1
        char_accs.append(ca)
        keystrokes.append(ks)
        gen_lengths.append(len(gen_text))
        ref_lengths.append(len(ref_text))

        examples.append({
            "text": text[:120],
            "mode": mode,
            "prefix": prefix[:60],
            "middle": middle[:60],
            "suffix": suffix[:60],
            "generated": gen_text[:60],
            "exact_match": em,
            "char_accuracy": ca,
            "keystrokes_saved": ks,
        })

        if (i + 1) % 50 == 0:
            em_rate = exact_matches / len(char_accs)
            avg_ca = sum(char_accs) / len(char_accs)
            avg_ks = sum(keystrokes) / len(keystrokes)
            print(f"  [{i+1}/{n_examples}] EM={em_rate:.1%}  char_acc={avg_ca:.1%}  "
                  f"ks_saved={avg_ks:.1f}")

    n = len(char_accs)
    metrics = {
        "n_examples": n,
        "exact_match_rate": exact_matches / n if n > 0 else 0,
        "avg_char_accuracy": sum(char_accs) / n if n > 0 else 0,
        "avg_keystrokes_saved": sum(keystrokes) / n if n > 0 else 0,
        "total_keystrokes_saved": sum(keystrokes),
        "total_reference_chars": sum(ref_lengths),
        "keystrokes_saved_pct": (sum(keystrokes) / sum(ref_lengths) * 100) if ref_lengths else 0,
        "avg_gen_length": sum(gen_lengths) / n if n > 0 else 0,
        "avg_ref_length": sum(ref_lengths) / n if n > 0 else 0,
    }

    return metrics, examples


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="FIM evaluation for morpheus Phase 6")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--sp-model", default="tokenizer/basque_unigram_fim.model",
                        help="FIM SentencePiece .model")
    parser.add_argument("--valid-file", default="data/valid/wiki_valid.txt",
                        help="Validation text file")
    parser.add_argument("--n-examples", type=int, default=200,
                        help="Number of FIM examples to evaluate")
    parser.add_argument("--boundary-bias", type=float, default=0.2,
                        help="Fraction of splits at linguistic boundaries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show-examples", type=int, default=10,
                        help="Show N example details")
    parser.add_argument("--output", default="", help="Save JSON results to file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.sp_model}")
    sp = spm.SentencePieceProcessor()
    sp.Load(args.sp_model)
    print(f"  Vocab size: {sp.get_piece_size()}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, vocab_size = load_checkpoint(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Vocab: {vocab_size}, params: {n_params/1e6:.0f}M")

    # Load validation lines
    print(f"\nLoading validation: {args.valid_file}")
    lines = load_valid_lines(args.valid_file, max_lines=args.n_examples * 5)
    print(f"  {len(lines):,} lines (≥20 chars)")

    # Run eval
    print(f"\nEvaluating FIM on {args.n_examples} examples...")
    print(f"  Boundary bias: {args.boundary_bias} ({args.boundary_bias*100:.0f}% linguistic)")
    print()

    metrics, examples = evaluate_fim(
        model, sp, lines, device,
        n_examples=args.n_examples,
        boundary_bias=args.boundary_bias,
        seed=args.seed,
    )

    # Print results
    print("\n" + "=" * 60)
    print("  FIM Evaluation Results")
    print("=" * 60)
    print(f"  Examples:              {metrics['n_examples']}")
    print(f"  Exact-match rate:      {metrics['exact_match_rate']:.1%}")
    print(f"  Avg char accuracy:     {metrics['avg_char_accuracy']:.1%}")
    print(f"  Avg keystrokes saved:  {metrics['avg_keystrokes_saved']:.1f} chars")
    print(f"  Total keystrokes saved:{metrics['total_keystrokes_saved']:,} chars")
    print(f"  Total reference chars: {metrics['total_reference_chars']:,} chars")
    print(f"  Keystrokes saved:      {metrics['keystrokes_saved_pct']:.1f}%")
    print(f"  Avg gen length:        {metrics['avg_gen_length']:.1f} chars")
    print(f"  Avg ref length:        {metrics['avg_ref_length']:.1f} chars")
    print("=" * 60)

    # Show examples
    if args.show_examples > 0:
        print(f"\n--- Top {args.show_examples} examples ---")
        # Show a mix of good and bad
        sorted_ex = sorted(examples, key=lambda e: e["char_accuracy"], reverse=True)
        step = max(1, len(sorted_ex) // args.show_examples)
        shown = 0
        for ex in sorted_ex[::step]:
            print(f"\n  [{ex['mode']}] char_acc={ex['char_accuracy']:.0%}  "
                  f"EM={'✓' if ex['exact_match'] else '✗'}")
            print(f"  Text:    {ex['text'][:100]}")
            print(f"  Prefix:  {repr(ex['prefix'][:50])}")
            print(f"  Middle:  {repr(ex['middle'][:50])}")
            print(f"  Suffix:  {repr(ex['suffix'][:50])}")
            print(f"  Gen:     {repr(ex['generated'][:50])}")
            print(f"  KS:      {ex['keystrokes_saved']} chars")
            shown += 1
            if shown >= args.show_examples:
                break

    # Save JSON
    if args.output:
        import json
        results = {"metrics": metrics, "examples": examples}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
