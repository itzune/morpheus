#!/usr/bin/env python3
"""
Run next-word CSR evaluation (demo-faithful keyboard simulation) across
multiple checkpoints on the GPU server.

Uses the PyTorch-native evaluate_next_word_csr() from src/eval_utils.py,
which faithfully replicates the deployed keyboard algorithm (retokenization
fallback, sticky merge, top-3 display, acceptance semantics).

This is the GPU equivalent of scripts/simulate_typing.py — no llama.cpp,
no deadlock risk, runs directly on PyTorch model.

Usage:
    python3 scripts/eval_next_word_csr.py \
        --checkpoints checkpoints/step_0032000.pt checkpoints/step_0054000.pt checkpoints/step_0074000.pt \
        --targets eval/v3-targets.json

    # Or specify output file
    python3 scripts/eval_next_word_csr.py \
        --checkpoints checkpoints/step_0032000.pt checkpoints/best.pt \
        --output eval/nw_csr_comparison.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval_utils import evaluate_next_word_csr, bootstrap_mean_ci

SEQ_LEN = 1024


def _pad_vocab(vocab_size, multiple):
    return ((vocab_size + multiple - 1) // multiple) * multiple


def build_model_from_checkpoint(checkpoint_path, device):
    """Load Mamba model from training checkpoint."""
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
    valid_loss = ckpt.get("valid_loss", None)
    del ckpt
    return model, raw_cfg, step, valid_loss


def load_csr_tests(targets_path):
    """Load CSR test sentences from targets.json."""
    with open(targets_path) as f:
        targets = json.load(f)
    for strategy in targets.get("strategies", []):
        if strategy.get("name") == "csr":
            return strategy.get("tests", [])
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Run next-word CSR eval across checkpoints (GPU, PyTorch)"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint .pt files to evaluate")
    parser.add_argument("--targets", default="eval/targets.json",
                        help="Eval targets JSON file")
    parser.add_argument("--sp-model", default="tokenizer/basque_unigram_4000.model",
                        help="SentencePiece model path")
    parser.add_argument("--output", default=None,
                        help="Save comparison results to JSON file")
    parser.add_argument("--device", default="cuda",
                        help="Device to run on (cuda or cpu)")
    args = parser.parse_args()

    # Load tokenizer and test data
    sp = spm.SentencePieceProcessor()
    sp.Load(args.sp_model)
    print(f"Loaded tokenizer: {args.sp_model} (vocab={sp.get_piece_size()})")

    csr_tests = load_csr_tests(args.targets)
    print(f"Loaded {len(csr_tests)} CSR test sentences from {args.targets}")
    if not csr_tests:
        print("ERROR: No CSR tests found!", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print()

    all_results = []

    for ckpt_path in args.checkpoints:
        name = Path(ckpt_path).name
        print(f"{'='*60}")
        print(f"  Loading {name}...")
        t0 = time.time()
        model, raw_cfg, step, valid_loss = build_model_from_checkpoint(ckpt_path, device)
        load_time = time.time() - t0
        print(f"  Step: {step}  Valid loss: {valid_loss}  Load: {load_time:.1f}s")

        # Run next-word CSR evaluation
        print(f"  Running next-word CSR simulation ({len(csr_tests)} sentences)...")
        t0 = time.time()
        nw_results = evaluate_next_word_csr(model, sp, csr_tests, device)
        eval_time = time.time() - t0

        # Aggregate
        total_chars = sum(r["total_chars"] for r in nw_results)
        total_ks = sum(r["keystrokes"] for r in nw_results)
        total_words = sum(r["n_words"] for r in nw_results)
        total_accepts = sum(r["taps"] for r in nw_results)
        total_top1 = sum(r["top1_accuracy"] * r["n_words"] for r in nw_results)
        total_top3 = sum(r["top3_accuracy"] * r["n_words"] for r in nw_results)
        total_top5 = sum(r["top5_accuracy"] * r["n_words"] for r in nw_results)
        all_probs = [p for r in nw_results for w, m, p in r["completed_words"] if p > 0]
        prefix_lens = [e.get("prefix_len", 0) for r in nw_results
                       for e in r["events"] if e["type"] == "accept"]
        macro_csrs = [r["csr"] for r in nw_results]
        csr_point, csr_lo, csr_hi = bootstrap_mean_ci(macro_csrs)

        # Count completions vs next-words vs manual
        completions = sum(1 for r in nw_results for w, m, p in r["completed_words"] if m == "completion")
        next_words = sum(1 for r in nw_results for w, m, p in r["completed_words"] if m == "next_word")
        manuals = sum(1 for r in nw_results for w, m, p in r["completed_words"] if m == "manual")

        summary = {
            "checkpoint": name,
            "step": step,
            "valid_loss": valid_loss,
            "valid_ppl": math.exp(valid_loss) if valid_loss else None,
            "nw_csr": round((total_chars - total_ks) / total_chars, 4) if total_chars > 0 else 0.0,
            "nw_csr_macro": round(csr_point, 4),
            "nw_csr_ci_lower": round(csr_lo, 4),
            "nw_csr_ci_upper": round(csr_hi, 4),
            "acceptance_rate": round(total_accepts / total_words, 4) if total_words > 0 else 0.0,
            "top1_accuracy": round(total_top1 / total_words, 4) if total_words > 0 else 0.0,
            "top3_accuracy": round(total_top3 / total_words, 4) if total_words > 0 else 0.0,
            "top5_accuracy": round(total_top5 / total_words, 4) if total_words > 0 else 0.0,
            "avg_prefix_before_accept": round(sum(prefix_lens) / len(prefix_lens), 2) if prefix_lens else 0.0,
            "avg_confidence": round(sum(all_probs) / len(all_probs), 4) if all_probs else 0.0,
            "n_tests": len(nw_results),
            "n_words": total_words,
            "n_completions": completions,
            "n_next_words": next_words,
            "n_manual": manuals,
            "eval_time_s": round(eval_time, 1),
        }

        print(f"  Completed in {eval_time:.1f}s")
        print(f"  ┌─ Next-Word CSR:  {summary['nw_csr']:.3f} (macro {summary['nw_csr_macro']:.3f}, CI [{summary['nw_csr_ci_lower']:.3f}, {summary['nw_csr_ci_upper']:.3f}])")
        print(f"  ├─ Top-1 Acc:      {summary['top1_accuracy']:.3f}")
        print(f"  ├─ Top-3 Acc:      {summary['top3_accuracy']:.3f}  (= acceptance)")
        print(f"  ├─ Top-5 Acc:      {summary['top5_accuracy']:.3f}")
        print(f"  ├─ Acceptance:     {summary['acceptance_rate']:.3f} ({total_accepts}/{total_words})")
        print(f"  ├─ Avg Prefix:     {summary['avg_prefix_before_accept']:.1f} chars")
        print(f"  ├─ Avg Confidence: {summary['avg_confidence']:.3f}")
        print(f"  └─ Breakdown:      {completions} completions, {next_words} next-words, {manuals} manual")
        print()

        all_results.append({"summary": summary, "details": nw_results})

        # Free model from GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Comparison table ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Next-Word CSR across checkpoints")
    print(f"{'='*70}")
    print(f"\n  {'Checkpoint':<25s} {'Step':>6s} {'PPL':>6s} {'NW-CSR':>8s} {'Top1':>8s} {'Top3':>8s} {'Top5':>8s} {'Accept':>8s} {'Prefix':>7s} {'Conf':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for r in all_results:
        s = r["summary"]
        ppl = f"{s['valid_ppl']:.2f}" if s['valid_ppl'] else "N/A"
        print(f"  {s['checkpoint']:<25s} {str(s['step']):>6s} {ppl:>6s} "
              f"{s['nw_csr']:>8.3f} {s['top1_accuracy']:>8.3f} {s['top3_accuracy']:>8.3f} {s['top5_accuracy']:>8.3f} {s['acceptance_rate']:>8.3f} "
              f"{s['avg_prefix_before_accept']:>7.1f} {s['avg_confidence']:>6.3f}")

    print(f"\n  CI ranges:")
    for r in all_results:
        s = r["summary"]
        print(f"    {s['checkpoint']:<25s} [{s['nw_csr_ci_lower']:.3f}, {s['nw_csr_ci_upper']:.3f}]")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Full results saved to {out_path}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
