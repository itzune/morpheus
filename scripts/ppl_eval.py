#!/usr/bin/env python3
"""
PPL evaluation on REAL held-out Basque prose (Wikipedia + Berria).

This is the D0 diagnostic from docs/eval-reform-proposal.md:
the smoothest, lowest-variance signal with no exact-match artifact.
If 54K < 32K in PPL AND the expert says 54K is better, that is
convergent evidence that CSR-on-30 was saturated, not the model.

Matches training semantics EXACTLY:
  - Line-by-line tokenization with </s> (id=2) separators between lines
    (same as scripts/pipeline/pretokenize.py)
  - 1024-token windows, shifted by 1 for next-token prediction
    (same as src/dataset.py::MemmapTokenDataset)
  - ignore_index=0 (<unk>); </s> (id=2) IS included in loss
    (same as train.py::evaluate)
  - bfloat16 autocast, NO BOS (model trained without BOS)
  - Token-weighted mean cross-entropy (more accurate than train.py's
    per-batch average; both reported)

Reports per-file and aggregate PPL for diagnostic granularity.

Modes:
  --corpus-dir  : PPL on raw text files (line-by-line tokenization + </s>)
  --valid-data  : PPL on pre-tokenized .npy (held-out validation set)
  Both can be used together for a complete picture.

Usage:
    python3 ppl_eval.py --checkpoint checkpoints/step_0032000.pt \\
        --corpus-dir eval/real_corpus --valid-data data/valid_tokens_4k.npy
    python3 ppl_eval.py --checkpoint checkpoints/step_0054000.pt \\
        --corpus-dir eval/real_corpus --valid-data data/valid_tokens_4k.npy
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


SEQ_LEN = 1024  # must match config/small.yaml


def _pad_vocab(vocab_size, multiple):
    return ((vocab_size + multiple - 1) // multiple) * multiple


def build_model_from_checkpoint(checkpoint_path, device):
    """Load Mamba model from training checkpoint (same as eval.py)."""
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
    return model, raw_cfg, step


def tokenize_corpus(corpus_dir, sp):
    """Tokenize real-corpus files into per-file token streams.

    Matches pretokenize.py: each line → tokens + </s> (eos_id).
    Returns dict: filename → np.array(token_ids).
    """
    eos_id = sp.eos_id()  # </s> = id=2
    files = {}
    corpus_dir = Path(corpus_dir)
    for txt_file in sorted(corpus_dir.glob("*.txt")):
        tokens = []
        with open(txt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = sp.encode(line, out_type=int)
                tokens.extend(ids)
                tokens.append(eos_id)  # </s> separator (same as training)
        if tokens:
            files[txt_file.name] = np.array(tokens, dtype=np.int64)
    return files


@torch.no_grad()
def compute_file_ppl(model, token_ids, device, seq_len=SEQ_LEN):
    """Compute token-weighted CE loss and PPL for one file's token stream.

    Chunks into seq_len windows (same as MemmapTokenDataset).
    Returns (total_ce_sum, n_tokens, n_windows).
    """
    total_ce_sum = 0.0
    n_tokens = 0
    n_windows = 0
    n_sequences = (len(token_ids) - 1) // seq_len
    # Handle remainder: if there are leftover tokens beyond the last full
    # window, process them as a shorter window (don't drop data).
    has_remainder = (len(token_ids) - 1) % seq_len > 0
    total_windows = n_sequences + (1 if has_remainder else 0)

    for i in range(total_windows):
        start = i * seq_len
        end = min(start + seq_len + 1, len(token_ids))  # +1 for target shift
        if end - start < 2:  # need at least 2 tokens for x,y
            break
        chunk = token_ids[start:end]
        x = torch.from_numpy(np.ascontiguousarray(chunk[:-1])).unsqueeze(0).to(device)
        y = torch.from_numpy(np.ascontiguousarray(chunk[1:])).unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(x)
            # Sum CE (not mean) so we can weight by token count
            ce_sum = F.cross_entropy(
                output.logits.view(-1, output.logits.size(-1)),
                y.view(-1),
                ignore_index=0,  # <unk> only; </s> included (same as train.py)
                reduction="sum",
            )

        # Count non-ignored tokens
        mask = y.view(-1) != 0
        count = mask.sum().item()
        total_ce_sum += ce_sum.item()
        n_tokens += count
        n_windows += 1

    return total_ce_sum, n_tokens, n_windows


def main():
    parser = argparse.ArgumentParser(description="PPL eval on real corpus")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/basque_unigram_4000.model")
    parser.add_argument("--corpus-dir", default="eval/real_corpus",
                        help="Raw text files for PPL (line-by-line + </s>)")
    parser.add_argument("--valid-data", default=None,
                        help="Pre-tokenized .npy for held-out PPL (e.g. data/valid_tokens_4k.npy)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} (vocab={sp.vocab_size()}, eos_id={sp.eos_id()})")

    # Load model (load once, reuse for all modes)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    t0 = time.time()
    model, raw_cfg, step = build_model_from_checkpoint(args.checkpoint, device)
    print(f"  step={step}, d_model={raw_cfg['d_model']}, n_layer={raw_cfg['n_layer']}, "
          f"vocab={raw_cfg.get('vocab_size', '?')} ({time.time()-t0:.1f}s)")

    results = {"step": step}

    # ── Mode 1: PPL on pre-tokenized held-out validation set ──
    if args.valid_data:
        print(f"\n{'='*80}")
        print(f"  Perplexity on HELD-OUT Validation Set")
        print(f"  Checkpoint: {Path(args.checkpoint).name} (step {step})")
        print(f"  seq_len={SEQ_LEN}, ignore_index=0, </s> in loss, no BOS, bfloat16")
        print(f"{'='*80}")
        valid_ids = np.load(args.valid_data, mmap_mode='r')
        valid_arr = np.array(valid_ids, dtype=np.int64)
        print(f"  Validation tokens: {len(valid_arr):,}")
        ce_sum, n_tok, n_win = compute_file_ppl(model, valid_arr, device)
        vloss = ce_sum / n_tok if n_tok > 0 else float("inf")
        vppl = math.exp(min(vloss, 20))
        print(f"  Windows: {n_win}")
        print(f"  Mean cross-entropy: {vloss:.4f}")
        print(f"  Perplexity:         {vppl:.2f}")
        results["valid_loss"] = round(vloss, 4)
        results["valid_ppl"] = round(vppl, 2)
        results["valid_tokens"] = n_tok
        results["valid_windows"] = n_win
        del valid_arr

    # ── Mode 2: PPL on real-corpus raw text ──
    if args.corpus_dir:
        print(f"\nTokenizing real corpus from {args.corpus_dir}/ ...")
        t0 = time.time()
        files = tokenize_corpus(args.corpus_dir, sp)
        total_tokens = sum(len(v) for v in files.values())
        print(f"  {len(files)} files, {total_tokens:,} tokens ({time.time()-t0:.1f}s)")

        print(f"\n{'='*80}")
        print(f"  Perplexity on Real Corpus (Wikipedia + Berria)")
        print(f"  Checkpoint: {Path(args.checkpoint).name} (step {step})")
        print(f"  ⚠ CONTAMINATED: these articles appear in training sources.")
        print(f"  Absolute PPL is optimistic; relative 32K-vs-54K is still valid.")
        print(f"  seq_len={SEQ_LEN}, ignore_index=0, </s> in loss, no BOS, bfloat16")
        print(f"{'='*80}")
        print(f"{'File':<32s} {'Tokens':>8s} {'Windows':>8s} {'Loss':>8s} {'PPL':>10s}")
        print("-" * 80)

        grand_ce = 0.0
        grand_tokens = 0
        grand_windows = 0
        file_results = []

        for fname, token_ids in sorted(files.items()):
            ce_sum, n_tok, n_win = compute_file_ppl(model, token_ids, device)
            loss = ce_sum / n_tok if n_tok > 0 else float("inf")
            ppl = math.exp(min(loss, 20))
            grand_ce += ce_sum
            grand_tokens += n_tok
            grand_windows += n_win
            file_results.append({
                "file": fname, "tokens": n_tok, "windows": n_win,
                "loss": round(loss, 4), "ppl": round(ppl, 2),
            })
            print(f"{fname:<32s} {n_tok:>8,} {n_win:>8d} {loss:>8.4f} {ppl:>10.2f}")

        overall_loss = grand_ce / grand_tokens if grand_tokens > 0 else float("inf")
        overall_ppl = math.exp(min(overall_loss, 20))
        print("-" * 80)
        print(f"{'OVERALL (token-weighted)':<32s} {grand_tokens:>8,} {grand_windows:>8d} "
              f"{overall_loss:>8.4f} {overall_ppl:>10.2f}")
        print(f"\n  Mean cross-entropy: {overall_loss:.4f}")
        print(f"  Perplexity:         {overall_ppl:.2f}")
        results["corpus_loss"] = round(overall_loss, 4)
        results["corpus_ppl"] = round(overall_ppl, 2)
        results["corpus_tokens"] = grand_tokens
        results["corpus_files"] = file_results

    print(f"\n{'='*80}")
    print(f"  DONE — step {step}")
    if "valid_ppl" in results:
        print(f"  Held-out valid PPL:  {results['valid_ppl']:.2f}  (loss={results['valid_loss']:.4f})")
    if "corpus_ppl" in results:
        print(f"  Real corpus PPL:     {results['corpus_ppl']:.2f}  (loss={results['corpus_loss']:.4f}) [contaminated]")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    main()
