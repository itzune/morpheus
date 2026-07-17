#!/usr/bin/env python3
"""
Smoke test: confirm embedding resize (4000 → 4016) doesn't break AR quality.

Compares AR perplexity on the validation set between:
  1. Original checkpoint  (step_0074000.pt,     vocab 4000)
  2. Resized checkpoint   (step_0074000_fim.pt, vocab 4016)

The new 16 rows (4 FIM mean-init + 12 zero padding) should steal negligible
softmax mass, so perplexity must be essentially unchanged. A large jump would
indicate a bug in the resize.

Also verifies the FIM tokenizer can encode/decode the new special tokens.

Usage:
    python3 scripts/pipeline/smoke_test_fim.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


CKPT_ORIG = "checkpoints/step_0074000.pt"
CKPT_FIM = "checkpoints/step_0074000_fim.pt"
VALID_DATA = "data/valid_tokens_4k.npy"
SP_ORIG = "tokenizer/basque_unigram_4000.model"
SP_FIM = "tokenizer/basque_unigram_fim.model"

D_MODEL = 768
N_LAYER = 24
SEQ_LEN = 1024
N_BATCHES = 10  # quick: 10 batches × batch_size sequences


def build_model(vocab_size: int, device) -> MambaLMHeadModel:
    config = MambaConfig(
        d_model=D_MODEL,
        n_layer=N_LAYER,
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


@torch.no_grad()
def compute_loss(model, data, device, n_batches=N_BATCHES, batch_size=4):
    """Compute average AR cross-entropy loss on n_batches of validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_seq = (len(data) - 1) // SEQ_LEN

    for i in range(n_batches):
        # Deterministic sequences (not random) for reproducible comparison
        start = (i * 7) % n_seq * SEQ_LEN  # stride by 7 to spread across data
        end = start + SEQ_LEN + 1
        chunk = data[start:end].astype(np.int64)

        x = torch.from_numpy(chunk[:-1]).unsqueeze(0).expand(batch_size, -1).to(device)
        y = torch.from_numpy(chunk[1:]).unsqueeze(0).expand(batch_size, -1).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(x)
            loss = F.cross_entropy(
                output.logits.view(-1, output.logits.size(-1)),
                y.view(-1),
                ignore_index=0,
            )
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    return total_loss / total_tokens


def test_tokenizer():
    """Verify the FIM tokenizer: special tokens, atomicity, regression."""
    print("\n" + "=" * 60)
    print("  Part 1: FIM Tokenizer Verification")
    print("=" * 60)

    sp = spm.SentencePieceProcessor()
    sp.Load(SP_FIM)

    checks = []

    # Vocab size
    vs = sp.get_piece_size()
    checks.append(("vocab size = 4004", vs == 4004, f"got {vs}"))

    # FIM token IDs
    for tok, expected_id in [("<PRE>", 4000), ("<SUF>", 4001), ("<MID>", 4002), ("<EOT>", 4003)]:
        tid = sp.piece_to_id(tok)
        checks.append((f"{tok} → id {expected_id}", tid == expected_id, f"got {tid}"))

    # Regression: normal text tokenizes same as original
    sp_old = spm.SentencePieceProcessor()
    sp_old.Load(SP_ORIG)
    sample = "Euskal Herriko hizkuntza ofiziala da euskara"
    same = sp.EncodeAsIds(sample) == sp_old.EncodeAsIds(sample)
    checks.append(("normal text unchanged", same))

    # Atomicity
    test = "Kaixo<PRE>mundua<SUF>"
    ids = sp.EncodeAsIds(test)
    checks.append(("<PRE> atomic", 4000 in ids))
    checks.append(("<SUF> atomic", 4001 in ids))

    all_pass = True
    for name, passed, detail in [(c[0], c[1], c[2] if len(c) > 2 else "") for c in checks]:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}" + (f" — {detail}" if detail and not passed else ""))
        if not passed:
            all_pass = False

    return all_pass


def test_ar_perplexity():
    """Compare AR perplexity: original (4000) vs resized (4016)."""
    print("\n" + "=" * 60)
    print("  Part 2: AR Perplexity — Original vs Resized")
    print("=" * 60)

    device = torch.device("cuda")
    print(f"  Device: {torch.cuda.get_device_name(0)}")

    # Load validation data
    data = np.load(VALID_DATA, mmap_mode='r')
    print(f"  Validation data: {len(data):,} tokens")

    # ── Original checkpoint ──
    print(f"\n  Loading original checkpoint ({CKPT_ORIG})...")
    ckpt_orig = torch.load(CKPT_ORIG, map_location="cpu", weights_only=False)
    model_orig = build_model(4000, device)
    model_orig.load_state_dict(ckpt_orig["model"])
    del ckpt_orig
    torch.cuda.empty_cache()

    loss_orig = compute_loss(model_orig, data, device)
    ppl_orig = math.exp(min(loss_orig, 20))
    print(f"  Original:  loss={loss_orig:.4f}  ppl={ppl_orig:.2f}")

    del model_orig
    torch.cuda.empty_cache()

    # ── Resized checkpoint ──
    print(f"\n  Loading resized checkpoint ({CKPT_FIM})...")
    ckpt_fim = torch.load(CKPT_FIM, map_location="cpu", weights_only=False)
    model_fim = build_model(4016, device)
    model_fim.load_state_dict(ckpt_fim["model"])
    del ckpt_fim
    torch.cuda.empty_cache()

    loss_fim = compute_loss(model_fim, data, device)
    ppl_fim = math.exp(min(loss_fim, 20))
    print(f"  Resized:   loss={loss_fim:.4f}  ppl={ppl_fim:.2f}")

    del model_fim
    torch.cuda.empty_cache()

    # ── Comparison ──
    delta_loss = abs(loss_fim - loss_orig)
    delta_ppl = abs(ppl_fim - ppl_orig)
    pct = (delta_loss / loss_orig) * 100

    print(f"\n  Δ loss: {delta_loss:.4f} ({pct:.2f}%)")
    print(f"  Δ ppl:  {delta_ppl:.2f}")

    # Threshold: <2% change is "negligible perturbation" (within batch noise)
    passed = pct < 2.0
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: perplexity change {'< 2% (negligible)' if passed else '>= 2% (investigate)'}")

    return passed, loss_orig, loss_fim


def main():
    tok_ok = test_tokenizer()
    ar_ok, loss_orig, loss_fim = test_ar_perplexity()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Tokenizer:     {'✓ PASS' if tok_ok else '✗ FAIL'}")
    print(f"  AR perplexity: {'✓ PASS' if ar_ok else '✗ FAIL'}  "
          f"(orig={loss_orig:.4f}, fim={loss_fim:.4f})")
    print("=" * 60)

    if tok_ok and ar_ok:
        print("\n  ✓ P2 complete: FIM tokens + embedding resize verified.")
        print("    Next: P3 (Basque FIM dataset — char-level PSM/SPM transform)")
        sys.exit(0)
    else:
        print("\n  ✗ Some checks failed. Investigate before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
