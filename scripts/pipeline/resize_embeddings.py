#!/usr/bin/env python3
"""
Resize the model embedding table for FIM tokens.

Loads a pretrained checkpoint (vocab 4000) and resizes the token embedding
+ LM head to 4016 (next multiple of 16 above 4004). The 4 new FIM token rows
(IDs 4000–4003) are mean-initialized (neutral starting point, standard HF
practice); the 12 padding rows (4004–4015) are zero-initialized (never used).

Existing token IDs and all other weights are UNCHANGED, so AR perplexity is
only minimally perturbed (the new rows contribute near-zero logits after the
mean is subtracted by the norm, and steal negligible softmax mass).

Embeddings are TIED: backbone.embedding.weight == lm_head.weight (verified
max abs diff = 0.0). Both are resized identically.

Usage:
    python3 scripts/pipeline/resize_embeddings.py \
        --checkpoint checkpoints/step_0074000.pt \
        --output checkpoints/step_0074000_fim.pt \
        --new-vocab 4004 \
        --padded-vocab 4016
"""

import argparse
import copy
from pathlib import Path

import torch


def resize_embedding(weight: torch.Tensor, new_vocab: int, padded_vocab: int) -> torch.Tensor:
    """Resize an embedding matrix [old_vocab, d_model] → [padded_vocab, d_model].

    Rows [0:old_vocab] = original (unchanged).
    Rows [old_vocab:new_vocab] = mean of original rows (FIM token init).
    Rows [new_vocab:padded_vocab] = zeros (padding, never used).
    """
    old_vocab, d_model = weight.shape
    assert new_vocab > old_vocab, f"new_vocab {new_vocab} must be > old_vocab {old_vocab}"
    assert padded_vocab >= new_vocab, f"padded_vocab {padded_vocab} must be >= new_vocab {new_vocab}"

    # Mean of existing embeddings — neutral init (standard HF resize_token_embeddings)
    mean_row = weight.float().mean(dim=0)  # [d_model]

    # Build new matrix
    new_weight = torch.zeros(padded_vocab, d_model, dtype=weight.dtype, device=weight.device)
    new_weight[:old_vocab] = weight                          # original
    new_weight[old_vocab:new_vocab] = mean_row.unsqueeze(0)  # FIM tokens (mean init)
    # new_weight[new_vocab:padded_vocab] stays zero (padding)

    return new_weight


def resize_checkpoint(checkpoint_path: str, output_path: str,
                       new_vocab: int, padded_vocab: int):
    """Resize embedding + lm_head in a checkpoint, preserving all other weights."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    cfg = ckpt["config"]

    old_vocab = cfg["vocab_size"]
    print(f"  step: {ckpt['step']}")
    print(f"  old vocab_size: {old_vocab}")
    print(f"  new vocab: {new_vocab} (FIM tokens: {old_vocab}–{new_vocab - 1})")
    print(f"  padded vocab: {padded_vocab}")

    # Find embedding + lm_head keys
    emb_key = "backbone.embedding.weight"
    head_key = "lm_head.weight"

    if emb_key not in state_dict:
        raise KeyError(f"{emb_key} not found in checkpoint. Keys: {list(state_dict.keys())[:5]}...")

    emb = state_dict[emb_key]
    print(f"\n  {emb_key}: {tuple(emb.shape)} dtype={emb.dtype}")

    has_lm_head = head_key in state_dict
    if has_lm_head:
        head = state_dict[head_key]
        print(f"  {head_key}: {tuple(head.shape)} dtype={head.dtype}")
        # Verify tied
        tied = torch.equal(emb.float(), head.float())
        print(f"  tied (identical)? {'✓' if tied else '✗ (will resize independently)'}")
    else:
        print(f"  {head_key}: not present (truly tied at module level)")

    # Resize
    print(f"\nResizing to [{padded_vocab}, {emb.shape[1]}]...")
    new_emb = resize_embedding(emb, new_vocab, padded_vocab)
    state_dict[emb_key] = new_emb
    print(f"  {emb_key}: {tuple(new_emb.shape)} ✓")

    if has_lm_head:
        new_head = resize_embedding(head, new_vocab, padded_vocab)
        state_dict[head_key] = new_head
        print(f"  {head_key}: {tuple(new_head.shape)} ✓")

    # Verify new rows
    mean_row = emb.float().mean(dim=0)
    print(f"\n  FIM rows [{old_vocab}:{new_vocab}] (should be mean-init):")
    print(f"    row {old_vocab} == mean? {torch.allclose(new_emb[old_vocab].float(), mean_row)}")
    print(f"    row {new_vocab - 1} == mean? {torch.allclose(new_emb[new_vocab - 1].float(), mean_row)}")
    print(f"  Padding rows [{new_vocab}:{padded_vocab}] (should be zero):")
    print(f"    row {new_vocab} all zero? {torch.all(new_emb[new_vocab] == 0)}")
    print(f"    row {padded_vocab - 1} all zero? {torch.all(new_emb[padded_vocab - 1] == 0)}")

    # Verify original rows unchanged
    print(f"  Original rows [0:{old_vocab}] unchanged? "
          f"{torch.equal(new_emb[:old_vocab], emb)}")

    # Update config
    new_cfg = copy.deepcopy(cfg)
    new_cfg["vocab_size"] = new_vocab
    new_cfg["padded_vocab_size"] = padded_vocab
    new_cfg["pad_vocab_size_multiple"] = 16
    new_cfg["fim_tokens"] = {
        "<PRE>": old_vocab,
        "<SUF>": old_vocab + 1,
        "<MID>": old_vocab + 2,
        "<EOT>": old_vocab + 3,
    }
    ckpt["config"] = new_cfg
    ckpt["model"] = state_dict

    # Save
    print(f"\nSaving: {output_path}")
    torch.save(ckpt, output_path)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  Saved ({size_mb:.0f} MB)")

    print(f"\n{'=' * 50}")
    print(f"✓ Embedding resize complete")
    print(f"  Checkpoint: {output_path}")
    print(f"  vocab: {old_vocab} → {new_vocab} (padded: {padded_vocab})")
    print(f"  FIM token IDs: <PRE>={old_vocab} <SUF>={old_vocab+1} "
          f"<MID>={old_vocab+2} <EOT>={old_vocab+3}")
    print(f"  Next: smoke test (load + AR perplexity check)")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Resize embeddings for FIM tokens")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint .pt")
    parser.add_argument("--output", required=True, help="Output checkpoint .pt")
    parser.add_argument("--new-vocab", type=int, default=4004,
                        help="New vocab size including FIM tokens (default: 4004)")
    parser.add_argument("--padded-vocab", type=int, default=4016,
                        help="Padded vocab size, multiple of 16 (default: 4016)")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    resize_checkpoint(args.checkpoint, args.output, args.new_vocab, args.padded_vocab)


if __name__ == "__main__":
    main()
