#!/usr/bin/env python3
"""
Simple text generation with a Morpheus Mamba-2 checkpoint.
Usage: python scripts/export/generate.py --ckpt checkpoints/small_v0_best.pt --prompt "Kaixo"
"""

import argparse
import os
import torch
import sentencepiece as spm


def load_checkpoint(ckpt_path, device="cpu"):
    """Load a training checkpoint and return model + tokenizer."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    padded_vocab = ((config["vocab_size"] + config["pad_vocab_size_multiple"] - 1)
                    // config["pad_vocab_size_multiple"]) * config["pad_vocab_size_multiple"]

    model_config = MambaConfig(
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        vocab_size=padded_vocab,
        ssm_cfg={
            "layer": config.get("ssm_layer", "Mamba2"),
            "d_state": config["d_state"],
            "d_conv": config["d_conv"],
            "expand": config["expand"],
            "headdim": config["headdim"],
            "chunk_size": config["chunk_size"],
        },
        rms_norm=True,
        residual_in_fp32=config["residual_in_fp32"],
        fused_add_norm=config["fused_add_norm"],
    )

    dtype = torch.bfloat16 if config.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    model = MambaLMHeadModel(model_config, device=device, dtype=dtype)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Find tokenizer relative to project root
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "tokenizer", "morpheus.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    return model, sp, config


@torch.no_grad()
def generate(model, sp, prompt, max_new_tokens=50, temperature=0.8, top_p=0.95, device="cpu"):
    """Generate text from a prompt."""
    tokens = sp.encode(prompt, out_type=int)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    dtype = model.backbone.embedding.weight.dtype

    generated = tokens.copy()
    for _ in range(max_new_tokens):
        with torch.amp.autocast(str(device), dtype=dtype):
            logits = model(ids).logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_id = torch.argmax(logits, dim=-1).item()

        if next_id == 0:
            break

        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    return sp.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Morpheus Mamba-2 text generation")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--prompt", default="Kaixo", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    model, sp, config = load_checkpoint(args.ckpt, device=args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"Config: d_model={config['d_model']}, n_layer={config['n_layer']}")
    print()

    print(f">>> {args.prompt}")
    result = generate(model, sp, args.prompt,
                      max_new_tokens=args.max_tokens,
                      temperature=args.temp,
                      top_p=args.top_p,
                      device=args.device)
    print(result)


if __name__ == "__main__":
    main()
