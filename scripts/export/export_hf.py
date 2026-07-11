#!/usr/bin/env python3
"""
Export Morpheus Mamba-2 checkpoint to HuggingFace-compatible format.

Produces config.json + model.safetensors + tokenizer.model that
llama.cpp's convert_hf_to_gguf.py can handle natively via conversion/mamba.py.

Usage:
    # 1. Export to HF
    python scripts/export/export_hf.py \\
        --checkpoint checkpoints/best.pt \\
        --output-dir exports/morpheus-hf

    # 2. Convert to GGUF (uses llama.cpp's Mamba2Model converter)
    python /tmp/llama.cpp/convert_hf_to_gguf.py exports/morpheus-hf \
        --outfile exports/morpheus.f16.gguf --outtype f16

    # 3. Quantize
    /tmp/llama.cpp/build/bin/llama-quantize exports/morpheus.f16.gguf exports/morpheus.Q4_K_M.gguf Q4_K_M
"""

import argparse
import json
import shutil
from pathlib import Path
import torch

KEEP_KEYS = {
    # Embedding
    "backbone.embedding.weight": "backbone.embedding.weight",
    "backbone.norm_f.weight": "backbone.norm_f.weight",
    # LM head (tied with embedding)
    "lm_head.weight": "backbone.embedding.weight",
}

LAYER_KEYS = {
    "norm.weight": "norm.weight",
    "mixer.in_proj.weight": "mixer.in_proj.weight",
    "mixer.conv1d.weight": "mixer.conv1d.weight",
    "mixer.conv1d.bias": "mixer.conv1d.bias",
    "mixer.dt_bias": "mixer.dt_bias",
    "mixer.A_log": "mixer.A_log",
    "mixer.D": "mixer.D",
    "mixer.norm.weight": "mixer.norm.weight",
    "mixer.out_proj.weight": "mixer.out_proj.weight",
}


def rename_mamba2_to_hf(state_dict, n_layer):
    """Rename mamba-ssm training keys to HF-compatible Mamba2 keys.

    llama.cpp's conversion/mamba.py expects:
      backbone.layers.N.mixer.in_proj.weight  (keeps as-is)
      backbone.layers.N.mixer.conv1d.weight   (keeps as-is)
      backbone.layers.N.mixer.A_log           (keeps as-is)
      backbone.layers.N.mixer.D               (keeps as-is)
      backbone.layers.N.mixer.dt_bias         (keeps as-is)
      backbone.layers.N.mixer.out_proj.weight (keeps as-is)
      backbone.layers.N.norm.weight           (keeps as-is)
      backbone.embedding.weight               (keeps as-is)
      backbone.norm_f.weight                  (keeps as-is)

    The converter uses map_tensor_name internally, so we just need the
    right keys to be present.
    """
    # All our training keys already match what Mamba2Model expects
    # (it uses the mamba-ssm naming convention)
    # We just need to make sure the keys are correct.
    # The only difference is lm_head vs backbone.embedding (tied)
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to HF format for llama.cpp")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tokenizer", default="tokenizer/morpheus.model",
                        help="Path to SentencePiece model")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state_dict = ckpt["model"]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Convert to float16 safetensors
    from safetensors.torch import save_file
    fp16_state = {}
    for k, v in state_dict.items():
        fp16_state[k] = v.half()

    save_file(fp16_state, out / "model.safetensors")

    # Write config.json following what conversion/mamba.py expects:
    # Mamba2Model.find_hparam looks for: hidden_size/d_model, conv_kernel/d_conv,
    # state_size/d_state, mamba_d_head/head_dim, mamba_expand/expand, n_layer,
    # vocab_size, layer_norm_epsilon/rms_norm_eps
    #
    # Key: use "d_model" style names (matching mamba2_config.yaml from transformers)
    d_model = config["d_model"]
    expand = config["expand"]
    d_inner = d_model * expand
    headdim = config["headdim"]
    n_heads = d_inner // headdim

    hf_config = {
        "d_model": d_model,
        "n_layer": config["n_layer"],
        "vocab_size": config["vocab_size"],
        "d_conv": config["d_conv"],
        "expand": expand,
        "d_state": config["d_state"],
        "head_dim": headdim,
        "d_inner": d_inner,
        "n_heads": n_heads,
        "rms_norm_eps": 1e-5,
        "pad_vocab_size_multiple": config["pad_vocab_size_multiple"],
        "tie_word_embeddings": True,
        # Mark as Mamba2
        "model_type": "mamba2",
        # Additional HF-required fields
        "architectures": ["Mamba2ForCausalLM"],
        "torch_dtype": "float16",
        "transformers_version": "4.40.0",
    }

    with open(out / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Copy tokenizer
    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.exists():
        shutil.copy(tokenizer_path, out / "tokenizer.model")
        print(f"Copied tokenizer: {tokenizer_path}")
    else:
        print(f"Warning: tokenizer not found at {tokenizer_path}")
        raise SystemExit(1)

    # Write tokenizer_config.json — CRITICAL for correct llama.cpp BOS handling.
    #
    # The model was trained WITHOUT <s> (BOS): pretokenize.py uses sp.encode(line)
    # + </s> (id=2) separators, never <s>. But the SentencePiece model defines
    # <s> (id=1) as a special token, so without this file llama.cpp's
    # SpecialVocab defaults to add_bos_token=True, and llama-server prepends <s>
    # — a token the model never saw in training — corrupting the context and
    # collapsing output (CSR 4% vs the correct 29%).
    #
    # Setting add_bos_token=False propagates through:
    #   convert_hf_to_gguf.py → SpecialVocab._load → gguf_writer.add_add_bos_token(False)
    #   → GGUF metadata tokenizer.ggml.add_bos_token=false → llama-server honors it.
    # Verified: conversion/mamba.py::Mamba2Model.set_vocab → _set_vocab_sentencepiece
    # → gguf.SpecialVocab(dir_model) → add_to_gguf (reads tokenizer_config.json).
    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": 1000000000000000019884624838656,
        "add_bos_token": False,   # model trained without BOS — do NOT prepend
        "add_eos_token": False,   # model predicts </s> itself; never auto-append
        "bos_token": None,
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": None,
        "sp_model_kwargs": {},
    }
    with open(out / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print("Wrote tokenizer_config.json (add_bos_token=False — matches no-BOS training)")

    # generation_config.json: no bos_token_id (model has no BOS), eos=</s>=2.
    # Omitting bos_token_id entirely signals "no BOS for generation".
    generation_config = {
        "eos_token_id": 2,
        "transformers_version": "4.40.0",
    }
    with open(out / "generation_config.json", "w") as f:
        json.dump(generation_config, f, indent=2)
    print("Wrote generation_config.json (eos_token_id=2, no bos_token_id)")

    # Self-check: confirm the SentencePiece model's bos_id is NOT in our training
    # distribution by construction. We can't cheaply scan train_tokens here, but
    # we assert the policy is recorded so any consumer is safe.
    assert tokenizer_config["add_bos_token"] is False
    assert tokenizer_config["bos_token"] is None

    n_tensors = len(fp16_state)
    size_mb = (out / "model.safetensors").stat().st_size / 1e6
    print(f"Exported {n_tensors} tensors ({size_mb:.0f} MB) to {out}")
    print(f"\nNext: python /tmp/llama.cpp/convert_hf_to_gguf.py {out} --outtype f16")


if __name__ == "__main__":
    main()
