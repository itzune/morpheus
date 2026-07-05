"""
Export a Mamba training checkpoint to HuggingFace-compatible format.

Converts PyTorch checkpoint → safetensors weights + config.json,
suitable for llama.cpp's convert_hf_to_gguf.py script.

Usage:
    python scripts/export_hf.py \\
        --checkpoint checkpoints/best.pt \\
        --output-dir exports/morpheus-v2-mamba-hf

Full implementation: Morpheus_v2_Mamba.md §7.2 Step 1
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(
        description="Export Mamba checkpoint to HuggingFace format"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to training checkpoint .pt file")
    parser.add_argument("--output-dir", required=True, help="Output directory for HF format")
    parser.add_argument("--tokenizer", default="tokenizer/basque_unigram_32k.model",
                        help="Path to SentencePiece model")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    state_dict = ckpt["model"]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Convert to float16 for export
    fp16_state = {k: v.half() for k, v in state_dict.items()}
    save_file(fp16_state, out / "model.safetensors")

    # Write config.json for llama.cpp conversion
    hf_config = {
        "model_type": "mamba2",
        "d_model": config.get("d_model", 960),
        "n_layer": config.get("n_layer", 32),
        "vocab_size": config.get("vocab_size", 32000),
        "ssm_cfg": config.get("ssm_cfg", {}),
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
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

    print(f"Exported to {out}")


if __name__ == "__main__":
    main()
