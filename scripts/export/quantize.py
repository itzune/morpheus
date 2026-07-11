#!/usr/bin/env python3
"""
End-to-end checkpoint → GGUF quantization pipeline.

Converts a Morpheus v2 Mamba-2 training checkpoint to quantized GGUF models
that run on llama.cpp CPU inference.

Pipeline:
  1. export_hf.py:       checkpoint .pt → HuggingFace format (config.json + model.safetensors)
  2. convert_hf_to_gguf: HuggingFace → F16 GGUF (llama.cpp's native Mamba2 converter)
  3. llama-quantize:     F16 GGUF → Q4_K_M + Q5_K_M (and any other requested quants)

Usage:
    # Default: produces Q4_K_M + Q5_K_M
    python scripts/export/quantize.py --checkpoint checkpoints/best.pt --output-dir exports

    # Custom quants
    python scripts/export/quantize.py --checkpoint checkpoints/best.pt --quants Q4_K_M Q8_0

    # Specify llama.cpp paths
    python scripts/export/quantize.py --checkpoint checkpoints/best.pt \\
        --llama-cpp-dir /tmp/llama.cpp --llama-cpp-build build

Dependencies (must be available on the machine running this):
  - torch, safetensors (for export_hf.py)
  - recent llama.cpp with convert_hf_to_gguf.py and llama-quantize built
    (must include `conversion/mamba.py` / `Mamba2ForCausalLM` support)

Note: export_hf.py must run on a GPU server (needs torch with CUDA to load
the checkpoint). Quantization runs on CPU only and is fast (< 5 seconds each).
Older llama.cpp versions may produce broken Mamba-2 GGUFs even if the binary exists.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


QUANT_MAP = {
    "Q4_K_M": {
        "desc": "Q4_K_M — 4-bit, medium (recommended)",
        "suffix": "Q4_K_M",
    },
    "Q5_K_M": {
        "desc": "Q5_K_M — 5-bit, medium (Basque morphology)",
        "suffix": "Q5_K_M",
    },
    "Q8_0": {
        "desc": "Q8_0 — 8-bit, round-to-nearest",
        "suffix": "Q8_0",
    },
    "F16": {
        "desc": "F16 — no quantization (baseline)",
        "suffix": "f16",
    },
}


def run(cmd, desc=None):
    """Run a command, printing output. Exit on failure."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print stdout (trim repetitive per-layer output)
    lines = result.stdout.splitlines()
    if len(lines) > 30:
        head = lines[:10]
        tail = lines[-10:]
        for l in head:
            print(f"  {l}")
        print(f"  ... ({len(lines) - 20} lines omitted) ...")
        for l in tail:
            print(f"  {l}")
    else:
        print(result.stdout)
    if result.returncode != 0:
        print(f"\n  ERROR:\n{result.stderr}")
        sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint → GGUF quantization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python scripts/export/quantize.py --checkpoint checkpoints/best.pt\n"
               "  python scripts/export/quantize.py --checkpoint checkpoints/step_5000.pt --name my-model\n",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt training checkpoint")
    parser.add_argument("--output-dir", default="exports", help="Output directory (default: exports)")
    parser.add_argument("--name", default=None,
                        help="Model name prefix (default: derived from checkpoint stem)")
    parser.add_argument("--tokenizer", default="tokenizer/basque_unigram_4000.model",
                        help="Path to SentencePiece .model file")
    parser.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q5_K_M"],
                        help=f"Quantization formats (default: Q4_K_M Q5_K_M). "
                             f"Available: {', '.join(QUANT_MAP.keys())}")
    parser.add_argument("--keep-f16", action="store_true",
                        help="Keep intermediate F16 GGUF (default: delete after quantizing)")
    parser.add_argument("--llama-cpp-dir", default="/root/llama.cpp",
                        help="Path to llama.cpp repo (default: /root/llama.cpp)")
    parser.add_argument("--llama-cpp-build", default="build",
                        help="Build dir relative to llama-cpp-dir (default: build)")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HF export (use existing --hf-dir)")
    parser.add_argument("--skip-gguf", action="store_true",
                        help="Skip F16 GGUF conversion (if F16 already exists)")
    parser.add_argument("--hf-dir", default=None,
                        help="Existing HuggingFace dir (when --skip-hf). Default: <output>/hf_<name>")
    args = parser.parse_args()

    # --- Validate paths ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    llama_cpp = Path(args.llama_cpp_dir)
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"Error: convert_hf_to_gguf.py not found at {convert_script}")
        print("  Clone llama.cpp: git clone https://github.com/ggml-org/llama.cpp.git")
        sys.exit(1)

    mamba_converter = llama_cpp / "conversion" / "mamba.py"
    if not mamba_converter.exists():
        print(f"Error: Mamba-2 converter not found at {mamba_converter}")
        print("  Use a recent llama.cpp checkout with conversion/mamba.py and Mamba2ForCausalLM support")
        sys.exit(1)

    quantize_bin = llama_cpp / args.llama_cpp_build / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print(f"Error: llama-quantize not found at {quantize_bin}")
        print(f"  Build it: cd {llama_cpp} && mkdir -p build && cd build && cmake .. && make llama-quantize -j$(nproc)")
        sys.exit(1)

    # --- Determine output names ---
    name = args.name or ckpt_path.stem  # e.g., "best", "step_0022000"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_dir = Path(args.hf_dir) if args.hf_dir else (out_dir / f"hf_{name}")
    f16_gguf = out_dir / f"{name}.f16.gguf"

    # --- Step 1: HF export ---
    if not args.skip_hf:
        # Clean previous HF dir
        if hf_dir.exists():
            shutil.rmtree(hf_dir)

        run([
            sys.executable, "scripts/export/export_hf.py",
            "--checkpoint", str(ckpt_path),
            "--output-dir", str(hf_dir),
            "--tokenizer", args.tokenizer,
        ], desc="Step 1/3: Export checkpoint → HuggingFace format")

    # --- Step 2: HF → F16 GGUF ---
    if not args.skip_gguf and not f16_gguf.exists():
        run([
            sys.executable, str(convert_script),
            str(hf_dir),
            "--outfile", str(f16_gguf),
            "--outtype", "f16",
        ], desc="Step 2/3: Convert HuggingFace → F16 GGUF")
    else:
        print(f"\n  F16 GGUF already exists: {f16_gguf} ({f16_gguf.stat().st_size / 1e6:.0f} MB)")

    # --- Step 3: Quantize ---
    results = {}
    for q in args.quants:
        if q == "F16":
            results["F16"] = f16_gguf
            continue

        q_info = QUANT_MAP.get(q)
        if not q_info:
            print(f"Warning: unknown quant format '{q}', skipping")
            continue

        q_gguf = out_dir / f"{name}.{q_info['suffix']}.gguf"
        run([
            str(quantize_bin),
            str(f16_gguf),
            str(q_gguf),
            q,
        ], desc=f"Step 3/3: Quantize → {q_info['desc']}")

        results[q] = q_gguf

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  Done! Quantized models:")
    print(f"{'='*60}")
    f16_size_mb = f16_gguf.stat().st_size / 1e6 if f16_gguf.exists() else None
    for q, path in results.items():
        size_mb = path.stat().st_size / 1e6
        if f16_size_mb and f16_size_mb > 0:
            bpw = size_mb * 16 / f16_size_mb
            print(f"  {q:10s}  {size_mb:6.0f} MB  ({bpw:.2f} BPW)  {path}")
        else:
            print(f"  {q:10s}  {size_mb:6.0f} MB             {path}")

    if not args.keep_f16:
        f16_gguf.unlink()
        print(f"\n  Removed intermediate: {f16_gguf.name}")

    print(f"\n  Next: use with llama.cpp")
    print(f"    llama-cli -m {list(results.values())[-1]} -p 'Kaixo, zer moduz' -n 50")


if __name__ == "__main__":
    main()
