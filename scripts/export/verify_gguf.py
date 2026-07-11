#!/usr/bin/env python3
"""Verify GGUF metadata for Morpheus exports."""
import sys
sys.path.insert(0, "/root/llama.cpp/gguf-py")
from gguf import GGUFReader

path = sys.argv[1] if len(sys.argv) > 1 else "exports/step_0016000.Q4_K_M.gguf"
r = GGUFReader(path)

keys = [
    "general.architecture", "general.name",
    "mamba.context_length", "mamba.embedding_length", "mamba.block_count",
    "mamba.ssm.conv_kernel", "mamba.ssm.state_size", "mamba.ssm.time_step_rank",
    "mamba.ssm.inner_size", "mamba.ssm.head_count",
    "tokenizer.ggml.model",
    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
]

print(f"=== {path} ===")
for key in keys:
    if key not in r.fields:
        continue
    f = r.fields[key]
    t = f.types[0] if f.types else None
    tname = t.name if hasattr(t, "name") else str(t)
    part = f.parts[f.data[0]]
    if tname in ("UINT32", "INT32"):
        print(f"  {key} = {int(part[0])}")
    elif tname == "STRING":
        print(f"  {key} = {bytes(part).decode('utf-8', errors='replace')[:60]}")
    elif tname == "FLOAT32":
        print(f"  {key} = {float(part[0])}")
    elif tname == "BOOL":
        print(f"  {key} = {bool(part[0])}")
    else:
        print(f"  {key} (type={tname})")

tok = r.fields.get("tokenizer.ggml.tokens")
if tok:
    print(f"  vocab_size (token count) = {len(tok.data)}")
