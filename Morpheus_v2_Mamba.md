# Morpheus v2 — Mamba SSM Implementation Guide (Path C)

> **Audience:** ML engineer implementing the Morpheus v2 autocomplete model
>
> **Architecture:** Mamba-2 (State Space Model), ~200M parameters
>
> **Goal:** On-device Basque autocomplete with P90 ≤ 50ms on consumer CPU
>
> **Hardware:** 1× NVIDIA L40 (48 GB VRAM) for training; consumer x86-64 CPU for inference

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Model Configuration](#2-model-configuration)
3. [Environment Setup](#3-environment-setup)
4. [Data Pipeline](#4-data-pipeline)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Deployment: Export & CPU Inference](#7-deployment-export--cpu-inference)
8. [Inference Server Integration](#8-inference-server-integration)
9. [Phased Execution Plan](#9-phased-execution-plan)

---

## 1. Architecture Overview

### 1.1 Why Mamba for Morpheus

Mamba is a **Selective State Space Model (SSM)** that shares the key inference property of LSTMs — **constant-size hidden state, no KV cache** — while dramatically improving language modeling capacity through:

- **Selective state updates:** Input-dependent gating (analogous to LSTM gates) that dynamically controls what the model remembers and forgets
- **Parallelizable training:** Unlike LSTMs, the recurrence can be computed as a parallel scan during training, achieving GPU utilization comparable to Transformers
- **Linear-time inference:** O(n) in context length, with constant memory per step

### 1.2 Mamba-2 vs Mamba-1

| Feature | Mamba-1 (2023) | Mamba-2 (2024) |
|---------|---------------|---------------|
| Core algorithm | Selective Scan | **SSD (Structured State Space Duality)** |
| Training speed | ~3× Transformer | **~6-8× Transformer** (optimized for tensor cores) |
| State dimension | 16 | **64–128** (larger states for better quality) |
| Head structure | None | **Multi-head** (similar to multi-head attention) |
| Hardware utilization | Good | **Excellent** (matrix multiply formulation) |

**Recommendation: Use Mamba-2.** It is strictly superior in both training throughput and model quality, and is the standard for new projects in 2026.

### 1.3 How Mamba Generates Text

At inference time, Mamba operates as a recurrent model:

```
For each token t in the input sequence:
    1. Embed token → x_t (d_model-dimensional vector)
    2. For each Mamba-2 layer:
        a. Project x_t → (B_t, C_t, Δ_t)    # Input-dependent SSM params
        b. Discretize continuous SSM:   A_bar = exp(Δ_t * A)
        c. Update hidden state:         h_t = A_bar * h_{t-1} + B_t * x_t
        d. Compute output:              y_t = C_t * h_t
    3. Project y_t → logits over vocabulary
    4. Sample/argmax → next token
```

The hidden state `h` has a **fixed size** regardless of how many tokens have been processed. This is the property that enables constant-memory, constant-latency inference — exactly what Morpheus needs.

---

## 2. Model Configuration

### 2.1 Target: ~200M Parameters

Based on the Mamba-2 scaling curve (130M → 370M reference configs), here is the target configuration:

```python
# morpheus_v2_config.py
MORPHEUS_MAMBA2_CONFIG = {
    # Architecture
    "d_model": 960,              # Hidden dimension (divisible by headdim)
    "n_layer": 32,               # Number of Mamba-2 blocks
    "d_state": 64,               # SSM state dimension (Mamba-2 default)
    "d_conv": 4,                 # Local convolution width
    "expand": 2,                 # MLP expansion factor (inner_dim = expand * d_model)
    "headdim": 64,               # Head dimension for multi-head SSM
    "chunk_size": 256,           # Chunk size for SSD parallel scan

    # Vocabulary & Embedding
    "vocab_size": 32000,         # Reuse Morpheus v1 SentencePiece Unigram tokenizer
    "pad_vocab_size_multiple": 16, # Pad for hardware alignment

    # Regularization
    "residual_in_fp32": True,    # Keep residual stream in FP32 for stability
    "fused_add_norm": True,      # Fused RMSNorm + residual add

    # Precision
    "dtype": "bfloat16",         # Training precision
}
```

### 2.2 Parameter Count Estimate

The parameter count is approximately:

```
Embedding:    vocab_size × d_model = 32,000 × 960  ≈ 30.7M
Per layer:    ~3 × expand × d_model² = 3 × 2 × 960² ≈ 5.5M
All layers:   32 × 5.5M                             ≈ 176.6M
LM Head:      (weight-tied with embedding)           ≈ 0M
──────────────────────────────────────────────────────────────
TOTAL                                                ≈ 207M params
```

This puts the model at ~207M parameters, with **85% in the Mamba layers** (vs 86% in embeddings for the v1 LSTM). This is a far more efficient parameter allocation.

### 2.3 Configuration Variants

| Variant | d_model | n_layer | d_state | ~Params | Use case |
|---------|---------|---------|---------|---------|----------|
| **Morpheus-Small** | 768 | 24 | 64 | ~130M | Fast iteration, baseline |
| **Morpheus-Base** | 960 | 32 | 64 | ~200M | Primary target (**recommended**) |
| **Morpheus-Large** | 1024 | 48 | 128 | ~370M | Maximum quality, if latency allows |

Start with **Morpheus-Small** for rapid experimentation, then scale to **Morpheus-Base** for production.

---

## 3. Environment Setup

### 3.1 Docker Environment

Since the project uses Docker for all Node/dev commands, extend the existing setup with a GPU-capable training container:

```dockerfile
# Dockerfile.train
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git wget cmake build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

# PyTorch (CUDA 12.4)
RUN pip install torch==2.4.* torchvision --index-url https://download.pytorch.org/whl/cu124

# Mamba dependencies (must install in this order)
RUN pip install causal-conv1d>=1.4.0 --no-build-isolation
RUN pip install mamba-ssm --no-build-isolation

# Training utilities
RUN pip install \
    sentencepiece==0.2.* \
    wandb \
    safetensors \
    tqdm \
    numpy \
    datasets

WORKDIR /workspace
```

```yaml
# docker-compose.train.yml
services:
  morpheus-train:
    build:
      context: .
      dockerfile: Dockerfile.train
    volumes:
      - .:/workspace
      - ./data:/data
      - ./checkpoints:/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: '16g'  # Shared memory for DataLoader workers
```

### 3.2 Verify Installation

```bash
docker compose -f docker-compose.train.yml run morpheus-train python3 -c "
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

# Quick smoke test
config = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
model = MambaLMHeadModel(config, device='cuda', dtype=torch.bfloat16)
x = torch.randint(0, 100, (1, 16), device='cuda')
out = model(x)
print(f'Smoke test OK — output shape: {out.logits.shape}')
"
```

---

## 4. Data Pipeline

### 4.1 Reuse v1 Assets

| Asset | Path | Reusable? |
|-------|------|-----------|
| SentencePiece model | `tokenizer/basque_unigram_32k.model` | ✅ Yes |
| Train split | `data/splits/train/` | ✅ Yes |
| Valid split | `data/splits/valid/` | ✅ Yes |
| Test split | `data/splits/test/` | ✅ Yes |

### 4.2 Pre-tokenize and Pack into Binary Format

For training efficiency, pre-tokenize the entire corpus and store it as a memory-mapped numpy array. This avoids tokenization overhead during training and enables efficient random access.

```python
# scripts/pretokenize.py
"""Pre-tokenize corpus into memory-mapped numpy array."""
import numpy as np
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm

SP_MODEL = "tokenizer/basque_unigram_32k.model"
INPUT_DIR = Path("data/splits/train")
OUTPUT_FILE = "data/train_tokens.npy"
SEQ_SEP_TOKEN = 0  # <eos> / document separator

def main():
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)

    # First pass: count total tokens
    all_tokens = []
    for txt_file in sorted(INPUT_DIR.glob("*.txt")):
        with open(txt_file) as f:
            for line in tqdm(f, desc=txt_file.name):
                line = line.strip()
                if not line:
                    continue
                tokens = sp.encode(line, out_type=int)
                all_tokens.extend(tokens)
                all_tokens.append(SEQ_SEP_TOKEN)  # Document boundary

    # Write as memory-mapped uint16 (vocab < 65535)
    arr = np.array(all_tokens, dtype=np.uint16)
    np.save(OUTPUT_FILE, arr)
    print(f"Saved {len(arr):,} tokens to {OUTPUT_FILE}")
    print(f"Size: {arr.nbytes / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
```

### 4.3 Dataset Class

```python
# src/dataset.py
"""Memory-mapped dataset for causal LM training."""
import numpy as np
import torch
from torch.utils.data import Dataset

class MemmapTokenDataset(Dataset):
    """Loads pre-tokenized data from a numpy file via memory mapping."""

    def __init__(self, path: str, seq_len: int = 512):
        self.data = np.load(path, mmap_mode='r')
        self.seq_len = seq_len
        # Number of non-overlapping sequences
        self.n_sequences = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target shift

        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])   # input:  tokens[0..seq_len-1]
        y = torch.from_numpy(chunk[1:])    # target: tokens[1..seq_len]
        return x, y
```

---

## 5. Training Pipeline

### 5.1 Hyperparameters

Based on Mamba-2 training best practices and the Morpheus corpus size:

```python
# config/train_config.py
TRAIN_CONFIG = {
    # Data
    "seq_len": 512,                # Context window for training
    "batch_size": 128,             # Per-GPU batch size
    "gradient_accumulation": 1,    # Effective batch = 128

    # Optimization
    "optimizer": "AdamW",
    "learning_rate": 2e-3,         # Peak LR (Mamba is less LR-sensitive than Transformers)
    "min_lr": 1e-5,                # Cosine decay floor
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,              # Gradient clipping (important for SSM stability)

    # Schedule
    "warmup_tokens": 50_000_000,   # ~50M tokens warmup (~0.5% of total)
    "total_tokens": 10_000_000_000, # 10B tokens (multi-epoch over 1.17B-word corpus)

    # Precision
    "dtype": "bfloat16",           # BF16 for training stability
    "compile": True,               # torch.compile for speed

    # Logging
    "log_interval": 50,            # Log every N steps
    "eval_interval": 1000,         # Eval every N steps
    "save_interval": 5000,         # Checkpoint every N steps
}
```

### 5.2 Training Script

```python
# train.py
"""Morpheus v2 Mamba-2 training script."""
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

from src.dataset import MemmapTokenDataset
from config.train_config import TRAIN_CONFIG as C

# ─── Model Setup ────────────────────────────────────────────────
config = MambaConfig(
    d_model=960,
    n_layer=32,
    vocab_size=32000,
    ssm_cfg={
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

model = MambaLMHeadModel(config, device="cuda", dtype=torch.bfloat16)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Weight tying: share embedding and LM head
model.lm_head.weight = model.backbone.embedding.weight

# Optional: compile for speed (PyTorch 2.4+)
if C["compile"]:
    model = torch.compile(model)

# ─── Data ────────────────────────────────────────────────────────
train_ds = MemmapTokenDataset("data/train_tokens.npy", seq_len=C["seq_len"])
valid_ds = MemmapTokenDataset("data/valid_tokens.npy", seq_len=C["seq_len"])

train_loader = DataLoader(
    train_ds,
    batch_size=C["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=C["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ─── Optimizer ───────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=C["learning_rate"],
    betas=(C["beta1"], C["beta2"]),
    weight_decay=C["weight_decay"],
)

# ─── LR Schedule: Linear Warmup + Cosine Decay ──────────────────
total_steps = C["total_tokens"] // (C["seq_len"] * C["batch_size"] * C["gradient_accumulation"])
warmup_steps = C["warmup_tokens"] // (C["seq_len"] * C["batch_size"] * C["gradient_accumulation"])

def get_lr(step):
    if step < warmup_steps:
        return C["learning_rate"] * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return C["min_lr"] + 0.5 * (C["learning_rate"] - C["min_lr"]) * (
        1 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def evaluate(model, loader, max_batches=100):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0
    n = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.cuda(), y.cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(x)
            loss = F.cross_entropy(
                output.logits.view(-1, output.logits.size(-1)),
                y.view(-1),
                ignore_index=0,
            )
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / n


def save_checkpoint(model, optimizer, step, path):
    """Save training checkpoint."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.__dict__,
        },
        path,
    )
    print(f"  Saved checkpoint: {path}")


# ─── Training Loop ───────────────────────────────────────────────
global_step = 0
tokens_seen = 0
best_valid_loss = float("inf")
t0 = time.time()

for epoch in range(100):  # Train until total_tokens reached
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        if tokens_seen >= C["total_tokens"]:
            break

        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Update LR
        lr = get_lr(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(x)
            logits = output.logits                  # (B, T, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=0,                     # Ignore <eos> padding
            )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += x.numel()
        global_step += 1

        # Logging
        if global_step % C["log_interval"] == 0:
            ppl = math.exp(min(loss.item(), 20))
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            print(
                f"step={global_step:>6d}  loss={loss.item():.4f}  ppl={ppl:.1f}"
                f"  lr={lr:.2e}  tok/s={tps:.0f}"
            )

        # Validation
        if global_step % C["eval_interval"] == 0:
            valid_loss = evaluate(model, valid_loader)
            valid_ppl = math.exp(min(valid_loss, 20))
            print(f"  [VALID] loss={valid_loss:.4f}  ppl={valid_ppl:.1f}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_checkpoint(model, optimizer, global_step, "checkpoints/best.pt")

        # Checkpoint
        if global_step % C["save_interval"] == 0:
            save_checkpoint(
                model, optimizer, global_step,
                f"checkpoints/step_{global_step}.pt",
            )

    if tokens_seen >= C["total_tokens"]:
        break

print(f"Training complete. Total tokens: {tokens_seen:,}")
```

### 5.3 Expected Training Profile on L40

| Metric | Estimate |
|--------|----------|
| VRAM usage (batch 128, seq 512) | ~15-20 GB |
| Training throughput | ~25-40K tokens/sec |
| Tokens per epoch (1.17B words ≈ 2.5B subword tokens) | ~2.5B |
| Epochs to 10B tokens | ~4 |
| Wall-clock per epoch | ~17-28 hours |
| **Total training time** | **~3-5 days** |

### 5.4 Training Tips

1. **Monitor BF16 stability.** If you see NaN losses, reduce the learning rate or increase warmup. Mamba's selective scan can amplify numerical issues in early training.

2. **Gradient clipping is essential.** Set `grad_clip=1.0`. SSM models can experience gradient spikes, especially in early training when the discretization parameters (Δ) are not yet calibrated.

3. **Don't skip the warmup.** The Δ (delta/timescale) parameters need a gentle start. At least 50M tokens of warmup is recommended.

4. **`torch.compile` helps significantly.** On Ada Lovelace (L40), `torch.compile` with `mode="reduce-overhead"` can provide 20-40% throughput improvement for Mamba-2.

5. **Sequence packing is optional but helpful.** For a multi-epoch setup, simple non-overlapping windows (as in our Dataset class) are sufficient. Sequence packing (concatenating short documents to fill the window) can be added later for marginal improvement.

---

## 6. Evaluation Protocol

### 6.1 Metrics (Carry Forward from v1)

| Metric | Target | How |
|--------|--------|-----|
| **Valid Perplexity** | < 50 | Cross-entropy on held-out valid set, exponentiated |
| **Top-1 Accuracy** | > 20% | Exact match of top prediction vs actual next word |
| **Hit@3** | > 40% | Actual next word in top-3 predictions |
| **Hit@5** | > 50% | Actual next word in top-5 predictions |
| **Morphological agreement** | Qualitative | Manual review of ergative/absolutive suffix predictions |
| **Inference latency (P90)** | ≤ 50ms | Benchmark on target consumer CPU |
| **Post-quantization PPL degradation** | ≤ 3% | Compare FP32 vs quantized PPL |

### 6.2 Evaluation Script

```python
# eval.py
"""Evaluate Morpheus v2 on next-word prediction accuracy."""
import torch
import sentencepiece as spm
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def evaluate_autocomplete(model, sp, test_sentences, device="cuda", k_values=[1, 3, 5]):
    """Compute Hit@K for next-word prediction on test sentences."""
    model.eval()
    hits = {k: 0 for k in k_values}
    total = 0

    for sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=int)
        if len(tokens) < 3:
            continue

        # For each position (after context of >= 2 tokens), predict next token
        for i in range(2, len(tokens)):
            context = torch.tensor([tokens[:i]], device=device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(context)
                logits = output.logits[0, -1, :]  # Last position logits

            target = tokens[i]
            top_k_preds = torch.topk(logits, max(k_values)).indices.tolist()

            for k in k_values:
                if target in top_k_preds[:k]:
                    hits[k] += 1
            total += 1

    results = {}
    for k in k_values:
        results[f"Hit@{k}"] = hits[k] / total if total > 0 else 0.0
    results["total_predictions"] = total
    return results
```

---

## 7. Deployment: Export & CPU Inference

### 7.1 Deployment Strategy: llama.cpp (GGUF)

> ⚠️ **Critical finding from research:** Direct ONNX export of Mamba models is **not viable** for production CPU inference. The selective scan operator has no efficient ONNX equivalent — decomposing it into ONNX primitives results in orders-of-magnitude slowdown.
>
> **Do NOT use the v1 ONNX pipeline for Mamba. Use llama.cpp instead.**

`llama.cpp` has **native Mamba/Mamba-2 architecture support** with optimized CPU kernels and GGUF quantization.

### 7.2 Export Pipeline: PyTorch → HuggingFace → GGUF

#### Step 1: Convert checkpoint to HuggingFace format

```python
# scripts/export_hf.py
"""Convert Mamba training checkpoint to HuggingFace-compatible format."""
import torch
import json
from pathlib import Path
from safetensors.torch import save_file

CHECKPOINT = "checkpoints/best.pt"
OUTPUT_DIR = "exports/morpheus-v2-mamba-hf"

def export():
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    config = ckpt["config"]
    state_dict = ckpt["model"]

    # Save model weights in safetensors format
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # Convert to float16 for export
    fp16_state = {k: v.half() for k, v in state_dict.items()}
    save_file(fp16_state, out / "model.safetensors")

    # Write config.json for llama.cpp conversion
    hf_config = {
        "model_type": "mamba2",
        "d_model": config["d_model"],
        "n_layer": config["n_layer"],
        "vocab_size": config["vocab_size"],
        "ssm_cfg": config.get("ssm_cfg", {}),
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
    }

    with open(out / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Copy tokenizer
    import shutil
    shutil.copy("tokenizer/basque_unigram_32k.model", out / "tokenizer.model")

    print(f"Exported to {OUTPUT_DIR}")

if __name__ == "__main__":
    export()
```

#### Step 2: Convert to GGUF

```bash
# Clone llama.cpp (if not already)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Convert HF model to GGUF (FP16 baseline)
python3 convert_hf_to_gguf.py \
    ../exports/morpheus-v2-mamba-hf \
    --outfile ../exports/morpheus-v2-f16.gguf \
    --outtype f16

# Quantize to Q5_K_M (recommended for Basque morphology preservation)
./build/bin/llama-quantize \
    ../exports/morpheus-v2-f16.gguf \
    ../exports/morpheus-v2-Q5_K_M.gguf \
    Q5_K_M
```

#### Step 3: Validate quantized model

```bash
# Quick perplexity check
./build/bin/llama-perplexity \
    -m ../exports/morpheus-v2-Q5_K_M.gguf \
    -f ../data/splits/valid/valid_sample.txt
```

**Quantization quality gate:** If quantized PPL degrades > 3% vs FP16, try Q6_K or Q8_0 instead of Q5_K_M.

### 7.3 Expected Model Sizes

| Format | Size on disk | Notes |
|--------|-------------|-------|
| FP32 (PyTorch checkpoint) | ~800 MB | Training only |
| FP16 (GGUF) | ~400 MB | Reference quality |
| **Q5_K_M (GGUF)** | **~145 MB** | **Recommended for deployment** |
| Q4_K_M (GGUF) | ~115 MB | Smaller, slight quality loss |
| Q8_0 (GGUF) | ~210 MB | Near-FP16 quality |

### 7.4 CPU Inference Latency

With llama.cpp on a modern consumer CPU (Intel i7-13700 / AMD Ryzen 7 7700):

| Model size | Quant | Estimated tok/s | Per-token latency | 5-token suggestion |
|-----------|-------|----------------|-------------------|-------------------|
| 130M | Q5_K_M | 80-120 | ~8-12 ms | ~40-60 ms ✅ |
| **200M** | **Q5_K_M** | **50-80** | **~12-20 ms** | **~60-100 ms** ⚠️ |
| 200M | Q4_K_M | 60-90 | ~11-17 ms | ~55-85 ms ✅ |
| 370M | Q5_K_M | 30-50 | ~20-33 ms | ~100-165 ms ❌ |

> **Assessment:** 200M at Q5_K_M is at the edge of the 50ms single-token target. For a 3-token suggestion (the common case), it's ~36-60ms — **within budget**. For 5-token suggestions, consider Q4_K_M or the 130M variant as fallback.
>
> **MWE token injection** (from v1) reduces autoregressive steps from 5 to 2-3 for common phrases, which brings the 200M model comfortably within budget.

---

## 8. Inference Server Integration

### 8.1 Option A: llama.cpp Server Mode (Simplest)

```bash
# Run the Morpheus v2 inference server
./llama-server \
    -m morpheus-v2-Q5_K_M.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -c 512 \
    -n 5 \
    --threads 4 \
    --mlock
```

Client request (compatible with the existing v1 WebSocket demo):

```bash
curl http://127.0.0.1:8080/completion \
    -d '{"prompt": "Gaur egun oso", "n_predict": 3, "temperature": 0.0}'
```

### 8.2 Option B: Custom Rust Serving Binary

If tighter integration with the existing Morpheus demo server is needed, use the `llama-cpp-rs` Rust bindings:

```toml
# Cargo.toml
[dependencies]
llama-cpp-2 = "0.1"
actix-web = "4"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

This allows replacing the v1 ONNX Runtime Rust binary with a llama.cpp-backed binary while keeping the same WebSocket/REST API surface.

### 8.3 Inference Architecture (Updated from v1)

```
┌──────────────────────────────────────────────────────────────┐
│                     Client Application                        │
│  (Text editor plugin / browser extension / OS input method)  │
└──────────────────────┬───────────────────────────────────────┘
                       │  WebSocket / local REST (loopback)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                Morpheus v2 Inference Server                   │
│          (llama.cpp server or custom Rust binary)             │
│                                                               │
│   ┌────────────────────────────────────────────────────┐     │
│   │             Trigger Heuristic Layer                  │     │
│   │  • Fire on: ≥50ms keystroke pause  OR  spacebar    │     │
│   │  • Debounce: suppress during rapid burst typing    │     │
│   └────────────────────┬───────────────────────────────┘     │
│                        │                                      │
│   ┌────────────────────▼───────────────────────────────┐     │
│   │             Tokenization Layer                      │     │
│   │  • SentencePiece Unigram model (32k vocab)         │     │
│   │  • Phrase-level MWE tokens (1,000 injected entries)│     │
│   └────────────────────┬───────────────────────────────┘     │
│                        │                                      │
│   ┌────────────────────▼───────────────────────────────┐     │
│   │           Mamba-2 Language Model (GGUF)             │     │
│   │  • 200M params, Q5_K_M quantized (~145 MB)         │     │
│   │  • Constant-size hidden state (no KV cache!)       │     │
│   │  • Autoregressive: generate 3-5 tokens             │     │
│   └────────────────────┬───────────────────────────────┘     │
│                        │                                      │
│   ┌────────────────────▼───────────────────────────────┐     │
│   │            Detokenization + Ranking                  │     │
│   │  • Beam search / top-k sampling over suggestions   │     │
│   │  • Detokenize subwords → surface Basque words      │     │
│   │  • Return top-3 ranked completions                 │     │
│   └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

**Key difference from v1:** The BoW Context Encoder is completely eliminated. Mamba processes the full token context natively through its recurrent state — no lossy averaging of embeddings. This should dramatically improve contextual predictions, especially for morphological suffix agreement.

---

## 9. Phased Execution Plan

### Phase 1: Foundation (Days 1-2)

| Task | Output |
|------|--------|
| Set up Docker training environment | `Dockerfile.train`, verified GPU access |
| Verify `mamba-ssm` installation on L40 | Smoke test passes |
| Pre-tokenize corpus | `data/train_tokens.npy`, `data/valid_tokens.npy` |
| Implement Dataset + DataLoader | `src/dataset.py` |
| Implement training script | `train.py` |
| Run Morpheus-Small (130M) for 1 epoch | Verify training loop, loss decreasing |

**Milestone:** Training loop produces decreasing loss on Morpheus-Small.

### Phase 2: Training (Days 3-6)

| Task | Output |
|------|--------|
| Start Morpheus-Base (200M) training | Training running on L40 |
| Monitor loss curves, LR, grad norms | W&B dashboard |
| Run eval every 1000 steps | Valid PPL tracking |
| Checkpoint every 5000 steps | `checkpoints/step_*.pt` |

**Milestone:** Valid PPL < 50 achieved (target: 30-50 range).

### Phase 3: Evaluation (Day 6-7)

| Task | Output |
|------|--------|
| Run full eval suite on best checkpoint | Top-1, Hit@3, Hit@5 metrics |
| Qualitative morphological review | Manual review of suffix agreement predictions |
| Compare with v1 LSTM results | Side-by-side comparison table |

**Milestone:** Hit@3 > 40% on test set.

### Phase 4: Deployment (Day 7-8)

| Task | Output |
|------|--------|
| Export best checkpoint to HuggingFace format | `exports/morpheus-v2-mamba-hf/` |
| Convert to GGUF via llama.cpp | `morpheus-v2-f16.gguf` |
| Quantize to Q5_K_M (and Q4_K_M fallback) | `morpheus-v2-Q5_K_M.gguf` |
| Validate quantized PPL ≤ 3% degradation | Quantization quality gate |
| Benchmark CPU inference latency | P90 latency report on target hardware |
| Integrate with demo server | Working autocomplete demo |

**Milestone:** End-to-end demo with < 100ms suggestion latency on consumer CPU.

### Phase 5: Polish (Day 8-10)

| Task | Output |
|------|--------|
| If 200M too slow → try 130M variant | Latency fallback |
| If quality insufficient → try 370M variant | Quality escalation |
| Tune beam search / top-k / temperature | Optimal sampling parameters |
| Document final architecture decisions | Updated ADRs |
| Benchmark against Morpheus v1 | Final comparison report |

**Milestone:** Production-ready model meeting all v2 targets.

---

## Appendix A: Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.4 | Deep learning framework |
| `mamba-ssm` | ≥ 2.2 | Mamba-2 architecture + CUDA kernels |
| `causal-conv1d` | ≥ 1.4 | Optimized 1D convolution for Mamba |
| `sentencepiece` | ≥ 0.2 | Tokenizer (reuse v1) |
| `safetensors` | ≥ 0.4 | Model weight serialization |
| `llama.cpp` | Latest | GGUF conversion + CPU inference |
| `wandb` | Latest | Experiment tracking |

## Appendix B: Key References

| Topic | Source |
|-------|--------|
| Mamba-1 paper | Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) |
| Mamba-2 paper | Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality" (2024) |
| Official code | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| PyPI package | [mamba-ssm](https://pypi.org/project/mamba-ssm/) |
| llama.cpp Mamba support | [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) (native Mamba/Mamba-2 GGUF) |
| LFM-2.5-230M (reference SSM deployment) | [Liquid AI](https://liquid.ai) |
| Morpheus v1 architecture | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| Morpheus v2 research doc | [Morpheus_v2_RESEARCH.md](./Morpheus_v2_RESEARCH.md) |
