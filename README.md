# Morpheus v2 Mamba — On-Device Basque Autocomplete

> **20× capacity boost over v1.** Mamba-2 State Space Model, ~200M parameters.
>
> **Goal:** P90 ≤ 50ms inference latency on consumer CPU. On-device, zero network calls.
>
> **Hardware:** 1× NVIDIA L40 (48 GB) for training; consumer x86-64 CPU for inference.

---

## What This Is

Morpheus v2 replaces the v1 9.5M-parameter LSTM with a **~200M-parameter Mamba-2 State Space Model** (SSM). Mamba shares the key inference property of LSTMs — constant-size hidden state, no KV cache — while dramatically improving language modeling capacity through selective state updates and parallelizable training.

**Key improvements over v1:**
- **20× more sequence-modeling capacity** (176M in Mamba layers vs 1.3M in v1 LSTM)
- **No BoW encoder** — Mamba processes full token context natively through its recurrent state
- **Native context handling** — eliminates the context-OOD bug that plagued v1 (ADR-018)
- **Modern SSM architecture** — 6-8× faster training than equivalent Transformers

## Quick Links

| Document | Purpose |
|----------|---------|
| [`Morpheus_v2_Mamba.md`](./Morpheus_v2_Mamba.md) | **Full implementation guide** — architecture, training, deployment |
| [`SETUP.md`](./SETUP.md) | Environment, dependencies, and first-run instructions |
| [`DATA.md`](./DATA.md) | Corpus strategy, data pipeline, and tokenizer reuse from v1 |
| [`DECISIONS.md`](./DECISIONS.md) | Architecture Decision Records for v2 |
| [`PLAN.md`](./PLAN.md) | Phased execution plan with milestones and acceptance criteria |

## Architecture at a Glance

```
User Keystrokes
      │
      ▼
 Trigger Heuristic (≥50ms pause OR spacebar)
      │
      ▼
 SentencePiece Unigram Tokenizer ←── Reused from v1 (32k vocab)
      │
      ▼
 Mamba-2 Language Model (200M params, Q5_K_M GGUF ~145 MB)
      │  • Selective state updates (input-dependent gating)
      │  • Constant-size hidden state — no KV cache
      │  • Autoregressive: generate 3-5 tokens
      │
      ▼
 Top-3 Suggestions → Client  (< 50ms wall-clock)
```

**Inference runtime:** llama.cpp with GGUF quantization (not ONNX — Mamba's selective scan has no efficient ONNX equivalent).

## v1 vs v2 Comparison

| Dimension | v1 LSTM | v2 Mamba-2 |
|-----------|---------|------------|
| Architecture | BoW + 2-layer LSTM | 32-layer Mamba-2 SSM |
| Parameters | 9.5M (86% in embedding) | ~207M (85% in SSM layers) |
| Context encoding | Lossy BoW averaging | Native recurrent state |
| Training | `torch` + ONNX export | `mamba-ssm` + llama.cpp GGUF |
| Inference | ONNX Runtime INT8 | llama.cpp Q5_K_M |
| Model size on disk | ~12 MB (INT8) | ~145 MB (Q5_K_M) |
| v1 valid PPL (epoch 2) | 120 | Target: < 50 |
| Deployment format | ONNX `.onnx` | GGUF `.gguf` |

## Repository Relationship

```
itzune/
├── morpheus/              # v1 LSTM — baseline and data/tokenizer provider
└── morpheus-mamba/        # v2 Mamba-2 — this repo
```

**Reused from v1:**
- SentencePiece tokenizer: `tokenizer/basque_unigram_32k.model`
- Data splits: `data/splits/{train,valid,test}/`
- Documentation: `DATA.md` corpus strategy, `RESEARCH.md` linguistic analysis

**Not reused:**
- All model code (LSTM → Mamba)
- All training scripts
- ONNX export/quantization pipeline (replaced by llama.cpp GGUF)
- Inference server (ONNX Runtime → llama.cpp)

## Quick Start

```bash
# 1. Set up environment
# See SETUP.md for full Docker-based GPU training setup

# 2. Pre-tokenize corpus (reusing v1 tokenizer + data)
python scripts/pretokenize.py \
    --sp-model tokenizer/basque_unigram_32k.model \
    --input-dir ../morpheus/data/splits/train \
    --output data/train_tokens.npy

# 3. Train Morpheus-Small (130M) for fast iteration
python train.py --config config/small.yaml

# 4. Train Morpheus-Base (200M) for production
python train.py --config config/base.yaml

# 5. Export to GGUF for CPU inference
python scripts/export_hf.py --checkpoint checkpoints/best.pt
# Then use llama.cpp: convert_hf_to_gguf.py → llama-quantize

# 6. Run inference server
./llama-server -m exports/morpheus-v2-Q5_K_M.gguf --host 127.0.0.1 --port 8080
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model architecture | Mamba-2 (via `mamba-ssm` package) |
| Training | PyTorch 2.4+ with CUDA 12.4 |
| Data pipeline | Memory-mapped numpy arrays |
| Tokenization | SentencePiece Unigram (reused from v1) |
| Model export | PyTorch → HuggingFace safetensors → llama.cpp GGUF |
| CPU inference | llama.cpp with Q5_K_M quantization |
| Experiment tracking | W&B |
| Training hardware | 1× NVIDIA L40 (48 GB VRAM) |
| Inference hardware | Consumer x86-64 CPU (Intel i7-13700 / AMD Ryzen 7 7700 class) |

## Target Metrics

| Metric | v1 (epoch 2) | v2 Target |
|--------|-------------|-----------|
| Valid perplexity | 120 | < 50 |
| Top-1 accuracy | 2.0% | > 20% |
| Hit@3 | 4.0% | > 40% |
| Hit@5 | — | > 50% |
| Inference P90 (per token) | — | ≤ 50ms |
| Post-quantization PPL degradation | — | ≤ 3% |
| Model params | 9.5M | ~200M |
| Model size on disk | ~12 MB | ~145 MB |
