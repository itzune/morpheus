# DECISIONS.md — Architecture Decision Records for Morpheus v2

This document records every major technical decision for Morpheus v2 Mamba. Decisions are immutable once recorded. v1 ADRs ([`../morpheus/DECISIONS.md`](../morpheus/DECISIONS.md)) are referenced where applicable.

---

## V2-001: Architecture — Mamba-2 State Space Model over Transformer or xLSTM

**Status:** Accepted

**Context:**

v1's 9.5M-parameter LSTM achieved valid PPL 120 after 2 epochs — far from the < 50 target. The root cause was capacity starvation: only 1.3M of 9.5M parameters (14%) were doing sequence modeling; 86% was in the embedding table.

Three candidate architectures were evaluated for v2 (see [`Morpheus_v2_RESEARCH.md`](../morpheus/Morpheus_v2_RESEARCH.md)):

- **Path A — xLSTM:** Modern LSTM variant with matrix memory and exponential gating. Better than vanilla LSTM but fundamentally still an RNN — inference is O(1) per step but expressiveness is limited at scale compared to SSMs and Transformers. Rejected: maximum capacity on L40 hardware is ~150M params (due to sequential training), below our 200M target.

- **Path B — Distilled Transformer:** Leverages pre-trained Basque models (Latxa) via distillation. Latxa uses Llama architecture — KV cache bottleneck on CPU makes P90 ≤ 50ms infeasible without extreme distillation that destroys quality. Rejected: inference latency risk is too high.

- **Path C — Mamba-2 SSM:** Selective State Space Model with O(n) training via parallel scan, O(1) inference per step via recurrent state, no KV cache. ~200M params easily fits on L40. Mamba-2's SSD algorithm is optimized for tensor cores, providing 6-8× Transformer training throughput.

**Decision:**

Use **Mamba-2** (Structured State Space Duality) as implemented in the `mamba-ssm` package. Full implementation guide: [`Morpheus_v2_Mamba.md`](./Morpheus_v2_Mamba.md).

**Rationale:**
- Constant-size hidden state → O(1) per-step inference (no KV cache)
- Parallelizable training → full GPU utilization on L40
- ~200M params at Q5_K_M → ~145 MB on disk, fits in consumer RAM
- Mamba's selective state updates naturally handle Basque's long-range morphological dependencies
- llama.cpp has native Mamba-2 support with optimized CPU kernels

**Consequence:**
- Complete departure from v1 ONNX Runtime pipeline → llama.cpp GGUF deployment
- Training requires GPU with CUDA 12.4 (L40)
- Export chain: PyTorch → HuggingFace safetensors → llama.cpp GGUF
- Training from scratch (~3-5 days on L40 for 200M model)
- No weight tying needed (Mamba allocates parameters efficiently across layers)

**References:**
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Dao & Gu, "Transformers are SSMs" (2024)
- [`Morpheus_v2_Mamba.md`](./Morpheus_v2_Mamba.md) §1 — Architecture Overview

---

## V2-002: Model Configuration — 200M Target with Three Variants

**Status:** Accepted

**Context:**

The Mamba-2 paper provides reference configurations at 130M, 370M, 780M, 1.4B, and 2.8B parameters. The L40's 48 GB VRAM constrains maximum model size during training. The 50ms P90 latency target constrains maximum model size during inference.

**Decision:**

Three configuration tiers, with **Morpheus-Base (200M)** as the primary target:

| Variant | d_model | n_layer | d_state | ~Params | Use case |
|---------|---------|---------|---------|---------|----------|
| Morpheus-Small | 768 | 24 | 64 | ~130M | Fast iteration, baseline |
| Morpheus-Base | 960 | 32 | 64 | ~200M | **Primary production target** |
| Morpheus-Large | 1024 | 48 | 128 | ~370M | Quality escalation, if latency allows |

Start with Small for rapid experimentation (1 epoch in ~8 hours), then scale to Base for production.

**Rationale:**
- 200M provides sufficient capacity for Basque morphology (20× v1's 1.3M sequence-modeling params)
- Fits in L40 VRAM during training (~15-20 GB at batch 128, seq 512)
- At Q5_K_M, ~145 MB on disk — fits in consumer CPU memory
- Estimated CPU inference at 50-80 tok/s → 12-20 ms per token → within 50ms budget for 3-token suggestions

**Consequence:**
- Three YAML configs in `config/{small,base,large}.yaml`
- Training starts with Small for pipeline validation
- Base is the production target; Large is the fallback if Base quality is insufficient

---

## V2-003: Inference Runtime — llama.cpp with GGUF over ONNX Runtime

**Status:** Accepted

**Context:**

v1 used ONNX Runtime with INT8 dynamic quantization for CPU inference. Mamba's selective scan operator has no efficient ONNX equivalent — decomposing it into ONNX primitives results in orders-of-magnitude slowdown (documented in [`Morpheus_v2_Mamba.md`](./Morpheus_v2_Mamba.md) §7.1).

**Decision:**

Use **llama.cpp** with GGUF quantization for CPU inference. llama.cpp has native Mamba/Mamba-2 architecture support with hand-optimized CPU kernels.

**Alternatives considered:**

- ONNX Runtime (v1 pipeline): Rejected — no efficient selective scan primitive. "Do NOT use the v1 ONNX pipeline for Mamba."
- Candle (Rust/WASM): Viable for browser deployment but slower than llama.cpp on native x86-64. Consider for Phase 5 if browser-based deployment is needed.
- Custom CUDA inference: Rejected — GPU not available on target consumer hardware.

**Rationale:**
- llama.cpp is the definitive standard for CPU-bound model inference
- Native Mamba-2 support with AVX2/AVX-512 optimized kernels
- Q5_K_M quantization provides ~65% size reduction with < 3% PPL degradation
- Memory-mapped model loading (mmap) — near-zero startup time
- Built-in HTTP server mode for easy integration with existing demo

**Consequence:**
- Export chain: PyTorch checkpoint → HuggingFace safetensors → GGUF (via `convert_hf_to_gguf.py`) → Q5_K_M quantized (via `llama-quantize`)
- Inference server: llama.cpp server mode or custom Rust binary with `llama-cpp-rs`
- ONNX Runtime dependency removed entirely from v2

**References:**
- [`Morpheus_v2_Mamba.md`](./Morpheus_v2_Mamba.md) §7 — Deployment: Export & CPU Inference
- [llama.cpp Mamba support](https://github.com/ggerganov/llama.cpp)

---

## V2-004: Training Precision — BF16 with FP32 Residual Stream

**Status:** Accepted

**Context:**

Mamba's selective scan involves exponential operations (Δ discretization, A_bar computation) that can amplify numerical instability. FP16 is insufficient for stable training; FP32 is too slow and memory-intensive.

**Decision:**

Train in **bfloat16** with the **residual stream kept in FP32** (`residual_in_fp32=True`). This is the Mamba-2 recommended configuration.

**Rationale:**
- BF16 has the same exponent range as FP32 (8 bits) — critical for exponential stability
- FP32 residual stream prevents error accumulation across 32 layers
- `residual_in_fp32` adds < 5% memory overhead for significant stability improvement
- L40 has native BF16 tensor core support (Ada Lovelace architecture)

**Consequence:**
- `dtype=torch.bfloat16` for model parameters and activations
- `residual_in_fp32=True` in MambaConfig
- Gradient clipping at 1.0 (critical for SSM stability — see V2-006)

---

## V2-005: Sequence Length — 512 over 128

**Status:** Accepted

**Context:**

v1 trained at `seq_len=128`. Mamba v2 can handle longer contexts efficiently (O(n) inference, not O(n²)). Longer context windows improve Basque suffix agreement prediction by giving the model more morphological context.

**Decision:**

Train at **`seq_len=512`** for v2.

**Rationale:**
- 128 tokens ≈ 75 Basque words (at fertility 1.7) — insufficient for multi-sentence context
- 512 tokens ≈ 300 Basque words — covers most email paragraphs
- Mamba's O(n) inference means longer context adds linear cost, not quadratic
- Memory increase from 128→512 is ~4× (sequential, not quadratic)
- At batch_size=128, seq_len=512: ~65K tokens per batch — fits in L40 VRAM at BF16

**Consequence:**
- Training file (train_tokens.npy) is ~4 GB (uint16) — fits in RAM
- Dataset class uses `seq_len=512` by default
- Checkpoint memory increases proportionally (still < 1 GB for 200M model at BF16)

---

## V2-006: Gradient Clipping — 1.0 Mandatory

**Status:** Accepted (non-negotiable)

**Context:**

SSM models can experience gradient spikes, especially in early training when the discretization parameters (Δ) are not yet calibrated. Mamba's selective scan computes `A_bar = exp(Δ * A)` — if Δ is large, this produces exponentials that blow up gradients.

**Decision:**

Apply **gradient clipping at 1.0** (`grad_clip=1.0`). This is mandatory — not optional — for stable Mamba training.

**Rationale:**
- Documented in Mamba-2 training best practices
- Prevents NaN losses from exponential blow-up in selective scan
- Particularly important during warmup when Δ is being calibrated
- 1.0 is the recommended value from the Mamba authors

**Consequence:**
- `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` after every backward pass
- Training script includes this by default
- If NaN losses still occur, reduce learning rate before relaxing gradient clip

---

## V2-007: Repository Strategy — Separate Repo from v1

**Status:** Accepted

**Context:**

Morpheus v2 Mamba shares nothing at the code level with v1 LSTM: different architecture, different training framework, different inference runtime, different export pipeline. The reusable assets are data (read-only inputs) and the tokenizer.

**Decision:**

Create a **separate repository** (`morpheus-mamba`) at the same directory level as `morpheus`. Symlink the tokenizer and data splits from v1.

**Rationale:**
- Clean separation of concerns — v1 remains as baseline/evaluation reference
- Different dependencies (mamba-ssm, llama.cpp vs onnx, onnxruntime)
- Different build pipelines (Dockerfile.train with CUDA vs uv-managed Python)
- Git history clarity — v2 is a complete rewrite, not an incremental change
- v1 remains the fallback if v2 latency targets aren't met

**Consequence:**
```
itzune/
├── morpheus/        # v1 LSTM (unchanged, baseline)
└── morpheus-mamba/  # v2 Mamba-2 (this repo)
```

---

## V2-008: Tokenizer Reuse — v1 SentencePiece without Retraining

**Status:** Accepted

**Context:**

The v1 tokenizer (32K Unigram, fertility 1.71) was trained on 5.45M domain-filtered Basque documents. Retraining on the same data would produce an identical model. The tokenizer format (SentencePiece `.model`) is architecture-agnostic.

**Decision:**

Reuse the v1 SentencePiece model as-is. No retraining. No vocabulary modification.

**Rationale:**
- Fertility 1.71 already meets the ≤ 2.0 target
- 32K vocabulary provides sufficient morphological coverage
- MWE tokens (1,000 injected entries) are already included
- Retraining would produce identical results and waste compute
- The tokenizer is a statistical model on the data, not tied to any neural architecture

**Consequence:**
- Copy or symlink `tokenizer/basque_unigram_32k.model` from v1
- No tokenizer training step in v2 pipeline
- Tokenizer validation: verify fertility and vocabulary size match v1

---

## V2-009: Quantization Floor — Q5_K_M Minimum

**Status:** Accepted

**Context:**

v1's ADR-008 established INT8 as the minimum quantization floor for Basque — INT4 and below destroy morphological suffix agreement. v2 uses llama.cpp GGUF quantization, which offers multiple K-Quant levels.

**Decision:**

Use **Q5_K_M as the minimum quantization level** for production deployment. Q4_K_M is acceptable as a latency fallback only if Q5_K_M fails the 50ms P90 target.

**Rationale:**
- Q5_K_M provides ~65% size reduction vs FP16 while preserving > 97% of quality
- Q4_K_M provides ~75% reduction but risks Basque morphological degradation (same underlying problem as v1 ADR-008)
- Q6_K or Q8_0 available as quality escalation if Q5_K_M degrades > 3% PPL
- Mixed precision (5-bit for FFN, 6-bit for attention-like components) naturally protects sensitive layers

**Consequence:**
- Post-quantization validation gate: PPL degradation ≤ 3% vs FP16
- If Q5_K_M fails → try Q6_K → if still fails → investigate training issues (not quantization)
- Q4_K_M reserved as emergency latency fallback only

**References:**
- v1 ADR-008 — Minimum Quantization Floor
- [`Morpheus_v2_Mamba.md`](./Morpheus_v2_Mamba.md) §7.2-7.3

---

## V2-010: Experiment Tracking — W&B Required

**Status:** Accepted

**Context:**

v2 training runs 3-5 days on expensive GPU hardware. We need to monitor loss curves, gradient norms, LR schedule, and validation metrics in real-time to catch issues early.

**Decision:**

Log all training metrics to **Weights & Biases**. At minimum: train loss, valid PPL, learning rate, gradient norm, tokens/sec, GPU memory usage.

**Rationale:**
- 3-5 day training runs are too long for manual log-file inspection
- Real-time alerts for NaN losses or gradient spikes
- Automatic comparison between Small/Base/Large runs
- Standard tooling; no infrastructure cost for single-user projects

**Consequence:**
- `wandb.init(project="morpheus-v2-mamba")` in training script
- Log interval every 50 steps
- Config logged at startup (hyperparameters, model architecture)
- Checkpoints saved independently of W&B (local filesystem)
