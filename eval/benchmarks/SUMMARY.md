# Morpheus v2 Inference Benchmarks

**Date**: 2026-07-12  
**Model**: Morpheus v2 Mamba-2 (91M parameters, 4K Unigram vocabulary)  
**Checkpoint**: step 74K (best, valid PPL = 7.13)

## Hardware Tested

| Label | Hardware | Specs |
|-------|----------|-------|
| L40 GPU | NVIDIA L40 | 46 GB VRAM, server GPU |
| EPYC CPU | AMD EPYC 9474F | 48-core server CPU (8 cores allocated), 3.65 GHz |
| i7 CPU | Intel i7-8550U | 4 cores / 8 threads, 1.80 GHz base / 4.0 GHz turbo (2017 consumer laptop) |

## Results Summary

| Hardware | Quant | Model (MB) | RAM (MB) | VRAM (MB) | Decode (tok/s) | Prefill (tok/s) | Latency (ms) |
|----------|-------|-----------|----------|-----------|----------------|-----------------|-------------|
| L40 GPU | Q5_K_M | 63 | 336 | 607 | 277 | 3,891 | 35.3 |
| L40 GPU | Q4_K_M | 52 | 326 | 597 | 282 | 4,594 | 32.6 |
| EPYC 9474F CPU | Q5_K_M | 63 | 350 | — | 440 | 3,215 | 32.2 |
| i7-8550U CPU (4t) | Q5_K_M | 63 | 119 | — | 224 | 328 | 115.1 |
| i7-8550U CPU (4t) | Q4_K_M | 52 | 155 | — | 318 | 459 | 97.1 |
| i7-8550U CPU (8t) | Q4_K_M | 52 | 155 | — | 235 | 312 | 129.4 |

## Key Findings

### 1. Consumer laptop is viable for real-time autocomplete
The i7-8550U (a 2017 consumer laptop CPU) achieves **224–318 tok/s** decode speed and **97–115 ms** autocomplete latency. Human typing speed is ~5 chars/s (~1.5 tokens/s), so the model generates tokens 150–200× faster than a human types. Even the worst case (Q5_K_M, 4 threads: 115 ms latency) is well under the 200 ms threshold for perceived "instant" response.

### 2. Server CPU outperforms GPU for decode
The EPYC 9474F CPU achieves **440 tok/s** — faster than the L40 GPU (277 tok/s). This is because Mamba-2's state size is tiny (~16 KB per layer) compared to transformer KV cache (MB-scale). For a 91M parameter model, GPU kernel launch overhead dominates actual computation time. The model is too small to benefit from massive GPU parallelism.

### 3. Q4 quantization is significantly faster on CPU
Q4_K_M is 41% faster than Q5_K_M on the i7 (318 vs 224 tok/s). On GPU, quantization makes minimal difference (282 vs 277 tok/s) — GPU memory bandwidth is not the bottleneck at this scale.

### 4. Hyperthreading hurts performance
8 threads is slower than 4 threads on the i7 (235 vs 318 tok/s, 38% slower). This is a memory-bound workload where hyperthreading causes L1/L2 cache contention. The optimal thread count equals physical core count.

### 5. Memory footprint is minimal
- **Model file**: 52–63 MB (fits in L2 cache of modern CPUs)
- **Process RSS**: 119–350 MB (well under 1 GB)
- **GPU VRAM**: 597–607 MB (dominated by CUDA context, not model weights)
- No KV cache growth: Mamba-2 uses O(1) memory per token (fixed state), unlike transformers which grow linearly with context length

### 6. Decode speed is context-length independent
Mamba-2's SSM state is fixed-size (~16 KB per layer regardless of context length). Decode speed remains constant whether the context is 4 tokens or 129 tokens. This is a critical advantage for text editors where context grows over a long editing session.

## Methodology

- **Inference engine**: llama.cpp (llama-server), build includes commit `dc2187d48` (SSM_SCAN fix for Mamba-2)
- **Benchmark script**: `scripts/benchmark_inference.py`
- **Metrics**: Uses llama-server's built-in `/completion` endpoint timing data (`prompt_ms`, `predicted_ms`, `prompt_n`, `predicted_n`)
- **Prompts**: Basque text at 4 context lengths (4, 18, 40, 129 tokens) + 5 autocomplete scenarios (3–5 token generation)
- **Runs**: 5 per configuration, averaged
- **Latency**: Wall-clock time for short-prompt autocomplete (HTTP round-trip + prefill + decode)
