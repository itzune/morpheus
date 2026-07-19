# Latxa 8B vs Morpheus — Autocomplete Eval

**Date:** 2026-07-19
**Server:** NVIDIA L40 (46GB), `10.2.121.210`, CUDA llama-server v3
**Same-day, same-hardware, same-test-set comparison**

## Models

| | Morpheus | Latxa 8B |
|---|---|---|
| Architecture | Mamba-2 SSM (91M) | Llama-3.1 transformer (8B) |
| Quant | Q5_K_M (64MB) | Q6_K (6.6GB) |
| Backend | `morpheus-sp-fim` (SentencePiece 4K) | `llama-fim` (Llama-3 BPE) |
| Origin | Trained from scratch on 4.62B Basque tokens | HiTZ CPT of Llama-3.1-8B on 4.2B Basque tokens |
| Port | 9091 | 9090 |
| Offload | 99 layers GPU | 99 layers GPU |

## CSR (Character Savings Rate) — the product metric

**Same 30 tests** from `eval/targets.json` (Wikipedia CSR short+long).
Algorithm: free-acceptance (Trnka & McCoy 2008), 1 token/step.

| Model | Macro CSR | Micro CSR | Keystrokes saved | Accept rate |
|-------|-----------|-----------|------------------|-------------|
| Morpheus v3_fim Q5_K_M | 24.83% | 25.63% | 307 / 1198 | 27.5% (245/891) |
| **Latxa 8B Q6_K** | **33.18%** | **32.97%** | **395 / 1198** | **40.2% (323/803)** |
| *Δ* | *+8.35 pts* | *+7.34 pts* | *+88 chars* | *+12.7 pts* |

Latxa saves **~1/3 more keystrokes** than Morpheus on identical test sentences.

## Real-corpus qualitative — same 40 prompts (Wikipedia + Berria), 8 tokens

| Signal | Morpheus | Latxa 8B |
|--------|----------|----------|
| Nonempty suggestions | 100% (40/40) | 100% (40/40) |
| Digit artifacts | 0% | 0% |
| First-word exact match | 10% (4/40) | 10% (4/40) |
| Any overlap (1 of 3 words) | 25% (10/40) | **32.5% (13/40)** |
| Prefix on track | 2% (1/40) | 0% (0/40) |
| Avg confidence | 0.347 | **0.491** |

## Latency & resource footprint

Structured benchmark (`scripts/bench_latency.py`): 6 prompts × reps, 8 tokens
per request, through the full demo stack (`/api/autocomplete/greedy`), with
concurrent `ps`/`nvidia-smi` sampling of the model server process. Run on two
hosts: the GPU server (NVIDIA L40) and a localhost laptop (CPU-only).

### GPU server — NVIDIA L40 (46 GB VRAM, AMD EPYC 9474F, 30 GB RAM)

Both models offloaded `-ngl 99`, served simultaneously.

| Model | mean | p50 | p95 | min | max | tok/s | VRAM | host CPU | RSS |
|-------|-----:|-----:|-----:|----:|----:|------:|-----:|----:|----:|
| Morpheus Q5_K_M (91M) | 75 ms | 66 ms | 120 ms | 43 ms | 144 ms | 106.0 | 602 MiB | 24% | 1535 MiB |
| **Latxa 8B Q6_K** | **115 ms** | 116 ms | 121 ms | 106 ms | 125 ms | 69.5 | **6988 MiB** | 3% | 1073 MiB |

GPU utilization (shared): mean 32%, peak 78%. Both models stay well under the
150 ms autocomplete threshold. Latxa's distribution is tighter (p95 121 vs
120 ms — essentially equal) because the L40 saturates only at the larger model's
batch; Morpheus's variance comes from the demo's Python-side SentencePiece
encoding + retokenization-fallback path (also visible as its higher CPU%, 24%
vs 3%). Latxa's `llama-fim` backend does string passthrough, so its model
server is nearly idle on CPU — the L40 does the work.

### Localhost laptop — Intel i7-8550U 4c/8t @1.8 GHz, 15 GB RAM, **no GPU**

One model at a time, `-ngl 0 -c 2048 -t 8` (pure CPU).

| Model | mean | p50 | p95 | tok/s | RAM (RSS) | host CPU |
|-------|-----:|-----:|-----:|------:|----------:|----:|
| Morpheus Q5_K_M (91M) | 196 ms | 165 ms | 343 ms | 40.7 | 266 MiB | 86% |
| **Latxa 8B Q6_K** | **2869 ms** | 2796 ms | 3356 ms | **2.8** | **6648 MiB** | 534% |

### What the numbers say

1. **On GPU, both are viable autocomplete models.** Latxa costs +40 ms and
   11.6× the VRAM (6988 vs 602 MiB) for +8.4 CSR points. At 46 GB the L40 has
   room for ~6 concurrent Latxa instances or ~75 Morpheus instances.
2. **On CPU, only Morpheus is viable.** Latxa 8B at 2.8 tok/s means a single
   token takes ~360 ms and an 8-token completion takes ~2.9 s — ~19× over the
   150 ms threshold — while pinning 5.3 cores and 6.6 GB of RAM. It is not a
   real-time autocomplete model off-Gpu. Morpheus on the same laptop does
   40.7 tok/s (196 ms for 8 tokens; ~25 ms/token), comfortably inside the
   budget.
3. **The size gap is the deployment gap.** Latxa is 98× larger on disk
   (6290 vs 64 MiB) and 25× the resident RAM on CPU (6648 vs 266 MiB). That is
   the cost of the +8 CSR points — and it is only payable where a GPU is
   present.

## Verdict

Latxa 8B **confirms the UX impression quantitatively**:
- **+8.4 CSR points** (33.2% vs 24.8%) — meaningful keystroke savings gain
- **+7.5 pts overlap** with gold continuations
- **+0.14 higher confidence** (0.49 vs 0.35)
- Clean output (0% digit artifacts, 0% filter gap — BPE is artifact-free)
- Latency still acceptable (~125ms on L40)

**Tradeoff:** Latxa is 98× larger on disk (6.6 GB vs 64 MB) and, on GPU, costs
+40 ms latency and 11.6× VRAM for +8 CSR points. Crucially, **Latxa is
GPU-bound**: on a CPU laptop it collapses to 2.9 s/request (2.8 tok/s) and 6.6 GB
RAM — unusable for real-time autocomplete. Morpheus runs on that same laptop at
40.7 tok/s. This concretely fixes the two-tier split: **Morpheus is the
on-device/edge model (CPU-deployable); Latxa is the server-side model
(GPU-only)**. The +8 points and cleaner output justify the server deployment
where a GPU is present — and this is the **base model with no FIM training**.
A FIM fine-tune should close the remaining gap (FIM infill is currently
non-functional on Latxa; AR append works).

## Raw result files

- CSR Latxa: `20260719_084945_HiTZ.Latxa-Llama-3.1-8B.Q6_K_csr/`
- CSR Morpheus: `20260719_085248_v3_fim.Q5_K_M_csr/`
- Real-corpus Latxa: `20260719_085007_HiTZ.Latxa-Llama-3.1-8B.Q6_K_realcorpus/`
- Real-corpus Morpheus: `20260719_085039_v3_fim.Q5_K_M_realcorpus/`
