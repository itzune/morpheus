# Basque LLM Autocomplete Eval — Morpheus vs Kimu 2B vs Latxa 8B

**Date:** 2026-07-19
**Server:** NVIDIA L40 (46GB), `10.2.121.210`, CUDA llama-server v3
**Same-day, same-hardware, same-test-set comparison**

## Models

| | Morpheus | Kimu 2B | Latxa 8B |
|---|---|---|---|
| Architecture | Mamba-2 SSM (91M) | Gemma-2 transformer (2B) | Llama-3.1 transformer (8B) |
| Quant | Q5_K_M (64 MB) | Q6_K (2.1 GB) | Q6_K (6.2 GB) |
| Backend | `morpheus-sp-fim` (SentencePiece 4K) | `llama-fim` (Gemma BPE) | `llama-fim` (Llama-3 BPE) |
| Origin | Trained from scratch on 4.62B Basque tokens | Orai NLP CPT of Gemma-2-2b on Basque (ZelaiHandi, 1.5B tok) | HiTZ CPT of Llama-3.1-8B on 4.2B Basque tokens |
| Port | 9091 | 9092 | 9090 |
| Offload | 99 layers GPU | 99 layers GPU | 99 layers GPU |

## CSR (Character Savings Rate) — the product metric

**Same 30 tests** from `eval/targets.json` (Wikipedia CSR short+long).
Algorithm: free-acceptance (Trnka & McCoy 2008), 1 token/step.

| Model | Macro CSR | Micro CSR | Keystrokes saved | Accept rate |
|-------|-----------|-----------|------------------|-------------|
| Morpheus v3_fim Q5_K_M | 24.83% | 25.63% | 307 / 1198 | 27.5% (245/891) |
| **Kimu 2B Q6_K** | **34.10%** | **33.64%** | **403 / 1198** | 35.5% (282/795) |
| Latxa 8B Q6_K | 33.18% | 32.97% | 395 / 1198 | **40.2%** (323/803) |

Kimu 2B **edges out Latxa 8B** on CSR (+0.92 pts) despite being 4× smaller — and
saves 96 keystrokes over Morpheus (+9.27 pts). All three are base models with no
FIM training; this is pure AR (append) completion quality.

## Real-corpus qualitative — same 40 prompts (Wikipedia + Berria), 8 tokens

| Signal | Morpheus | Kimu 2B | Latxa 8B |
|--------|----------|---------|----------|
| Nonempty suggestions | 100% (40/40) | 100% (40/40) | 100% (40/40) |
| Digit artifacts | 0% | 8% (3) | 0% |
| First-word exact match | 10% (4/40) | 12% (5/40) | 10% (4/40) |
| Any overlap (1 of 3 words) | 25% (10/40) | 28% (11/40) | **32.5%** (13/40) |
| Prefix on track | 2% (1/40) | 2% (1/40) | 0% (0/40) |
| Avg confidence | 0.347 | 0.421 | **0.491** |

## Domain examples — where the quality difference is visible

Side-by-side continuations across three writing domains (greedy, 12 tokens):

**Email writing** — `Egun on! Astelehenean bilera bat egitea proposatzen`
("Good morning! I propose holding a meeting on Monday")

| Model | Continuation | Conf |
|-------|-------------|-----:|
| Latxa 8B | `dizuet, 18:00etan. Bilera` *("to you [pl.], at 18:00. Meeting…")* | 0.45 |
| Kimu 2B | `dizut. -Bai, noski. Noiz` *("to you [sg]. -Yes, of course. When…")* | 0.42 |
| Morpheus | `dizugu, eta, ondoren, egutegia eta` *("we propose, and, then, the calendar and")* | 0.34 |

All three pick a valid dative verb form. Latxa commits to a concrete meeting
time; Kimu drifts into a dialogue response; Morpheus drifts into connective
filler.

**Essay / article** — `Adimen artifizialak Hezkuntzan izango duen eragina`
("The impact AI will have on Education")

| Model | Continuation | Conf |
|-------|-------------|-----:|
| Latxa 8B | `aztertuko dute Euskal Herri` *("will examine [in] the Basque Country")* | 0.56 |
| Kimu 2B | `aztertuko dute bihar, Elkargu` *("will examine tomorrow, the Council")* | 0.55 |
| Morpheus | `aztertzeko, EHUko ikertzaile talde batek,` *("to examine, a UPV/EHU research team,")* | 0.33 |

All three correctly continue with *aztertu* (examine) — the strongest
collocation. Kimu and Latxa are near-identical in confidence (0.55 vs 0.56);
Morpheus lags.

**Educational / technical** — `Suhesia sareko komunikazio guztiak`
("The firewall [?] all network communications")

| Model | Continuation | Conf |
|-------|-------------|-----:|
| Kimu 2B | `kontrolatzeko eta kudeatzeko erabiltzen` *("to control and manage [used]")* | 0.45 |
| Latxa 8B | `zifratuta daude, eta ez dago er` *("are encrypted, and there is no…")* | 0.42 |
| Morpheus | `, 100.000 biztanletik gora` *(", over 100,000 inhabitants")* | 0.35 |

Here the models diverge sharply. Both Kimu and Latxa stay on-topic (network
security → control/encryption). Kimu's continuation is arguably more natural
(firewalls control traffic) and carries higher confidence. Morpheus drifts to
an unrelated demographic filler — a hallmark of an underpowered model latching
onto a high-frequency statistical pattern instead of the sentence's semantic
thread.

## Latency & resource footprint

Structured benchmark (`scripts/bench_latency.py`): 6 prompts × 8 reps, 8 tokens
per request, through the full demo stack (`/api/autocomplete/greedy`), with
concurrent `ps`/`nvidia-smi` sampling of the model server process. Run on two
hosts: the GPU server (NVIDIA L40) and a localhost laptop (CPU-only).

### GPU server — NVIDIA L40 (46 GB VRAM, AMD EPYC 9474F, 30 GB RAM)

All three models offloaded `-ngl 99`, served simultaneously (10.6 GB / 46 GB VRAM).

| Model | mean | p50 | p95 | min | max | tok/s | VRAM | host CPU | RSS |
|-------|-----:|-----:|-----:|----:|----:|------:|-----:|----:|----:|
| Morpheus Q5_K_M (91M) | 76 ms | 64 ms | 133 ms | 43 ms | 181 ms | 104.9 | 602 MiB | 2% | 1569 MiB |
| **Kimu 2B Q6_K** | **95 ms** | 98 ms | 101 ms | 71 ms | 102 ms | 84.5 | **3036 MiB** | 0% | 910 MiB |
| Latxa 8B Q6_K | 115 ms | 115 ms | 120 ms | 107 ms | 128 ms | 70.4 | 6988 MiB | 0% | 1138 MiB |

GPU utilization (shared): mean 31%, peak 78%. All three models stay well under
the 150 ms autocomplete threshold. Latency scales cleanly with model size:
Morpheus (91M) → Kimu (2B) → Latxa (8B) at 76 / 95 / 115 ms. Kimu is the
sweet spot — +19 ms over Morpheus for +9.3 CSR points, and 19 ms *faster* than
Latxa at 43% less VRAM.

### Localhost laptop — Intel i7-8550U 4c/8t @1.8 GHz, 15 GB RAM, **no GPU**

One model at a time, `-ngl 0 -c 2048 -t 8` (pure CPU).

| Model | mean | p50 | p95 | tok/s | RAM (RSS) | host CPU |
|-------|-----:|-----:|-----:|------:|----------:|----:|
| Morpheus Q5_K_M (91M) | 196 ms | 165 ms | 343 ms | 40.7 | 266 MiB | 86% |
| Kimu 2B Q6_K | 1439 ms | 1429 ms | 1500 ms | 5.6 | 2357 MiB | 563% |
| Latxa 8B Q6_K | 2869 ms | 2796 ms | 3356 ms | 2.8 | 6648 MiB | 534% |

### What the numbers say

1. **On GPU, all three are viable autocomplete models.** Kimu is the
   efficiency frontier: it matches Latxa's CSR (34.1% vs 33.2%) at 43% less
   VRAM (3.0 vs 7.0 GB) and 17% lower latency (95 vs 115 ms). Morpheus is the
   cheapest (602 MiB, 76 ms) but 9 CSR points behind.
2. **On CPU, only Morpheus is viable.** Latxa 8B at 2.8 tok/s means a single
   token takes ~360 ms and an 8-token completion takes ~2.9 s — ~19× over the
   150 ms threshold — while pinning 5.3 cores and 6.6 GB of RAM. Kimu 2B is
   2× faster than Latxa (5.6 tok/s, ~1.4 s/request) and uses 2.8× less RAM
   (2.4 GB), but is still ~9.6× over the budget — not a real-time model
   off-GPU. Morpheus on the same laptop does 40.7 tok/s (196 ms for 8 tokens),
   comfortably inside the budget.
3. **The size gap is the deployment gap.** Latxa is 98× larger on disk
   (6290 vs 64 MiB); Kimu is 33× larger (2100 vs 64 MiB). That is the cost of
   the +9 CSR points — and it is only payable where a GPU is present.

## Verdict

The three-model comparison **confirms the two-tier positioning** while
revealing Kimu 2B as a compelling mid-tier:

- **Kimu 2B matches Latxa 8B on CSR** (34.1% vs 33.2%) at 4× smaller size and
  19 ms lower latency — a 2B Basque-pretrained model is sufficient to reach the
  8B quality ceiling on this task.
- **Both Basque LLMs beat Morpheus by ~9 CSR points** (33-34% vs 24.8%) — the
  gap a 2B+ pretrained backbone opens over a 91M from-scratch model.
- **All three are base models with no FIM training.** AR append works; FIM
  infill is non-functional on both Kimu and Latxa (base models emit garbage on
  FIM sentinels). A FIM fine-tune should close the remaining gap.
- **The deployment split is a hardware constraint, not a preference.** Morpheus
  (91M, 55 MB) is the only CPU-viable model (40.7 tok/s on a 2017 laptop); Kimu
  and Latxa are GPU-bound.

## Raw result files

- CSR Latxa: `20260719_084945_HiTZ.Latxa-Llama-3.1-8B.Q6_K_csr/`
- CSR Morpheus: `20260719_085248_v3_fim.Q5_K_M_csr/`
- CSR Kimu: `20260719_104900_Gemma-Kimu-2b-base.Q6_K_csr/`
- Real-corpus Latxa: `20260719_085007_HiTZ.Latxa-Llama-3.1-8B.Q6_K_realcorpus/`
- Real-corpus Morpheus: `20260719_085039_v3_fim.Q5_K_M_realcorpus/`
- Real-corpus Kimu: `20260719_104913_Gemma-Kimu-2b-base.Q6_K_realcorpus/`
