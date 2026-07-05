# PLAN.md — Phased Execution Plan for Morpheus v2 Mamba

> **Target hardware:** 1× NVIDIA L40 (48 GB VRAM) for training; consumer x86-64 CPU for inference
>
> **Total estimated timeline:** 8-10 days
>
> **Model size guide:** Start with Small (130M) → fast iteration. Scale to Base (200M) → production.

---

## Phase 1: Foundation (Days 1-2)

**Goal:** Working training pipeline with Morpheus-Small (130M). Verify everything: GPU access, mamba-ssm installation, data pipeline, training loop.

### Tasks

| # | Task | Output | Verification |
|---|------|--------|-------------|
| 1.1 | Set up Docker training environment | `Dockerfile.train`, `docker-compose.train.yml` | `nvidia-smi` inside container |
| 1.2 | Install mamba-ssm + dependencies | Working Python environment | Smoke test (MambaLMHeadModel runs) |
| 1.3 | Link/copy v1 tokenizer | `tokenizer/basque_unigram_32k.model` | `sentencepiece.SentencePieceProcessor` loads it |
| 1.4 | Pre-tokenize corpus | `data/train_tokens.npy`, `data/valid_tokens.npy` | Token count and range validated |
| 1.5 | Implement MemmapTokenDataset | `src/dataset.py` | DataLoader produces correct shape batches |
| 1.6 | Implement training script | `train.py` | Loss decreasing after 100 steps |
| 1.7 | Run Morpheus-Small for 1K steps | Training metrics in W&B | Loss decreasing, no NaN |
| 1.8 | Run Morpheus-Small for 1 epoch | Valid PPL computed | PPL < 200 (directionally correct) |

**Milestone:** Training pipeline produces decreasing loss on Morpheus-Small (130M). GPU utilization > 80%. No NaN losses.

---

## Phase 2: Full Training — Morpheus-Base (Days 3-6)

**Goal:** Train the production 200M model to convergence (valid PPL < 50).

### Tasks

| # | Task | Output | Verification |
|---|------|--------|-------------|
| 2.1 | Launch Base (200M) training | Training running on L40 | W&B dashboard showing live metrics |
| 2.2 | Monitor warmup phase (first 50M tokens) | Stable LR ramp, no gradient spikes | grad_norm < 10 consistently |
| 2.3 | Monitor mid-training (50M → 5B tokens) | Loss and PPL steadily decreasing | 2+ epochs complete, PPL trending down |
| 2.4 | Monitor late training (5B → 10B tokens) | PPL approaching plateau | Cosine LR decay functioning |
| 2.5 | Save best checkpoint by valid PPL | `checkpoints/best.pt` | Valid PPL < 50 |

**Milestone:** Valid PPL < 50 achieved. If not, continue training (up to 15B tokens) or escalate to Large (370M).

### Expected Training Profile

| Metric | Estimate |
|--------|----------|
| VRAM usage (batch 128, seq 512) | ~15-20 GB |
| Training throughput | ~25-40K tokens/sec |
| Epochs to 10B tokens | ~4-5 |
| Wall-clock per epoch | ~17-28 hours |
| **Total training time** | **~3-5 days** |

---

## Phase 3: Evaluation (Day 6-7)

**Goal:** Quantify model quality on standard autocomplete metrics. Compare against v1 baseline.

### Tasks

| # | Task | Output | Verification |
|---|------|--------|-------------|
| 3.1 | Compute valid perplexity | Best checkpoint PPL | PPL < 50 |
| 3.2 | Compute Top-1 / Hit@3 / Hit@5 | Accuracy metrics on test set | Hit@3 > 40%, Hit@5 > 50% |
| 3.3 | Run morphological agreement review | Qualitative suffix analysis | Ergative/absolutive predictions correct |
| 3.4 | Compare with v1 LSTM | Side-by-side metrics table | v2 beats v1 on all metrics |
| 3.5 | Run generation samples | Example predictions for manual review | Suggestions are contextually appropriate |

**Milestone:** Hit@3 > 40% on test set. Morphological agreement qualitatively correct.

---

## Phase 4: Deployment — Export & Quantize (Day 7-8)

**Goal:** Convert trained model to deployable GGUF format, quantize, validate.

### Tasks

| # | Task | Output | Verification |
|---|------|--------|-------------|
| 4.1 | Export checkpoint to HuggingFace format | `exports/morpheus-v2-mamba-hf/` | `config.json` + `model.safetensors` |
| 4.2 | Copy tokenizer to export dir | `exports/morpheus-v2-mamba-hf/tokenizer.model` | File present |
| 4.3 | Convert HF → GGUF (FP16) | `exports/morpheus-v2-f16.gguf` | llama.cpp loads model |
| 4.4 | Quantize to Q5_K_M | `exports/morpheus-v2-Q5_K_M.gguf` | File size ~145 MB |
| 4.5 | Validate quantized PPL | Perplexity on valid set | Degradation ≤ 3% vs FP16 |
| 4.6 | Benchmark CPU latency | P50/P90 token generation speed | P90 ≤ 50ms for 3-token suggestion |

**Milestone:** Quantized model with < 3% PPL degradation and P90 ≤ 50ms on target CPU.

### Quantization Escalation Path

| If... | Then... |
|-------|---------|
| Q5_K_M PPL degrades > 3% | Try Q6_K or Q8_0 |
| Q5_K_M latency > 50ms (200M) | Try Q4_K_M or 130M variant |
| Q4_K_M PPL degrades > 5% | Accept 200M at Q5_K_M with slightly relaxed latency target (60ms) |

---

## Phase 5: Integration & Polish (Day 8-10)

**Goal:** End-to-end working autocomplete demo. Final benchmarks. Documentation.

### Tasks

| # | Task | Output | Verification |
|---|------|--------|-------------|
| 5.1 | Launch llama.cpp server with quantized model | `http://localhost:8080` serving completions | `curl` returns valid JSON |
| 5.2 | Adapt v1 demo client to llama.cpp API | Working autocomplete in browser | Suggestions appear on typing |
| 5.3 | Tune generation parameters | Optimal temperature/top-k for Basque | Suggestions feel natural |
| 5.4 | Final latency benchmark | P50/P90/P99 report on target CPU | All under targets |
| 5.5 | Final quality benchmark | Side-by-side vs v1 report | v2 > v1 on all metrics |
| 5.6 | Document final architecture | Updated README, ADRs, setup guide | Self-contained docs for new developers |

**Milestone:** Production-ready model meeting all v2 targets. End-to-end demo working.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 200M at Q5_K_M exceeds 50ms latency | Medium | Medium | Fall back to 130M or Q4_K_M (Phase 4) |
| NaN losses during Mamba training | Low | High | Gradient clipping at 1.0; reduce LR; increase warmup |
| Valid PPL > 50 after 10B tokens | Low | High | Escalate to Large (370M) or continue training |
| mamba-ssm build fails on L40 | Low | High | Use pre-built Docker image; try different CUDA version |
| llama.cpp Mamba support is buggy | Very Low | High | Use latest llama.cpp commit; report issue upstream |
| Quantized model quality is poor (Q5_K_M) | Low | Medium | Try Q6_K or Q8_0; accept larger model size |
| v2 doesn't beat v1 on quality metrics | Low | High | Root cause analysis; consider hybrid SSM-Transformer approach for v3 |
