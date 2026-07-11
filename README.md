# Morpheus v2 Mamba — On-Device Basque Autocomplete

> **20× capacity boost over v1.** Mamba-2 State Space Model, ~91M (Small) / ~200M (Base) parameters.
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
| [`DATA.md`](./DATA.md) | Corpus strategy, cleaning, and split policy |
| [`docs/tokenizer-pipeline.md`](./docs/tokenizer-pipeline.md) | **Canonical end-to-end tokenizer/training pipeline** |
| [`docs/tokenizer-fieldwork.md`](./docs/tokenizer-fieldwork.md) | 4K vs 8K vs 16K vs 32K tokenizer ablation results |
| [`DECISIONS.md`](./DECISIONS.md) | Architecture Decision Records for v2 |
| [`START.md`](./PLAN.md) | Phased execution plan with milestones and acceptance criteria + current status |

## Architecture at a Glance

```
User Keystrokes
      │
      ▼
 Trigger Heuristic (≥50ms pause OR spacebar)
      │
      ▼
 SentencePiece Unigram Tokenizer ←── Current canonical setup: 4K vocab
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

## Evaluation Logs

| Date | Step | PPL | Findings |
|------|------|-----|----------|
| [2026-07-01](./logs/cpu-vs-gpu-generation-2026-07-01.md) | 14K | ~30 | Early checkpoint: repetition dominant, Q4_K_M quality-neutral at this PPL |
| [2026-07-02](./logs/cpu-vs-gpu-generation-2026-07-02.md) | 36.5K | ~25.5 | Clear improvement: longer coherent runs, news strongest, Q4_K_M still equal to FP16 |

## Repository Relationship

```
itzune/
├── morpheus/              # v1 LSTM — baseline and data/tokenizer provider
└── morpheus-mamba/        # v2 Mamba-2 — this repo
```

**Reused from v1:**
- Corpus acquisition/cleaning strategy
- Some split conventions and documentation context
- Research framing around Basque autocomplete constraints

**Not reused:**
- The old 32K tokenizer as the canonical setup (it is now a preserved baseline only)
- All model code (LSTM → Mamba)
- All training scripts
- ONNX export/quantization pipeline (replaced by llama.cpp GGUF)
- Inference server (ONNX Runtime → llama.cpp)

## Canonical Reproduction Pipeline (Current 4K Setup)

If you want to reproduce the project from raw cleaned text to demo deployment, follow these steps **in order**. The authoritative detailed version lives in [`docs/tokenizer-pipeline.md`](./docs/tokenizer-pipeline.md).

### 0. Preconditions

- Corpus acquisition and cleaning have been completed as documented in [`DATA.md`](./DATA.md)
- Clean corpus (Phase 3) is present in `data/clean-v3/` as **11** `.txt` files (hplt-v1, BERnaT BSM, BOG, and aldizkariak excluded; Phase 2 + Phase 3 deep-cleaned)
- **Corpus size: 140M lines, 15 GB → 4,769,132,775 tokens (8.9 GB uint16 .npy)**
- Training budget: ~36.4K steps per epoch (1024 seq × 128 effective batch)
- `config/small.yaml` is the 4K config (`vocab_size: 4000`, `seq_len: 1024`)
- You have enough disk for token arrays + checkpoints
- **Gates 1 & 2 passed** (LLM audit: 11 clean sources; proxy overfit: 57.7% canary accuracy)
- If you want experiment tracking, export `WANDB_API_KEY`
- Keep `baseline_32k/` if you want future 32K vs 4K comparison

### 0.25 Exclude hplt-v1

The full-corpus fast audit (2026-07-03) found `HiTZ_latxa-corpus-v2_hplt-v1.txt` to be the worst-quality source:
- 83.80% duplicate cluster rate
- 4.9% Basque heuristic signal
- 9.4% clearly non-EU lines

```bash
# Move hplt-v1 out of the clean directory
mkdir -p data/excluded
mv data/clean/HiTZ_latxa-corpus-v2_hplt-v1.txt data/excluded/
```

### 0.5 Phase 2 deep-clean (recommended)

After excluding hplt-v1, apply a deeper structural cleaning to the remaining 14 sources. This removes HTML entities, splits concatenated social-media lines, suppresses repeated boilerplate, and handles very long lines.

```bash
# Run a source-stratified audit first to see what needs cleaning:
python scripts/pipeline/audit_corpus.py \
    --input data/clean \
    --output-json reports/corpus_quality_fast_audit.clean.json \
    --output-md reports/corpus_quality_fast_audit.clean.md

# Then apply Phase 2 deep-cleaning:
python scripts/pipeline/clean_phase2.py \
    --input data/clean \
    --output data/clean-v2

# Then apply Phase 3 digit/noise filtering:
python scripts/pipeline/clean_phase2.py \
    --input data/clean-v2 \
    --output data/clean-v3 \
    --no-split --digits  # re-process with digit+orphan filters on already-split text

# (Phase 2+3 can also be combined in one pass on new data:
#  python scripts/pipeline/clean_phase2.py --input data/clean --output data/clean-v3 --digits)

# Optionally spot-check the output with a dry-run first:
python scripts/pipeline/clean_phase2.py \
    --input data/clean/HiTZ_BERnaT-Diverse_BSMauthor.txt \
    --output /tmp/test.txt \
    --dry-run-lines 200
```

What Phase 2 adds on top of Phase 1:
- **HTML entity & escape cleanup** (decodes `&amp;`, strips tags)
- **Sentence splitting** for social-media feed lines
- **Long-line heuristics** (>2048 chars discarded, >512 flagged)
- **Repeated-line suppression** (within-document exact duplicates)

Measured on a BERnaT/BSM sample: HTML residue -98.7%, duplicate clusters eliminated, long-line burden -37%.

> If you skip Phase 2, the tokenizer will be trained on text that still has HTML entities, concatenated feed lines, and repeated boilerplate. The model may reproduce these at inference time.

### 1. Train the 4K tokenizer

```bash
python3 scripts/pipeline/train_tokenizer.py \
    --input-dir data/clean-v3 \
    --output-dir tokenizer \
    --vocab-size 4000
```

Expected artifact:
- `tokenizer/basque_unigram_4000.model`

### 2. Create or verify validation split

The training run assumes both train and validation token arrays exist. If you do not already have a held-out split, create one before pretokenization.

```bash
mkdir -p data/splits/valid
cp data/clean-v3/HiTZ_latxa-corpus-v2_wikipedia.txt data/splits/valid/
```

### 3. Pre-tokenize the full training corpus

The pre-tokenization script uses a two-pass approach (count → allocate → fill) to avoid OOM on large corpora. The first pass counts total tokens (no list accumulation), and the second pass fills a pre-allocated uint16 array.

```bash
python3 scripts/pipeline/pretokenize.py \
    --sp-model tokenizer/basque_unigram_4000.model \
    --input-dir data/clean-v3 \
    --output data/train_tokens_4k.npy
```

**Output:** `data/train_tokens_4k.npy` — **4,769,132,775 tokens (8.9 GB uint16)** for the 11-source Phase-3-cleaned corpus.

### 4. Pre-tokenize the validation corpus

```bash
python3 scripts/pipeline/pretokenize.py \
    --sp-model tokenizer/basque_unigram_4000.model \
    --input-dir data/splits/valid \
    --output data/valid_tokens_4k.npy
```

**Expected output:** `data/valid_tokens_4k.npy` — 1,848,467 tokens (3.5 MB uint16).

### 5. Launch training

```bash
PYTHONUNBUFFERED=1 \
nohup python3 -u train.py \
    --config config/small.yaml \
    > logs/train_4k_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 6. Evaluate checkpoints

```bash
python3 eval.py \
    --checkpoint checkpoints/best.pt \
    --tokenizer tokenizer/basque_unigram_4000.model \
    --targets eval/targets.json \
    --device cuda \
    --strategy all
```

### 7. Export to GGUF for CPU inference

```bash
python3 scripts/export/export_hf.py \
  --checkpoint checkpoints/best.pt \
  --tokenizer tokenizer/basque_unigram_4000.model \
  --output-dir exports/step_best

python /tmp/llama.cpp/convert_hf_to_gguf.py exports/step_best \
  --outfile exports/step_best.f16.gguf \
  --outtype f16

/tmp/llama.cpp/build/bin/llama-quantize \
  exports/step_best.f16.gguf \
  exports/step_best.Q4_K_M.gguf \
  Q4_K_M
```

### 8. Run the demo stack

```bash
/tmp/llama.cpp/build/bin/llama-server \
  -m exports/step_best.Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 -ngl 0

cd demo
uv sync
uv run python server.py
```

Open `http://localhost:9090`.

### Optional Phase E: morphology-aware pre-segmentation

This is **not** part of the canonical 4K retraining path. Only do it after the 4K model is trained and evaluated.

Use **Apertium Basque** for this phase, not Stanza or Morfessor:

```bash
sudo apt-get install -y apertium apertium-eu-es
python3 scripts/segment_morphemes.py \
    --input-dir data/clean \
    --output-dir data/segmented \
    --processes 4
```

Then retrain the tokenizer on `data/segmented/` and repeat pretokenization/training from scratch.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model architecture | Mamba-2 (via `mamba-ssm` package) |
| Training | PyTorch 2.4+ with CUDA 12.4 |
| Data pipeline | Memory-mapped numpy arrays |
| Tokenization | SentencePiece Unigram, **canonical setup = 4K vocab** |
| Optional morphology layer | Apertium Basque pre-segmentation (`lt-proc`) |
| Model export | PyTorch checkpoint → HuggingFace export (`export_hf.py`) → llama.cpp `convert_hf_to_gguf.py` |
| CPU inference | llama.cpp with Q4_K_M / Q5_K_M quantization |
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
| Model params | 9.5M | ~91M (Small) / ~200M (Base) |
| Model size on disk | ~12 MB | ~145 MB |
