# Tokenization Pipeline: Complete Specification for Morpheus v2.1

> **Status:** Coded, ready for execution on GPU server  
> **Date:** 2026-07-03  
> **Reference:** `docs/tokenizer-fieldwork.md` — experimental results confirming 4K optimal  

---

## Summary of Findings

Our controlled vocabulary ablation experiment (§4.5.4 in morpheus-on-device-basque-autocompletion.md) on Basque text
confirms the QuechuaTok finding for Basque:

| Vocab Size | MorphAcc Consistency | Fertility | Regime |
|-----------|---------------------|-----------|--------|
| 4K | **66.7%** | 2.58 | Morpheme-aligned splitting |
| 8K | 61.9% | 2.28 | Transitional (inconsistent) |
| 16K | 52.4% | 2.06 | Surface-form memorization begins |
| 32K | 28.6% | 1.85 | Surface-form memorization dominates |

**Decision:** Re-train Morpheus-Small with a 4K Unigram tokenizer. The 2.3× MorphAcc
improvement (28.6% → 66.7%) justifies the 39% fertility penalty.

---

## Pipeline Architecture

```
                     ┌──────────────────────────────────────┐
                     │   data/clean/ (22 GB, 14 .txt files) │
                     │   Latxa corpus v2 — hplt-v1 excluded │
                     └──────────────┬───────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  Phase 0.5:        │
                          │  Deep-clean        │
                          │  (scripts/         │
                          │   clean_phase2.py) │
                          │  → data/clean-v2/  │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  Phase A: 4K      │
                          │  Tokenizer        │
                          │  (scripts/        │
                          │   train_4k_       │
                          │   tokenizer.py)   │
                          └─────────┬─────────┘
                                    │
                          tokenizer/basque_unigram_4000.model
                                    │
                          ┌─────────▼─────────┐
                          │  Phase B: Pre-    │
                          │  tokenize corpus  │
                          │  (scripts/        │
                          │   pretokenize.py) │
                          └─────────┬─────────┘
                                    │
                      data/train_tokens_4k.npy
                      data/valid_tokens_4k.npy
                                    │
                          ┌─────────▼─────────┐
                          │  Phase C: Train   │
                          │  Morpheus-Small   │
                          │  (train.py        │
                          │   --config        │
                          │   config/         │
                          │   small.yaml)     │
                          └─────────┬─────────┘
                                    │
                          checkpoints/best.pt
                                    │
                          ┌─────────▼────────────────┐
                          │  Phase D: Eval + Export  │
                          │  (eval.py →           │
                          │   export_hf.py →         │
                          │   llama-quantize →       │
                          │   demo deployment)       │
                          └──────────────────────────┘
```

### Phase E: Morphological Analyzer Enhancement (Deferred)

The 4K tokenizer alone achieves 66.7% MorphAcc. To reach 80%+, we need
morphological pre-segmentation as a second-order enhancement. This is
deferred to **after** the 4K model is trained and evaluated.

**Selected analyzer:** **Apertium Basque** (`lt-proc` + `apertium-eu-es` data)

We compared the realistic Basque options:
- **Apertium Basque:** explicit morpheme boundary output via `+` markers in the analysis stream
- **Stanza Basque:** good lemma + UD morphological features, but **no explicit boundary positions**
- **ixa-pipes / euMor-style analyzers:** useful for lemma/POS analysis, but not the simplest corpus-scale morpheme segmenter

**Decision:** use **Apertium** for Phase E. It is the only tested option that directly emits boundary structure needed for corpus pre-segmentation.

> **Why not Morfessor?** ADR-013 (v1) and fieldwork Phase 2 (v2) both failed:
> Morfessor's MDL cost function mis-segments Basque on mixed Spanish/English text
> (e.g., `cookie` → `coo|kie`, `duten` → `duten|ez`). A rule-based Basque
> morphological analyzer is required for correct boundaries.

---

## Reproducibility Checklist

When reproducing the 4K pipeline in the future, verify these preconditions before running anything:

- [ ] `data/clean/` exists and contains the **14** cleaned `.txt` source files (hplt-v1 excluded per corpus-quality audit)
- [ ] `baseline_32k/` is preserved if you need head-to-head comparison
- [ ] `config/small.yaml` points to `data/train_tokens_4k.npy` and `data/valid_tokens_4k.npy`
- [ ] `config/small.yaml` uses `vocab_size: 4000` and `seq_len: 1024`
- [ ] `train.py` default vocabulary assumptions are compatible with 4K
- [ ] `tokenizer/basque_unigram_4000.model` exists before pretokenization
- [ ] Validation split is defined before launching training
- [ ] `WANDB_API_KEY` is exported if W&B logging is desired
- [ ] GPU server has enough free disk for token arrays and checkpoints

**Canonical order of operations:**
1. Exclude hplt-v1 + BERnaT BSM + BOG + aldizkariak from `data/clean/`
2. Preserve remaining 11 sources in `data/clean/`
3. Run Phase 2 deep-clean → `data/clean-v2/`
4. Run Phase 3 digit/noise filtering → `data/clean-v3/` (or combine with Phase 2 in one pass)
5. Train 4K tokenizer on `data/clean-v3/`
6. Create/verify validation split from `data/clean-v3/`
7. Pre-tokenize train split (two-pass: count → fill)
8. Pre-tokenize validation split
9. Launch training
10. CSR/MorphAcc logged to W&B at eval_interval
11. Run eval/export periodically
12. Only after 4K convergence, consider Phase E morphological pre-segmentation

---

## Phase 0: Source Exclusion

### hplt-v1 is excluded from the training corpus

The 2026-07-03 full-corpus fast audit found that `HiTZ_latxa-corpus-v2_hplt-v1.txt`
(74.3M lines) has:
- **83.80% duplicate cluster rate** — most content is repeated boilerplate
- **4.9% Basque signal** (heuristic) — the lowest of any source
- **9.4% clearly non-EU lines** — highest non-Basque contamination

These numbers make it the single worst-quality source in the corpus. It is
**excluded** from tokenizer training. The remaining 14 sources proceed to
Phase 2 deep-cleaning.

To exclude it:
```bash
rm data/clean/HiTZ_latxa-corpus-v2_hplt-v1.txt
# or: mv data/clean/HiTZ_latxa-corpus-v2_hplt-v1.txt data/excluded/
```

---

## Phase 0.5: Phase 2 Deep-Cleaning (recommended before tokenizer training)

### Script: `scripts/pipeline/clean_phase2.py`

Builds on Phase 1 (`clean_quick.py`) and adds:
- **HTML entity & escape cleanup** — decodes `&amp;`, `&gt;`, strips leftover tags
- **Sentence splitting / de-concatenation** — splits social-media feed lines into actual sentences
- **Long-line heuristics** — discards >2048 char lines, flags >512 char lines
- **Repeated-line suppression** — removes exact duplicate lines within each document

All strategies are individually toggleable and reversible.

```bash
# Dry-run: inspect before/after for one file (no output written)
python scripts/pipeline/clean_phase2.py \
    --input data/clean/HiTZ_BERnaT-Diverse_BSMauthor.txt \
    --output /tmp/test.txt \
    --dry-run-lines 200

# Full clean across the entire corpus directory
python scripts/pipeline/clean_phase2.py \
    --input data/clean \
    --output data/clean-v2
```

**Expected effects (measured on BERnaT/BSM sample):**
- HTML residue: 0.70% → 0.01% (-98.7%)
- Duplicate cluster rate: 5.17% → 0%
- Long lines (>512 chars): 2.42% → 1.53% (-37%)
- Line count increases ~6% due to sentence splitting

> **When to apply:** after the source-stratified corpus audit confirms which files need it most.
> The audit script is `scripts/pipeline/audit_corpus.py`.

> **If you skip this step:** the tokenizer will be trained on corpus that still has
> HTML entities, concatenated social-media lines, and repeated boilerplate.
> The model will learn these patterns and may produce them at inference time.

---

## Phase A: Train 4K Unigram Tokenizer

### Script: `scripts/pipeline/train_tokenizer.py`

Trains a SentencePiece Unigram model at 4K vocabulary on the full 22 GB corpus.

```bash
ssh root@10.2.121.210
cd /root/morpheus-mamba

python3 scripts/pipeline/train_tokenizer.py \
    --input-dir data/clean-v3 \
    --output-dir tokenizer \
    --vocab-size 4000 \
    --character-coverage 0.9995
```

**Expected output:**
```
tokenizer/basque_unigram_4000.model   (~500 KB)
tokenizer/basque_unigram_4000.vocab   (~80 KB)
Fertility: ~2.58
Training time: ~5-10 minutes
```

**Validation:** Check that roots split from case suffixes:
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/basque_unigram_4000.model")
sp.encode("etxetik", out_type=str)  # Should be ["▁etxe", "tik"]
sp.encode("etxera", out_type=str)   # Should be ["▁etxe", "ra"]
```

---

## Phase B: Pre-tokenize Full Corpus

### Script: `scripts/pipeline/pretokenize.py`

Tokenizes all 15 source files and saves as memory-mapped uint16 arrays.

```bash
cd /root/morpheus-mamba

# Training data (~9 GB uint16, from ~15 GB text × 2.0 fertility)
python3 scripts/pipeline/pretokenize.py \
    --sp-model tokenizer/basque_unigram_4000.model \
    --input-dir data/clean-v3 \
    --output data/train_tokens_4k.npy
```

**Actual result (2026-07-05):** **4,769,132,775 tokens, 8.9 GB** for the 11-source Phase-3-cleaned corpus.
Uses a two-pass approach (count → allocate → fill) to avoid OOM during pre-tokenization on 140M lines.

**Validation split:** Create a separate validation split if needed:
```bash
# If data/splits/valid/ doesn't exist, create one from a held-out source file
mkdir -p data/splits/valid
# Use Wikipedia as validation (small, clean, representative)
cp data/clean-v3/HiTZ_latxa-corpus-v2_wikipedia.txt data/splits/valid/

python3 scripts/pipeline/pretokenize.py \
    --sp-model tokenizer/basque_unigram_4000.model \
    --input-dir data/splits/valid \
    --output data/valid_tokens_4k.npy
```

---

## Phase C: Train Morpheus-Small

### Config: `config/small.yaml`

Updated for 4K vocabulary with 1024 sequence length (compensates 39% fertility increase):

```yaml
vocab_size: 4000
seq_len: 1024           # was 512 — compensates for longer tokenized sequences
batch_size: 64
gradient_accumulation: 2
train_data: data/train_tokens_4k.npy
valid_data: data/valid_tokens_4k.npy
```

**Launch command:**

```bash
cd /root/morpheus-mamba

WANDB_API_KEY='<key>' \
PYTHONUNBUFFERED=1 \
nohup python3 -u train.py \
    --config config/small.yaml \
    > logs/train_4k_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $!  # Save the PID
```

**Expected training characteristics:**
- ~113M parameters (unchanged architecture, smaller embedding: 4000×768 vs 32000×768)
- Smaller embedding table saves VRAM: 4000×768×2 bytes ≈ 6 MB vs 32000×768×2 ≈ 49 MB
- ~1024 seq_len × 64 batch ≈ 65K tokens per step (effective 128K with accum=2)
- Total tokens: 4.77B
- Total steps: 4.77B tokens / 128K ≈ 36,400 steps per epoch
- VRAM: ~20-25 GB (fits on L40 with 46 GB)
- CSR and MorphAcc logged to W&B at eval_interval (every 500 steps) alongside valid/loss and valid/ppl

---

## Phase D: Evaluation and Export

### D.1: Checkpoint evaluation

```bash
# At each 10K-step checkpoint, run three-strategy eval
python3 eval.py \
    --checkpoint checkpoints/step_0010000.pt \
    --tokenizer tokenizer/basque_unigram_4000.model \
    --targets eval/targets.json \
    --device cuda \
    --strategy all
```

### D.2: Export to GGUF

```bash
# Export HF format
python3 scripts/export/export_hf.py \
    --checkpoint checkpoints/best.pt \
    --tokenizer tokenizer/basque_unigram_4000.model \
    --output-dir exports/step_best

# Convert to GGUF (HuggingFace → GGUF)
python3 llama.cpp/convert_hf_to_gguf.py \
    exports/step_best \
    --outtype f16 \
    --outfile exports/step_best.f16.gguf

# Quantize
llama.cpp/llama-quantize \
    exports/step_best.f16.gguf \
    Q4_K_M \
    exports/step_best.Q4_K_M.gguf
```

### D.3: Compare against 32K baseline

```bash
# Load 32K baseline (preserved in baseline_32k/)
python3 eval.py \
    --checkpoint baseline_32k/best.pt \
    --tokenizer tokenizer/basque_unigram_32k.model \
    --targets eval/targets.json \
    --device cuda \
    --strategy all \
    --output eval/results/32k_baseline

# Load 4K result
python3 eval.py \
    --checkpoint checkpoints/best.pt \
    --tokenizer tokenizer/basque_unigram_4000.model \
    --targets eval/targets.json \
    --device cuda \
    --strategy all \
    --output eval/results/4k_final
```

**Expected comparison:**

| Metric | 32K baseline | 4K projected | Source |
|--------|-------------|-------------|--------|
| CSR (macro) | 81.7% | 90%+ | Extrapolation from MorphAcc improvement |
| MorphAcc@5 | 6.9% boundary mass | 40-50% | Fieldwork: 66.7% consistency → better prob mass |
| Paradigm Hit@1 | 3.6% | 20-40% | Tokens like ▁ra, ▁tik now exist independently |
| Fertility | 1.85 | 2.58 | Measured in fieldwork |

---

## Phase E: Morphological Analyzer Enhancement (Deferred)

> **Trigger:** Execute if 4K MorphAcc@5 plateaus below 50% at model convergence.

### Why Phase E is needed

The 4K tokenizer correctly splits 66.7% of single-layer case suffixes but fails on:
- Multi-layer suffixes: `etxe+a+rentzat` → `▁etxe + a + rentzat` (should preserve the full case complex)
- Epenthetic/linker forms: `lagun + ekin` → `lagunarekin`, `lagun + tik` → `lagunetik`

A Basque morphological analyzer resolves these by segmenting at true morpheme boundaries.

### Selected approach: Apertium Basque

**Installation on Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y apertium apertium-eu-es
```

**Why `apertium-eu-es` and not `apertium-eus`?**
On the target server, the packaged Basque analyzer data was available as `apertium-eu-es`, and its `lt-proc` automaton exposed the needed morphological analyses.

**What Apertium returns:**

```bash
echo "etxetik" | lt-proc /usr/share/apertium/apertium-eu-es/eu-es.automorf.bin
# ^etxetik/etxe<n>+a<det><art><sg>+tik<post>$

echo "etxearentzat" | lt-proc /usr/share/apertium/apertium-eu-es/eu-es.automorf.bin
# ^etxearentzat/etxe<n>+a<det><art><sg>+entzat<post>$
```

The `+` signs give the analyzer's morpheme boundaries.

### Critical representation rule: use surface-preserving segmentation

Do **not** blindly inject the analyzer's underlying morphemes into the training corpus. Apertium may expose underlying structure that is not literally visible in the surface word:

- analysis: `etxe+a+tik`
- surface word: `etxetik`

For training text, keep the original orthography and insert only recoverable boundaries:

- `etxetik` → `etxe#tik`
- `etxearekin` → `etxe#a#rekin`
- `etxearentzat` → `etxe#a#rentzat`

This is the representation implemented in `scripts/segment_morphemes.py`.

> **⚠️ Script status: prototype, not production-ready.** `segment_morphemes.py` has
> been validated on small samples (e.g., `etxe#tik`, `etxe#a#rekin`), but raw
> `lt-proc` output can glue punctuation, quotes, and newlines to neighboring
> words. Running the script over the full 22 GB corpus will require robust
> postprocessing to handle these edge cases, plus a pass of surface-preserving
> validation on a large random sample (not just a few hand-picked words)
> before the output can be trusted for tokenizer training.

> **⚠️ Phase E is conditional on 4K training results.** Our fieldwork shows
> plain 4K Unigram alone improves MorphAcc from 28.6% to 66.7%. Phase E
> (Apertium pre-segmentation) is only justified if the 4K-trained model still
> shows clear morphology gaps: MorphAcc < 80%, Paradigm Hit@1 single-digit,
> multi-layer suffixes still failing. If the 4K model learns these by itself,
> Phase E is unnecessary. Do not run Phase E before establishing the 4K baseline.

### Full Phase E procedure

```bash
cd /root/morpheus-mamba

# 1. Segment the corpus with Apertium
python3 scripts/segment_morphemes.py \
    --input-dir data/clean \
    --output-dir data/segmented \
    --processes 4

# 2. Train a new tokenizer on the segmented corpus
python3 scripts/pipeline/train_tokenizer.py \
    --input-dir data/segmented \
    --output-dir tokenizer_segmented \
    --vocab-size 4000

# 3. Pre-tokenize segmented train/valid data
python3 scripts/pipeline/pretokenize.py \
    --sp-model tokenizer_segmented/basque_unigram_4000.model \
    --input-dir data/segmented \
    --output data/train_tokens_4k_segmented.npy

# 4. Update config paths, then retrain from scratch
python3 train.py --config config/small.yaml
```

**Operational note:** Apertium throughput was measured at roughly **9.4×10^5 words/second** on the server, so full-corpus segmentation is operationally cheap compared with tokenizer training or model training.

### Non-selected alternatives

- **Stanza Basque:** useful for lemma + UD features, but lacks explicit boundary positions for direct pre-segmentation.
- **ixa-pipes / euMor:** promising linguistic tooling, but not the simplest validated path for immediate corpus-scale segmentation in this project.

### After segmentation, expected outcome

Training a 4K tokenizer on the segmented corpus is expected to push MorphAcc beyond the plain-4K result and closer to morphology-aware results reported in QuechuaTok/PRPE-style settings.

---

## File Inventory

| File | Purpose | Status |
|------|---------|--------|
| `scripts/pipeline/train_tokenizer.py` | Train 4K Unigram on full corpus | ✅ Created |
| `scripts/pipeline/pretokenize.py` | Pre-tokenize corpus to .npy | ✅ Updated (4K docs + uint16 validation) |
| `config/small.yaml` | Training config for 4K tokenizer | ✅ Updated (vocab=4000, seq_len=1024) |
| `train.py` | Training loop | ✅ Updated (default vocab=4000) |
| `eval.py` | Three-strategy eval | ✅ Compatible (takes --tokenizer arg) |
| `scripts/train_tokenizers.py` | Fieldwork: train multiple tokenizers | ✅ Exists (fieldwork only) |
| `scripts/train_morfessor.py` | Fieldwork: train Morfessor (abandoned) | ✅ Exists (reference) |
| `docs/tokenizer-fieldwork.md` | Full experiment report | ✅ Complete |
| `docs/tokenizer-pipeline.md` | This file | ✅ Created |
| `morpheus-on-device-basque-autocompletion.md` | §4.5.4: fieldwork results integrated | ✅ Updated |
