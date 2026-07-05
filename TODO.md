# TODO.md — Morpheus Mamba

> **Current phase:** 4K tokenizer retraining in progress (run `summer-dust-9`, step ~23K, PPL 10.7).
> **State:** Phase 1+2 cleaning complete. Gazettes contamination identified. Phase 3 cleaning design needed.

---

## 🚨 Active Issue: Gazette Number Pollution (2026-07-05)

**Symptom:** Model predicts bare numbers and year-like compounds ("201", "196", "19ko", "1964ko", "1907an") for almost any context at step 22K. At 3+ tokens, the model collapses into repetitive digit pattern loops regardless of prompt.

**27% of 3-token predictions contain digits** (measured across 15 diverse Basque prompts).

**Root cause:** Official gazettes (BOG, BOPV, BOTHA) contain legal decree numbers, postal codes, and budget line items in sentence positions where the model learns `. [0-9]` as a grammatical continuation:

| Pattern | Example | Source |
|---|---|---|
| Year as standalone sentence after period | `kanpainak. 2010` → `2010` is own line | BOG |
| Postal codes after dots | `. 20010 Donostia`, `. 20100 Lezo` | BOG, BOPV |
| Article/decreto numbers | `. 206. artikulua`, `. 165.1.b)` | BOG, BOPV, BOTHA |
| Budget amounts on own lines | `... 292.000,00`, `862.126,67` | BOG |
| Date fragments | `irailaren 28ko 23/2010` → standalone `23/2010` | BOG |

**Why Phase 2 didn't catch this:** The sentence-splitting regex is:
```
(?<=[a-záéíóúüñ])([.?!])\s+(?=[A-ZÁÉÍÓÚÜÑ])
```
It requires a **capital letter** after the period. Gazette text has lowercase + period + **digit**, which bypasses the rule entirely:
- `kanpainak. 2010` → NOT split (digit after period)
- `salbuespenarekin. 2022ko` → NOT split (digit after period)
- `47/2015). 2022` → NOT split (parenthesis before period)

**Scale of contamination (3 gazette sources, 141M total lines):**

| Source | Lines | Lines with `\. [0-9]{3,4}` | 4-digit years |
|---|---|---|---|
| BOG | 6.4M | 11,615 | 333,006 |
| BOPV | 2.0M | 567 | 250,177 |
| BOTHA | 1.8M | 3,316 | 253,064 |
| All other sources (10 files) | 130M | ~95K | TBD |

**Fix needed (Phase 3):** Add numeric-aware sentence boundary patterns to the cleaner. At minimum:

```python
# Split when period is followed by whitespace + digit
NUMERIC_SENT_BOUNDARY = re.compile(
    r"(?<=[a-záéíóúüñ])"          # preceded by lowercase letter
    r"([.?!])\s+(?=\d)"            # then punct + whitespace + digit
)

# Also: remove orphaned number-only lines
NUMBER_ONLY_LINE = re.compile(r"^\s*[\d\s.,/€]+\s*$")
```

**Impact:** Re-cleaning all 13 sources with this fix and re-tokenizing from scratch would add ~2 hours of CPU processing. This should be done before the next full training run.

---

## Pipeline Status

| Phase | Status | Script | Notes |
|---|---|---|---|
| Phase 1 (basic cleaning) | ✅ Complete | `scripts/clean_quick.py` | URLs, mentions, emails, repeated chars |
| Phase 2 (deep cleaning) | ✅ Complete | `scripts/clean_phase2.py` | HTML, sentence split, long lines, dedup |
| Phase 3 (numeric pollution) | 🔴 TODO | — | Gazette number patterns, orphan number lines |
| Source exclusions | ✅ Complete | — | hplt-v1 + BERnaT BSM removed |
| 4K tokenizer | ✅ Complete | `scripts/train_4k_tokenizer.py` | Fertility 2.51, morphology verified |
| Pre-tokenization | ✅ Complete | `scripts/pretokenize.py` | 5.07B tokens, 9.5 GB |

---

## Phase 3 Cleaning Design

### Must-have

| # | Strategy | Impact | Notes |
|---|---|---|---|
| 13 | Numeric-aware sentence boundaries | **High** | Gazette number pollution — see above |
| 14 | Orphan number-line removal | **High** | Lines that are only numbers/currency amounts |
| 15 | Budget line-item filtering | Medium | Euro amounts on standalone lines (e.g., `... 292.000,00`) |

### Nice-to-have (deferred)

| # | Strategy | Notes |
|---|---|---|
| 5 | Language detection & filtering | Preserve Euskañol code-switching, only strip clear non-Basque |
| 9 | Emoji-rich line removal | Tighter threshold than Phase 1 |
| 11 | PDF boilerplate removal | Headers, footers, page numbers — per-doc structure required |
| 12 | Corpus-level n-gram dedup | MinHash LSH across full corpus |

---

## Eval v3 Framework

Three-strategy evaluation implemented in `eval_v3.py`. Target file: `eval/v3-targets.json`.

| Strategy | Metric | Step 4K (5%) | Step 22K (28%) | 32K baseline (step 30K) |
|---|---|---|---|---|
| CSR | Character Savings Rate | 68.4% | 67.5% | 83.6% |
| MorphAcc@5 | Morpheme Boundary Accuracy | 60% | 60% | 20% |
| Paradigm Hit@1 | Case paradigm completion | 9.5% | 8.3% | 3.6% |
| Paradigm Hit@3 | | 20.2% | 20.2% | — |
| Paradigm Hit@5 | | 27.4% | 26.2% | — |

**Key finding:** 4K MorphAcc is already 3× the 32K baseline. MorphAcc doesn't improve between 5%→28% training — it's a tokenizer property, not a training length property.

---

## Deferred: Phase E — Apertium Pre-segmentation

Script prototyped at `scripts/segment_morphemes.py` but not production-ready (punctuation glue in `lt-proc` output). Only execute after 4K training + evaluation is complete and if MorphAcc still lags on multi-layer suffixes.

---

## Demo

Running on `localhost:9090` via Docker. Current model: `step22k.Q4_K_M.gguf` (52 MB). Configurable via `MORPHEUS_MODEL` env var.

### Greedy default token count (2026-07-05)

Evaluated 15 diverse Basque prompts at 1–12 tokens with greedy decoding. Findings:

- **At 1 token**: reasonable but short completions (e.g., `Etxe` → `a` ✓)
- **At 3+ tokens**: model enters repetition loops (`goazenak, goazenak,`) or gazette patterns — not converged yet at step 22K
- **Conclusion**: set greedy default to **1 token** until model converges further (~step 50K+)
- **Post-hoc filter (2026-07-05)**: upgraded `filter_numeric_after_punct` to strip ALL digit-containing tokens (not just period+digit). Catches `19ko`, `1964ko`, `2017ko` year-compounds.
