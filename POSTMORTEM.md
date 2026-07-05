# Post-Mortem: 4K Training Run summer-dust-9

**Date:** 2026-07-05
**Run:** `summer-dust-9` (W&B), steps 0–25,100 (33% of total)
**GPU:** NVIDIA L40, ~960 GPU-hours consumed
**Decision:** Killed — model not converging to useful autocomplete quality

---

## 1. What went wrong

The model at step 25,100 (33% training, PPL 8.0) produces nonsensical autocomplete
suggestions. Analysis of 14 Basque Wikipedia sentences typed character-by-character
shows:

| Token count | CSR (keystroke savings) | Acceptance rate | Notes |
|---|---|---|---|
| 1 | 9.4% | 6.6% | Best option, but still wrong 93% of the time |
| 2 | 4.6% | 2.0% | Degrades fast |
| 3+ | <3% | <2% | Repetition loops, gazette number artifacts |

For the sentence `Eskerrik asko zure laguntzagatik.` typed character by character,
only 1 of 26 autocomplete calls matched the actual text — and that was an empty
suggestion matching anything.

### Root causes

1. **Gazette number pollution:** The model learned digit patterns (`19ko`, `1964ko`,
   `2017ko`) as default fillers regardless of context. At 3+ tokens, 27% of
   predictions contained digits. The post-hoc filter (`filter_numeric_after_punct`)
   was upgraded to strip all digit tokens, but this only masks the problem.

2. **Data quality ceiling:** Official gazettes (BOG, BOPV, BOTHA) are 100% Basque
   but contain legal decree numbers, postal codes, budget amounts, and date
   fragments in sentence positions. Phase 2 cleaning's sentence-splitting regex
   didn't catch these because it required a capital letter after periods, but
   gazettes have `lowercase + period + digit` patterns.

3. **No pre-training validation:** We committed 76K steps (~4 days) of GPU time
   without first verifying that the model could produce coherent text on a small
   proxy run. The PPL was improving (16→8), but PPL ≠ autocomplete quality.

---

## 2. What we learned

### 2.1 PPL is not a proxy for autocomplete quality

The model's validation PPL dropped from 16.1 to 8.0 across 25K steps — an
improvement of 50%. But actual autocomplete CSR went _down_ from 9.4% to
essentially 0% as we added more tokens. PPL measures next-token probability
on held-out text; it doesn't measure generation quality, coherence, or
resistance to data artifacts.

### 2.2 Gazette contamination survived Phase 2 cleaning

Phase 2 was designed to handle HTML, long lines, duplicates, and punctuation
runs. But the sentence boundary regex `(?<=[a-z])([.?!])\s+(?=[A-Z])` was
blind to the `lowercase. digit` pattern that dominates gazette text:

```
kanpainak. 2010eko → NOT split (no capital after period)
salbuespenarekin. 2022ko → NOT split (no capital after period)
```

These lines train the model that `. [0-9]` and `. [0-9]ko` are valid
grammatical continuations, which it then applies universally.

### 2.3 MorphAcc is a tokenizer property, not a convergence property

MorphAcc@5 was 60% at step 4K and 60% at step 22K — identical. The
morphology improvement came from the 4K vocabulary size, not from
training length. This confirms the QuechuaTok fieldwork finding.

### 2.4 The 32K baseline CSR (83.6%) may have been inflated

The 32K model at step 30K scored CSR 83.6% on our eval-v3 targets — much
higher than the 4K model at equivalent training. But those targets were
synthetic (crafted prefixes), not real sentences. The model may have
memorized training-data patterns that happened to match the synthetic
targets. We never tested the 32K model on real Wikipedia sentences.

### 2.5 We ran 32K+4K training = ~1700 GPU-hours without a before-training gate

Between the 32K baseline (stopped at step 41K) and the 4K training (stopped
at step 25K), we committed ~1700 GPU-hours before discovering the model
wasn't useful for autocomplete. The only pre-training validation we did was
a corpus audit (Phase 2), which caught duplicates but not content artifacts.

---

## 3. Pre-Training Validation Protocol

Based on research into proxy-model validation (Qin et al. 2025, "Can Small
Training Runs Reliably Guide Data Curation?") and MLOps smoke-testing
practices, we define a **three-gate protocol** that must pass before
committing a full training run.

### Gate 1: Corpus Content Audit (CPU, ~30 min, zero GPU)

**Checks:**
- [ ] Per-source duplicate rate (< threshold)
- [ ] Per-source line-length distribution (flag long-line contamination)
- [ ] Per-source digit density (flag gazette/document-number patterns)
- [ ] Regex-based content pattern scan (URLs, HTML, PDF artifacts, ISBNs, dates)
- [ ] N-gram frequency skew check (top-100 trigrams — if "19ko" or "aktiboak" dominate, flag)

**Output:** Source-level pass/fail with specific remediation recommendations.
Script: `scripts/audit_corpus_quality.py` (exists, extend with digit-density scan).

### Gate 2: Proxy Overfit Test (GPU, ~2–5 min, ~$0.02)

**Method:** Train a tiny version of the model on a small data subset for a few
hundred steps and check if it can overfit a "canary" sentence injected into
the training data.

**Procedure:**
1. Take a random 10MB slice of the cleaned corpus.
2. Inject 5 "canary" sentences at known positions (unique Basque sentences
   not in the corpus, e.g., `"Lore moreak mendi berdeetan hazten dira udaberrian."`).
3. Train the model on this slice for 200–500 steps (enough to memorize).
4. Check: does the model complete the canary sentences correctly?

**Success criterion:** Model can complete canary sentences it was trained on.
If it CANNOT overfit 5 canaries in 500 steps, the data is too noisy or the
learning signal too weak — DO NOT commit to full training.

**Implementation:** New script `scripts/gate_proxy_overfit.py`.

### Gate 3: Autocomplete Smoke Test (GPU, ~5 min)

**Method:** After proxy training, run the CSR evaluation against real Basque
sentences (the Wikipedia-based test we built in `eval/v4_token_count.py`).

**Success criterion:** CSR > 30% at 1 token on the proxy-trained model.
If CSR < 30%, data quality or tokenization is insufficient.

---

## 4. Phase 3 Cleaning Specification

The missing cleaning step that would have prevented the gazette problem:

### 4.1 Numeric-aware sentence boundaries

Replace the current regex with a pattern that also splits when a period is
followed by a digit (not just a capital letter):

```python
# Current (broken):
r'(?<=[a-záéíóúüñ])([.?!])\s+(?=[A-ZÁÉÍÓÚÜÑ])'

# Fixed (numeric-aware):
r'(?<=[a-záéíóúüñ])([.?!])\s+(?=[A-ZÁÉÍÓÚÜÑ0-9])'
```

This will split `kanpainak. 2010eko` into separate lines.

### 4.2 Orphan number-line removal

Lines that are predominantly digits/currency should be removed:
- Lines where >50% of characters are digits
- Lines matching `€?\d{1,3}(\.\d{3})*([,.]\d{2})?` (currency amounts)
- Lines matching `\d{4}ko\s+(urte|maiatz|ekain|uztail|abuztu|irail|urri|azar|abendu)` (date fragments)
- Lines that are ONLY a number or year (e.g., `2010`, `2021`)

### 4.3 Budget line-item filtering

Gazette text contains budget tables where numbers dominate:
- Remove lines with >3 distinct number tokens
- Remove lines matching `\d{1,3}(\.\d{3})+(,\d{2})?` (formatted currency)

### Implementation

Extend `scripts/clean_phase2.py` with a `phase_3` flag or create
`scripts/clean_phase3.py`.

---

## 5. Updated TODO

### Immediate
1. [ ] Implement Gate 1 extensions (digit density, date-pattern scan) in audit script
2. [ ] Implement Gate 2 (proxy overfit test) — `scripts/gate_proxy_overfit.py`
3. [ ] Implement Gate 3 (autocomplete smoke test) — adapt from `eval/v4_token_count.py`
4. [ ] Implement Phase 3 cleaning (numeric-aware sentence boundaries, orphan number removal)

### After Phase 3
5. [ ] Re-clean corpus with Phase 3 → `data/clean-v3/`
6. [ ] Re-train 4K tokenizer on clean-v3
7. [ ] Pre-tokenize clean-v3
8. [ ] Run Gate 2 (proxy overfit) on clean-v3 data
9. [ ] Run Gate 3 (autocomplete smoke test) on proxy model
10. [ ] ONLY if Gates 2+3 pass: commit to full 4K training

### Research done
- [x] Post-mortem analysis of summer-dust-9 run
- [x] Research on proxy-model validation and smoke-testing
- [x] Pre-training validation gates designed
- [x] Phase 3 cleaning specified

---

## 6. Key Takeaway

> **Never train for 4 days what you can verify in 10 minutes.**

A 10-minute proxy overfit test + autocomplete smoke test would have caught
the gazette number problem before we burned 960 GPU-hours. Every future
training run must pass the three-gate protocol first.

## References

- Qin et al. (2025). "Can Small Training Runs Reliably Guide Data Curation?
  Rethinking Proxy-Model Practice." arXiv:2512.24503v2.
  — Proxy models need tiny learning rates (1-2 orders below standard) for
    reliable data quality ranking. Standard LR proxies produce rankings
    that don't transfer to target models.
- MLOps Community (2025). "Smoke Testing for ML Pipelines."
  — Run pipeline end-to-end with tiny synthetic data before committing
    to full training. Catch schema breaks, preprocessing bugs, and
    data format mismatches in seconds.
