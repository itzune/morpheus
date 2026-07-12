# Q4_K_M vs Q5_K_M — Head-to-Head Quality Comparison

**Date**: 2026-07-12  
**Checkpoint**: step 74K (best, valid PPL = 7.13)  
**Same checkpoint, same eval, same sentences — only quantization differs.**

## Multi-Token CSR (free-acceptance, 30 sentences, filtered/deployed)

| Metric | Q4_K_M | Q5_K_M | Δ (Q4−Q5) |
|--------|-------:|-------:|----------:|
| Macro CSR | 27.31% | 27.05% | +0.26pp |
| Micro CSR | 27.46% | 27.46% | 0.00pp |
| Keystrokes saved | 329 | 329 | 0 |
| Accepted (matched) | 253 | 246 | +7 |

**Verdict**: Identical. Micro CSR is exactly equal. The 0.26pp macro CSR difference is within noise.

## Next-Word Keyboard Simulation (15 sentences, Top-K metrics)

| Metric | Q4_K_M | Q5_K_M | Δ (Q4−Q5) |
|--------|-------:|-------:|----------:|
| Top-1 accuracy | 43.0% | 48.9% | −5.9pp |
| Top-3 accuracy | 75.6% | 75.6% | 0.0pp |
| Top-5 accuracy | 80.0% | 82.2% | −2.2pp |
| Acceptance rate | 75.6% | 75.6% | 0.0pp |
| Simulated CSR | 26.7% | 26.6% | +0.1pp |
| Avg confidence | 0.417 | 0.375 | +0.042 |
| Avg prefix before accept | 3.2 | 3.2 | −0.1 |

### Per-Language

| Language | Metric | Q4_K_M | Q5_K_M |
|----------|--------|-------:|-------:|
| Basque | Top-1 | 33.3% | 41.0% |
| Basque | Top-3 | 82.1% | 82.0% |
| Basque | Top-5 | 84.6% | 89.7% |
| Basque | CSR | 21.5% | 21.5% |
| English | Top-1 | 36.0% | 44.0% |
| English | Top-3 | 70.0% | 66.0% |
| English | Top-5 | 78.0% | 76.0% |
| English | CSR | 27.2% | 26.3% |
| Spanish | Top-1 | 58.7% | 60.9% |
| Spanish | Top-3 | 76.1% | 80.4% |
| Spanish | Top-5 | 78.3% | 82.6% |
| Spanish | CSR | 31.6% | 32.0% |

## Key Findings

1. **CSR is identical** (the product metric): Multi-token CSR differs by 0.26pp (noise). Next-word CSR differs by 0.1pp. Micro CSR is exactly equal.

2. **Top-3 and acceptance rate are identical**: The deployed UX shows 3 chips. Whether the correct word appears in those 3 chips is the same for both quants (75.6% acceptance rate).

3. **Q5 is modestly better at Top-1** (~6pp, consistent across all 3 languages): Higher precision helps the model rank the correct word as #1 more often. However, this does not affect the product metrics because the correct word still appears in the top-3 equally well with Q4.

4. **Q5 is slightly better at Top-5** (~2pp): The raw model output ceiling is marginally higher with Q5.

5. **Q4 has higher confidence** (0.417 vs 0.375): Q4's probability distribution is slightly more peaked. This is expected — lower precision quantization sharpens the distribution.

## Inference Speed Comparison (i7-8550U, 4 threads)

| Metric | Q4_K_M | Q5_K_M | Advantage |
|--------|-------:|-------:|-----------|
| Decode speed | 318 tok/s | 224 tok/s | Q4 +41% |
| Autocomplete latency | 97 ms | 115 ms | Q4 −16% |
| Model file size | 52 MB | 63 MB | Q4 −17% |

## Conclusion

**Q4_K_M is the correct deployment default.** The two quantizations are statistically indistinguishable on all product metrics (CSR, acceptance rate, Top-3). Q5's modest Top-1 advantage (~6pp) does not translate to any CSR or acceptance-rate improvement because both quants surface the correct word in the top-3 equally well. Meanwhile, Q4_K_M is 41% faster and 17% smaller — a free win with no quality cost on the metrics that matter for the deployed system.

The one nuance: if the UX were **single-chip** (only showing the #1 prediction), Q5's Top-1 advantage would matter. But with a 3-chip display, it is irrelevant.
