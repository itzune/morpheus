# Cross-Model Baseline Evaluation Summary

**Date**: 2026-07-11
**Evaluator**: `scripts/eval_baselines.py` (direct PyTorch forward pass on full-precision BF16 HuggingFace weights; not the Q6_K GGUF deployment)
**Corpus**: `eval/real_corpus/` (14 files, 475,750 chars — Wikipedia + Berria)
**CSR targets**: `eval/targets.json` (30 tests, 149 words)
**⚠ Corpus contamination caveat**: Wikipedia/Berria may appear in all three models' training data. Absolute BPC is optimistic; relative comparison is valid.

## Results

| Model | Params | Vocab | BPC | PPL (token) | CSR (macro) | 95% CI | Word Acc |
|-------|--------|-------|-----|-------------|-------------|--------|----------|
| GPT-2 eus-euscrawl | 124M | 50K | 0.9811 | 29.21 | 0.1102 | [0.060, 0.167] | 37.6% (56/149) |
| **Morpheus v2 (Mamba-2)** | **91M** | **4K** | **0.9697** | **9.83** | **0.0941** | [0.058, 0.131] | **60.4% (90/149)** |
| Kimu 2B (base) | 2,000M | 256K | 0.7435 | 4.60 | 0.2146 | [0.173, 0.255] | 61.7% (92/149) |
| Latxa-Qwen3.5-2B (instruct) | 1,882M | 248K | 0.8219 | 4.89 | 0.2369 | [0.201, 0.275] | 68.5% (102/149) |
| Latxa 8B (base) | 8,000M | 128K | 0.4896 | 2.51 | 0.2659 | [0.226, 0.305] | 75.2% (112/149) |

### BPC (Bits Per Character) — Tokenizer-Independent

BPC normalizes across different vocabularies, enabling fair cross-model comparison.

- **Latxa 8B (base)** achieves the lowest BPC (0.490), as expected for an 8B-parameter model.
- **Kimu 2B (base)** is the efficiency frontier: BPC 0.744, better than the 1.9B Latxa-Qwen3.5-2B instruct (0.822) despite being base-trained on fewer tokens (1.5B vs 4.2B).
- **Morpheus** (0.970) outperforms **GPT-2** (0.981) despite having 27% fewer parameters (91M vs 124M).
  - Key factor: Morpheus was trained on ~10B tokens vs GPT-2's ~423M tokens (24× more training data).
- The BPC gap between Morpheus and Latxa 8B (0.480 bits/char) reflects the 88× parameter difference.

### PPL (Per-Token) — NOT Cross-Model Comparable

PPL is tokenizer-dependent: a smaller vocab produces more, shorter tokens, inflating per-token PPL.
- GPT-2's PPL of 29.21 looks terrible vs Morpheus's 9.83, but BPC shows the models are close.
- This is exactly why BPC is the correct cross-model metric.

### Simplified Next-Word CSR — No Inference Engineering

All models evaluated with the same simplified CSR: greedy decode until word boundary, compare first word to target.

- **Latxa 8B** has the highest CSR (0.266) and word accuracy (75.2%). The base Basque LLMs confirm the scale-quality gradient: Latxa 8B (0.266) > Latxa-Qwen3.5-2B instruct (0.237) > Kimu 2B base (0.215).
- **GPT-2** and **Morpheus** have overlapping CIs — no statistically significant CSR difference.
- **Morpheus** has 1.6× higher word accuracy than GPT-2 (60.4% vs 37.6%) despite similar CSR.
  - This is another instance of the CSR paradox: Morpheus predicts correct words more often, but saves fewer keystrokes per correct prediction (longer Basque words require more characters before the model converges).
- **Latxa-Qwen3.5-2B completions are noisy** — it's an instruct model, so raw text completion produces web navigation artifacts. The base Basque LLMs (Kimu 2B, Latxa 8B) produce clean output, confirming that instruct-tuning is the confound. This highlights a key advantage of Morpheus: it's a base LM naturally suited for autocomplete.

### Inference Engineering Value

| Regime | Simplified CSR | With engineering | Gap |
|--------|----------------|------------------|-----|
| Ghost-text (smart context, ghost suffix, garbage filter) | 0.266 | 0.271 | +0.005 (CSR-neutral) |
| Keyboard (retokenization, sticky merge, top-k) | 0.094 | 0.362 | **3.9×** |

Two regimes: the ghost-text strategies (§5.4 of the write-up) are CSR-neutral — they clean the ghost text (no byte-fallback garbage, no digit artifacts) but do not change keystroke savings, because junk output never matches the gold text anyway. The keyboard strategies (retokenization fallback, sticky merge) change *which words the model can reach* via multiple token paths, producing the large 3.9× effect. The keyboard engineering is documented in the companion futo-basque project.

## Key Takeaways

1. **BPC is the correct cross-model metric** — PPL is misleading across different vocabularies.
2. **Morpheus beats GPT-2 on BPC** despite being smaller — more training data matters.
3. **Latxa 8B is better but 88× larger** — Morpheus offers a favorable efficiency/BPC trade-off for on-device deployment. Kimu 2B (BPC 0.744) is the efficiency frontier among Basque LLMs, outperforming the instruct Latxa-Qwen3.5-2B (0.822) despite fewer training tokens.
4. **Word accuracy is more informative than CSR** for cross-model comparison (avoids the CSR paradox).
5. **Two inference-engineering regimes** — ghost-text strategies (smart context, garbage filter) are CSR-neutral (+0.005 gap); keyboard strategies (retokenization, sticky merge) add 3.9× CSR by changing which words are reachable. The former is documented in §5.4; the latter in the companion futo-basque project.
6. **Morpheus is a base LM** — naturally suited for autocomplete. Latxa-Qwen3.5-2B is an instruct model and produces noisy completions; the base Basque LLMs (Kimu 2B, Latxa 8B) produce clean output but are GPU-bound.

## Deployment Size Comparison

| Model | Quantized Size | Architecture |
|-------|---------------|--------------|
| Morpheus v2 | 55 MB (Q4_K_M) | Mamba-2 (linear-time, no KV cache) |
| GPT-2 eus-euscrawl | ~50 MB (Q4) | Transformer (quadratic attention) |
| Kimu 2B (base) | 2.1 GB (Q6_K) | Gemma-2 (base, Basque CPT) |
| Latxa-Qwen3.5-2B | ~1.2 GB (Q4) | Qwen3.5 (instruct, multimodal) |
| Latxa 8B (base) | 6.6 GB (Q6_K) | Llama-3.1 (base, Basque CPT) |

Morpheus is deployable on any device; the Basque LLMs require a GPU for real-time autocomplete.
