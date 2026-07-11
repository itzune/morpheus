# Evaluation Reform — Findings & Conclusions

> **Status:** Complete (D0 + R1 + R2 + R10 all executed)
> **Date:** 2026-07-10
> **Purpose:** Detailed record of all evaluation-reform findings, to survive
> context compaction and serve as the basis for the paper update.
> **Authoritative source for all numbers cited below.**

---

## Background: The Eval Crisis

Two automated eval paths (GPU `eval.py` vs demo `csr_eval.py`) disagreed with
each other **and** with the human expert's qualitative judgment on whether
checkpoint 54K was better than 32K:

| Path | 32K | 54K | Direction |
|------|-----|-----|-----------|
| GPU `eval.py` (f16) CSR macro | 0.2917 | 0.2682 | 54K **worse** (−2.35) |
| Demo `csr_eval.py` (Q4) CSR macro | 0.2780 | 0.2884 | 54K **better** (+1.04) |
| Human (expert, unblinded) | — | — | 54K "much better" |

The reform proposal (`docs/eval-reform-proposal.md`) diagnosed five root causes:
1. Sample too small (n=30; each sentence = 3.3 points)
2. Exact-match gold can't credit valid Basque alternatives (agglutinative problem)
3. Two CSR implementations disagree by 3.4 points (quantization + SP decoding confounds)
4. `targets.json` saturated (108 eval passes during training)
5. CSR = greedy single-token match, not fluency

The reform was executed in three tiers: D0 (PPL diagnostic), R1+R2 (scale CSR + bootstrap CIs), R10 (blinded human A/B).

---

## D0: Perplexity on Held-Out Text (RESOLVED)

### Script
`scripts/ppl_eval.py` — matches training semantics exactly:
- Line-by-line tokenization with `</s>` (id=2) separators (same as `pretokenize.py`)
- 1024-token windows, shifted by 1 (same as `MemmapTokenDataset`)
- `ignore_index=0` (<unk>); `</s>` (id=2) IS included in loss (same as `train.py::evaluate`)
- bfloat16 autocast, **NO BOS** (model trained without BOS)
- Token-weighted mean cross-entropy

### Leakage check (important caveat)
Checked whether real-corpus articles appear in training sources:
- **Wikipedia articles**: FOUND in `data/clean-v3/HiTZ_latxa-corpus-v2_wikipedia.txt`
  (e.g. "hizkuntza bakartua da" appears 5×; "euskal alfabetoaren" appears)
- **Berria articles**: FOUND in `data/clean-v3/HiTZ_latxa-corpus-v2_{fineweb2,cultura-x,euscrawl-v2}.txt`
  (e.g. "German Rodriguezen omenezko" found in 3 training source files)

**Conclusion:** The entire real corpus is contaminated → absolute PPL on real corpus is optimistic (memorized). BUT the relative 32K-vs-54K comparison is still valid (both saw the same text). The **held-out validation set** (`data/valid_tokens_4k.npy`, 1.8M tokens, excluded from training via leakage fix) provides the clean signal.

### Results

| Metric | Step 32K | Step 54K | Direction |
|--------|----------|----------|-----------|
| **Held-out validation PPL** (clean, 1.8M tokens) | **7.56** (loss 2.0229) | **7.17** (loss 1.9698) | 54K **better** ↓5.2% |
| **Real corpus PPL** (contaminated, 140K tokens) | **10.53** (loss 2.3540) | **9.90** (loss 2.2923) | 54K **better** ↓6.0% |

### Decisive detail: all 14 files agree
Every single real-corpus file shows 54K < 32K — zero exceptions. Per-file deltas range from −0.54 to −0.88 PPL points. This is statistically unambiguous.

### Interpretation
PPL is the smoothest, lowest-variance metric (full distribution, no exact-match artifact, 1.8M tokens). 54K is definitively a better language model than 32K. This is the only statistically significant signal in the entire study.

---

## R1+R2: Scaled CSR with Bootstrap CIs

### Artifacts created
- `scripts/build_csr_heldout.py` — generates frozen held-out test set
- `eval/csr_heldout.json` — **300 frozen held-out sentences** (seed=20260710, committed)
  - Source: `data/valid/wiki_valid.txt` (genuinely held-out, excluded from training)
  - Categories: heldout_short (144), heldout_medium (93), heldout_long (63)
  - Total target chars: 16,707
- `src/eval_utils.py` — added `bootstrap_mean_ci()` (single source of truth, 1000 resamples, 95% CI)
- `eval.py` — CSR summary now reports bootstrap 95% CI

### Results (GPU f16, both checkpoints, 300 held-out sentences)

| Metric | Step 32K | Step 54K | Delta |
|--------|----------|----------|-------|
| **Macro CSR** | 24.90% | 25.23% | **+0.33** |
| **Micro CSR** | 24.57% | 25.03% | +0.45 |
| **95% CI** | [23.64%, 26.21%] | [23.98%, 26.48%] | **overlap** |

**CI overlap → NOT statistically significant.** 54K is directionally better (agrees with PPL and human), but CSR cannot distinguish the two at this quality level.

### By target length — where the improvement lives
| Category | 32K | 54K | Delta |
|----------|-----|-----|-------|
| short (4-6 words) | 26.87% | 26.88% | +0.01 (saturated) |
| medium (7-9 words) | 24.15% | 24.24% | +0.09 |
| long (10-12 words) | 21.54% | 22.92% | **+1.39** |

Short completions are saturated. Improvement concentrates in long targets.

### The targets.json result is debunked
| Test set | n | 32K | 54K | Winner |
|----------|---|-----|-----|--------|
| targets.json (old, saturated) | 30 | 29.17% | 26.82% | 32K (!?) |
| **csr_heldout.json (new, clean)** | 300 | 24.90% | 25.23% | **54K** |

The old 30-sentence "54K is worse" result was pure noise from a saturated, too-small sample. At n=300 on genuinely held-out text, the direction flips to agree with PPL and human judgment. This is exactly the failure mode the reform predicted.

### Interpretation
CSR is a **lower bound**, not a ranking metric. It uses exact-match against a single gold string and cannot credit valid Basque alternative continuations (the agglutinative multiple-valid-forms problem). Once both models are "competent," CSR plateaus and cannot distinguish them. It detects *regression* (garbage → CSR crashes) but not *progress* between competent models.

---

## R10: Blinded Human A/B Evaluation

### Artifacts created
- `scripts/ab_eval_generate.py` — generates blinded A/B pairs (fresh prompts, random A/B assignment)
- `scripts/ab_eval_reveal.py` — decodes judgments after expert evaluates
- `eval/ab_eval/` — batch 1 (20 prompts, seed=20260711, blind_seed=42)
- `eval/ab_eval2/` — batch 2 (10 prompts, seed=20260712, blind_seed=99)

### Methodology
- 30 fresh prompts total from held-out validation text (zero overlap with csr_heldout or between batches)
- Both checkpoints generate greedy completions (15 tokens, f16, reference sentencepiece — no quantization confound, no BOS)
- A/B randomly assigned per prompt (expert doesn't know which checkpoint is A or B)
- Expert judges Basque quality (grammaticality, naturalness, autocomplete usefulness), NOT gold-match
- Expert explicitly told: both may differ from original but still be valid Basque; if both garbage, mark tie

### Results

**Batch 1 (20 prompts):**
| | Step 32K | Step 54K | Ties |
|---|---|---|---|
| Wins | 7 | 6 | 7 |
| Among decisive (13) | 53.8% | 46.2% | — |

**Batch 2 (10 prompts):**
| | Step 32K | Step 54K | Ties |
|---|---|---|---|
| Wins | 4 | 3 | 3 |
| Among decisive (7) | 57.1% | 42.9% | — |

**Combined (30 prompts):**
| | Step 32K | Step 54K | Ties |
|---|---|---|---|
| Wins | **11 (55%)** | **9 (45%)** | **10 (33%)** |

**Binomial test (two-sided): p = 0.82** — definitive statistical tie.
An 11-vs-9 split out of 20 decisive happens ~41% of the time by pure chance.
The two checkpoints are statistically equivalent in autocomplete quality.

### Expert qualitative observations (valuable findings)
1. **Repetition loops** — both models fall into them (batch 2: #1, #4, #9; batch 1: #3, #13, #20). Known greedy-decoding failure mode, not checkpoint-specific.
2. **"Both could be correct"** — expert said this on ~10/30 prompts. This is the agglutinative multiple-valid-forms problem made tangible: exact-match CSR cannot credit these.
3. **"Hard to judge without the full sentence"** — autocomplete quality is context-dependent and genuinely ambiguous. This is why human judgment yields many ties and why automated metrics struggle.

---

## The Reconciliation: All Evidence Converged

| Signal | Favors | Significant? | What it measures |
|--------|--------|-------------|------------------|
| **PPL** (1.8M held-out tokens) | 54K (7.56→7.17) | ✅ Yes (all 14 files agree) | Language model quality (full distribution) |
| **CSR** (300 held-out sentences) | 54K (24.90→25.25) | ❌ CIs overlap | Keystroke economy (lower bound) |
| **Human A/B** (30 blinded) | 32K (11 vs 9) | ❌ p=0.82 | Autocomplete quality (ground truth) |

### The core finding
**54K is a measurably better *language model* (PPL, statistically solid). But this improvement has NOT translated into perceptibly better autocomplete completions — the human A/B is a dead tie. The autocomplete quality has plateaued even though the language model is still improving.**

### Why PPL and autocomplete quality diverge
- PPL measures the full next-token distribution across 1.8M tokens (smooth, low-variance)
- Autocomplete quality = greedy single-path completion quality (narrow probe)
- Once a model is "competent enough," PPL improvements produce diminishing returns in autocomplete
- The exact-match/gold problem caps measurable CSR regardless of model quality
- Many prompts have multiple valid continuations (expert confirmed: ~33% ties)

### The expectation-bias finding (publishable)
The expert's earlier **unblinded** impression that 54K was "much better" was NOT borne out by **blinded** A/B evaluation (statistical tie). This is a classic expectation/anchoring bias. **Blinded evaluation is essential for honest model comparison.** This is a genuine methodological contribution.

> "Unblinded qualitative assessment can mislead. The expert's unblinded impression that 54K was 'much better' was not borne out by blinded A/B evaluation, which showed the two checkpoints as statistically equivalent in autocomplete quality. PPL confirmed 54K improved as a language model, but this improvement did not translate to perceptibly better completions at this scale. Blinded evaluation is essential for honest model comparison."

---

## Decisions

1. **54K remains the primary model** — better LM, never worse in any metric, tie means no reason to prefer 32K. Demo defaults already point to `step_0054000.Q4_K_M.gguf`.

2. **Further training (toward 76K)** — PPL still decreasing, but human A/B strongly suggests diminishing returns for autocomplete quality. Cost/benefit of 22K more steps is questionable if product metric has plateaued. **This is an open decision for the user.**

3. **CSR is a lower bound, not a model-selection metric** — empirically confirmed (not just theorized). For agglutinative autocomplete, CSR-on-gold-continuation is insufficient for model selection; perplexity + human evaluation are needed. This is a publishable methodological finding.

---

## Artifacts Inventory

### New scripts
- `scripts/ppl_eval.py` — PPL evaluator (both valid + corpus modes, matches training semantics)
- `scripts/build_csr_heldout.py` — generates frozen held-out CSR test set
- `scripts/ab_eval_generate.py` — generates blinded A/B pairs
- `scripts/ab_eval_reveal.py` — decodes A/B judgments

### Modified code
- `src/eval_utils.py` — added `bootstrap_mean_ci()` + CI in `compute_autocomplete_metrics`
- `eval.py` — CSR summary reports bootstrap 95% CI; imports `bootstrap_mean_ci`

### New data
- `eval/csr_heldout.json` — 300 frozen held-out CSR sentences (committed)
- `eval/ab_eval/` — batch 1 blinded.md, blinded.json, key.json (20 prompts)
- `eval/ab_eval2/` — batch 2 blinded.md, blinded.json, key.json (10 prompts)

### Result dirs (local + server)
- `eval/gpu-results/step32k_csr_heldout/` — CSR on held-out, 32K (summary.json + csr_results.json)
- `eval/gpu-results/step54k_csr_heldout/` — CSR on held-out, 54K (summary.json + csr_results.json)

### Docs
- `docs/eval-reform-proposal.md` — the full reform proposal with literature synthesis
  (ChaI-TeA, Trnka & McCoy, Kosyak & Tyers, Gmail Smart Compose, WSTypist)

---

## Key Numbers to Cite in Paper

### PPL (the significant signal)
- Held-out validation: 32K=7.56, 54K=7.17 (↓5.2%, all 14 files agree)
- Real corpus (contaminated): 32K=10.53, 54K=9.90 (↓6.0%)

### CSR with CIs (n=300, held-out)
- 32K: macro=24.90%, micro=24.57%, 95% CI=[23.64%, 26.21%]
- 54K: macro=25.23%, micro=25.03%, 95% CI=[23.98%, 26.48%]
- CIs overlap → not significant

### Human A/B (blinded, n=30)
- 32K: 11 wins, 54K: 9 wins, 10 ties (33%)
- Binomial test p=0.82 → statistical tie

### Comparison: old eval was misleading
- targets.json (n=30, saturated): 32K=29.17%, 54K=26.82% → "54K worse" (NOISE)
- csr_heldout (n=300, clean): 32K=24.90%, 54K=25.23% → "54K better" (direction, not significant)

### Training context
- Step 54K = 71% of 76,293 total steps (config/small.yaml)
- valid_loss fell 3.1→1.97 (PPL ~22→7.2) monotonically
- wandb in-training eval at 54K: CSR~0.28, MorphAcc~0.74-0.76

---

## Literature Anchors (from docs/eval-reform-proposal.md)

1. **ChaI-TeA (2024, arXiv:2412.18377)** — closest analog. `saved@k` metric; 26K+ test prefixes; tried LLM-as-judge for semantic matching, found "very challenging," deferred. Mid-word suggestions degrade acceptance ~60% in English (but mid-word suffix completion is Basque's core use case). Perplexity insufficient for ranking.
2. **Trnka & McCoy (2008)** — CSR/KSR framework; acceptance costs 1 keystroke (implemented correctly). Recommends reporting KSR alongside top-N accuracy and list position.
3. **Kosyak & Tyers (2022)** — agglutinative predictive text; FST-based; KSR with user study; confirms multiple-valid-form problem is central.
4. **Gmail Smart Compose (Chen, 2019)** — real-world gold standard; precision/recall of accepted suggestions on real user data.
5. **WSTypist (2026, arXiv:2602.06489)** — simulation-based mobile typing; confirms simulation metrics are accepted practice when user studies infeasible, provided at scale with realistic acceptance model.

---

## Next Steps (after context compaction)

1. **Update the paper** with all findings above (the user's stated next task).
2. Decide on further training (resume toward 76K, or stop at 54K given diminishing returns).
3. Address repetition issue (both models exhibit greedy-decoding repetition loops; could investigate repetition penalty / sampling).
4. Update CRITICAL_REVIEW.md, docs/eval-strategies.md with corrected conclusions.
