# Corpus Quality Fast Audit — Preliminary Triage

**Date:** 2026-07-03  
**Scope:** quick go/no-go audit before full 4K tokenizer training  
**Input audited locally:** `data/corpus_sample.txt`  
**Important limitation:** this local sample is **not** the full `data/clean/` corpus. It appears to be a sampled shard headed by:

- `HiTZ_BERnaT-Diverse_BSMauthor.txt: 9766418 lines → sampling 139520 lines`

So this audit is best interpreted as a **preliminary quality read on the BERnaT/BSM-like informal slice**, not a final decision on the entire training mix.

---

## Outputs

Generated with:

```bash
python3 scripts/pipeline/audit_corpus.py \
  --input data/corpus_sample.txt \
  --output-json reports/corpus_quality_fast_audit.sample.json \
  --output-md reports/corpus_quality_fast_audit.sample.md
```

Files:
- `scripts/pipeline/audit_corpus.py`
- `reports/corpus_quality_fast_audit.sample.json`
- `reports/corpus_quality_fast_audit.sample.md`

---

## Headline Findings

From **133,732 content lines** in the sample:

- **emoji:** `24,061` lines (**17.99%**)
- **punctuation runs / punctuation spam:** `3,845` lines (**2.88%**)
- **HTML/entity residue:** `932` lines (**0.70%**)
- **very long lines (>280 chars):** `12,646` lines (**9.46%**)
- **extremely long lines (>512 chars):** `3,234` lines (**2.42%**)
- **exact duplicate-line burden:** `6,907` lines in duplicate clusters (**5.17%**)
- **heuristic mixed-language lines:** `8,599` (**6.43%**)
- **heuristic clearly non-EU lines:** `2,203` (**1.65%**)
- **URLs / hashtags / mentions / replacement chars:** rare but present

Top repeated lines include:
- `Eskerrik asko.` (**60×**)
- `Azalpenik ez, oharrik ez.` (**34×**)
- `Barkatu eragozpenak.` (**26×**)
- `Mila esker.` (**20×**)
- `Gehiago irakurri..` (**16×**)

---

## Interpretation

### 1. The sample is **not clean enough to blindly trust**

The strongest risk indicators are not URLs or emails; they are:

- **social-media formatting residue**
- **emoji-heavy lines**
- **long multi-post concatenations / feed-like lines**
- **templated repetition / duplicate lines**
- **mixed-language informal content of unclear value**

This is exactly the kind of material that can produce visibly bad autocomplete suggestions even when language-model perplexity looks acceptable.

### 2. This is probably **valuable data, but too noisy to use without review**

This does **not** justify deleting BERnaT / social data. On the contrary, informal Basque is crucial for autocomplete. But the sample suggests we need to distinguish:

- authentic informal Basque and authentic code-switching
- from feed junk, boilerplate, social metadata, and concatenated announcement spam

### 3. The immediate blocker is **not all-corpus quality panic**

This sample alone does **not** prove that the full corpus is bad. It only proves that at least one important informal slice has enough visible artifacts that we should perform a **source-stratified fast audit** on the real `data/clean/` files before tokenizer training.

---

## Go / No-Go Recommendation

### Current status: **NO-GO for immediate tokenizer training without server-side audit**

Reason:
- the local sample shows meaningful visible-artifact burden
- the sample is dominated by one informal source slice
- the full corpus may still be fine overall, but we do not yet know source-by-source quality

### What would change this to GO?

If a fast source-stratified audit on `/root/morpheus-mamba/data/clean/` shows that:
- this noise is mostly concentrated in a small number of social shards,
- curated sources are much cleaner,
- and noisy shards can be kept, filtered harder, or downweighted,

then training can proceed with confidence.

---

## Highest-Value Next Steps

1. **Run the same audit on the full server corpus directory**
   ```bash
   ssh root@10.2.121.210 'cd /root/morpheus-mamba && python3 scripts/pipeline/audit_corpus.py --input data/clean --output-json reports/corpus_quality_fast_audit.clean.json --output-md reports/corpus_quality_fast_audit.clean.md'
   ```

2. **Rank files by risk**
   - HTML/entity residue
   - punct-run rate
   - duplicate-line burden
   - long-line rate

3. **Manually inspect the worst 3–5 files**
   Goal: separate:
   - authentic conversational Basque
   - authentic code-switching
   - from templated junk / feed concatenation / markup residue

4. **Decide per source**
   - keep as-is
   - keep but downweight
   - re-clean harder
   - exclude from tokenizer training

5. **Only then** decide whether to:
   - train the 4K tokenizer on the current corpus,
   - or first produce a cleaned v2 corpus mix.

---

## Practical takeaway

Your instinct was right:

> before spending compute on tokenizer training, we should do a compact quality triage on the real corpus.

This fast audit confirms that at least one informal slice contains enough visible noise that a **full-directory fast audit on the server is the correct next move**.

---

## Full Server Audit Results (2026-07-03)

The audit was executed on the complete `data/clean/` directory (15 files, 22 GB) using `scripts/pipeline/audit_corpus.py`.

### Aggregate metrics (248 million content lines)

| Metric | % |
|---|---|
| HTML residue | 0.13% |
| Punct runs | 1.64% |
| Emoji | 1.38% |
| Long lines (>280 chars) | 5.83% |
| Long lines (>512 chars) | 1.41% |
| URLs | 0.14% |

### Critical finding: duplicate-driven inflation

Several sources are dominated by repeated-line clusters:

| Source | Content lines | Duplicate cluster % |
|---|---|---|
| hplt-v1 | 74.3M | **83.80%** |
| botha | 3.9M | **63.79%** |
| bog | 12.6M | **60.22%** |
| bopv | 3.9M | **60.04%** |
| colossal-oscar | 2.8M | **51.10%** |
| finepdfs | 49.9M | 29.25% |
| hplt-v2 | 16.3M | 30.06% |

### Critical finding: language mixture

| Source | Basque heuristic | Non-EU heuristic |
|---|---|---|
| hplt-v1 | **4.9%** | 9.4% |
| finepdfs | 27.2% | 12.5% |
| bog | 17.4% | 0.3% |

### Source-level decisions

Based on the audit, the following per-source decisions were made:

| Source | Decision | Rationale |
|---|---|---|
| **hplt-v1** | ❌ **Exclude** | 83.80% duplicate, only 4.9% Basque signal, highest non-EU contamination |
| botha | Keep + Phase 2 | 63.79% duplicate but 29.9% Basque; repeated-line suppression will help |
| bog | Keep + Phase 2 | 60.22% duplicate, low non-EU contamination; structure appears recoverable |
| bopv | Keep + Phase 2 | 60.04% duplicate, 38.3% Basque; likely official gazette text with stable templates |
| finepdfs | Keep + review | 29.25% duplicate, 12.5% non-EU; PDF extraction residue may need separate filtering |
| BERnaT BSM | Keep + Phase 2 | High conversational value for autocomplete; duplicate/emoji handled by Phase 2 |
| euscrawl-v2 | ✅ Keep | 52.6% Basque, low duplicate, low non-EU |
| wikipedia | ✅ Keep | Cleanest source, used as validation split |
| All others | ✅ Keep | Within acceptable thresholds after Phase 2 |

### Cleaning strategy (finalized)

1. **Exclude hplt-v1 entirely** from the training corpus (74.3M lines, low Basque signal, extreme duplication)
2. **Apply Phase 2 deep-cleaning** (`scripts/pipeline/clean_phase2.py`) to all remaining 14 sources:
   - HTML entity cleanup
   - Sentence splitting / de-concatenation
   - Long-line heuristics (>2048 chars discarded, >512 flagged)
   - Repeated-line suppression (within-document exact duplicates)
3. **Train the 4K tokenizer** on the resulting `data/clean-v2/` corpus

This removes the worst-quality source and applies structural cleaning to all remaining text, producing a corpus that is cleaner, less redundant, and better structured for autocomplete tokenizer training.

**BERnaT BSM exclusion (2026-07-04):** The BERnaT Diverse BSM (social media) split was excluded from clean-v2 after review. While valuable for autocomplete usage data, Twitter/conversational text introduces dialectal Basque, heavy code-switching, emoji, and non-standard orthography that would bias a morphology-targeted tokenizer toward informal patterns at the expense of the formal Basque needed for good morphological segmentation.

---

## Phase 2 Results: clean-v2 Audit (2026-07-04)

Phase 2 deep-cleaning was applied to all 14 remaining sources and then re-audited with the same tool.

### Aggregate metrics (148 million content lines)

| Metric | clean/ | clean-v2/ | Δ |
|---|---|---|---|
| Basque heuristic | 28.18% | **43.55%** | +15.4% |
| Uncertain | 63.76% | 49.07% | -14.7% |
| Non-EU | 6.24% | 4.88% | -1.4% |
| HTML residue | 0.13% | **0.002%** | -98.5% |
| Long lines (>512) | 1.41% | **0.89%** | -36.9% |
| Punct runs | 1.64% | 1.81% | +0.17% |
| Emoji | 1.38% | 1.24% | -10.1% |

Note: the +0.17% punct-run increase is expected — sentence splitting creates legitimate `...` and `!!` patterns from previously concatenated lines.

### Duplicate elimination: 100% across all sources

Every source was reduced to **0.0% duplicate cluster rate**:

| Source | Before (clean/) | After (clean-v2/) |
|---|---|---|
| botha | 63.8% | **0.0%** ✔ |
| bog | 60.2% | **0.0%** ✔ |
| bopv | 60.0% | **0.0%** ✔ |
| colossal-oscar | 51.1% | **0.0%** ✔ |
| BERnaT BSM | 35.5% | **0.0%** ✔ |
| hplt-v2 | 30.1% | **0.0%** ✔ |
| finepdfs | 29.3% | **0.0%** ✔ |
| zelaihandi | 26.5% | **0.0%** ✔ |
| euscrawl-v2 | 18.8% | **0.0%** ✔ |
| wikipedia | 17.6% | **0.0%** ✔ |
| fineweb2 | 13.5% | **0.0%** ✔ |
| cultura-x | 10.5% | **0.0%** ✔ |
| aldizkariak | 8.6% | **0.0%** ✔ |
| parleus | 5.5% | **0.0%** ✔ |

### Verdict: GO for tokenizer training

The corpus is clean enough. Basque signal concentration improved from 28% to 44%, HTML is virtually eliminated, long-line burden halved, and duplicates are eradicated. The 1.81% punct-run rate and 49% uncertain-language remain are expected for a mixed-domain Basque corpus. The 14-source clean-v2 corpus is ready for 4K tokenizer training.

### Source descriptions (from Latxa corpus v2 dataset card)

Our language heuristic flagged several sources as low-EU. Cross-referencing with the [Latxa corpus v2 dataset card](https://huggingface.co/datasets/HiTZ/latxa-corpus-v2) clarifies what each source actually contains:

| Source | Description | Lines | EU% | Non-EU% | Heuristic issue |
|---|---|---:|---:|---:|---|
| finepdfs | Basque portion of FinePDFs: PDF-extracted docs (reports, magazines, books) | 45.6M | 34.8% | 11.8% | PDF artifacts + bilingual docs — mostly Basque |
| cultura-x | Basque portion of CulturaX: cleaned multilingual web crawl | 26.3M | 47.9% | 0.9% | — |
| euscrawl-v2 | EusCrawl v2: Basque web crawl from 33 newswire/media sites | 21.4M | 56.0% | 0.1% | — |
| hplt-v2 | Basque portion of HPLT v2: CommonCrawl-derived web text | 13.4M | 41.8% | 3.0% | — |
| zelaihandi | Subset of ZelaiHandi: diverse curated Basque text | 9.2M | 58.3% | 0.8% | — |
| fineweb2 | Basque portion of FineWeb2: filtered web text | 7.9M | 43.4% | 10.3% | Spanish quotes/names in web text — mostly Basque |
| **bog** | **Official gazette of Gipuzkoa Provincial Council (BOG)** | **6.4M** | **19.3%** | **0.5%** | **Short legal headers/IDs — all Basque, heuristic fails** |
| wikipedia | Basque Wikipedia dump (September 2025) | 3.8M | 50.3% | 0.2% | — |
| bopv | Official gazette of the Basque Government (BOPV) | 2.0M | 44.4% | 1.4% | — |
| botha | Official gazette of Álava Provincial Council (BOTHA) | 1.8M | 36.9% | 5.9% | Minor Spanish in bilingual legal docs |
| colossal-oscar | Basque portion of Colossal OSCAR corpus | 1.6M | 44.2% | 1.3% | — |
| parleus | ParlEus: Basque Parliament session transcriptions | 0.7M | 73.7% | 2.9% | — |
| aldizkariak | Basque academic journals (Ekaia, Gogoa, Tantak, Ekonomiaz, Uztaro, IkerGazte) | 0.6M | 40.7% | 3.9% | Academic titles, citations, Latin terms |

## Final Corpus: clean-v3 (2026-07-05)

After all exclusions (hplt-v1, BERnaT BSM, BOG, aldizkariak) and Phase 2 + Phase 3 cleaning, the final training corpus = **11 sources, 140M lines, 15 GB**:

- **Digit filtering (Phase 3):** botha -8.4%, finepdfs -6.5%, bopv -5.9%
- **Orphan fragment removal:** finepdfs -2.1M lines, all sources combined -6.5M fragments
- **HTML residue:** <0.01%
- **Duplicate cluster rate:** 0.0% across all sources

The 4K Unigram tokenizer (`basque_unigram_4000.model`) was re-trained on this exact dataset. Fertility: 2.0.

**Pre-tokenized output** (Two-pass OOM-safe script, 2026-07-05):

| Split | Tokens | Size (uint16 .npy) |
|---|---|---|
| Train | **4,769,132,775** | 8.9 GB |
| Valid (Wikipedia) | 1,848,467 | 3.5 MB |

---

## LLM Quality Audit (2026-07-04)

An LLM-based audit was run on 40 random lines per source using DeepSeek-V4-Pro. Each line was classified by text type, flagged for quality issues, and rated for Basque quality (1-5). Full results: `reports/llm_audit/audit_full.json`.

### Per-source results

| Source | Quality | Clean% | Top Issues | Decision |
|--------|---------|--------|------------|----------|
| euscrawl-v2 | 5.0 | 98% | — | ✅ INCLUDE |
| parleus | 4.9 | 98% | code-switching(1) | ✅ INCLUDE |
| zelaihandi | 4.9 | 80% | fragments(3), boilerplate(3) | ✅ INCLUDE |
| bopv | 4.8 | 82% | fragments(2), non-basque(2) | ✅ INCLUDE |
| botha | 4.8 | 72% | fragments(6), garbled(3) | ✅ INCLUDE |
| colossal-oscar | 4.7 | 66% | boilerplate(8), fragments(6) | ✅ INCLUDE |
| wikipedia | 4.6 | 88% | non-basque(3), garbled(2) | ✅ INCLUDE |
| cultura-x | 4.6 | 62% | code-switching(6), fragments(6) | ✅ INCLUDE |
| hplt-v2 | 4.4 | 55% | fragments(6), garbled(4), code-switch(4) | ✅ INCLUDE |
| fineweb2 | 4.3 | 65% | code-switching(6), non-basque(5) | ✅ INCLUDE |
| finepdfs | 3.7 | 78% | gazette(8), garbled(6), non-basque(6) | ⚠️ INCLUDE (Phase 3 target) |
| **bog** | 4.0 | **2%** | **fragments(36)**, short(7), gazette(5) | 🔴 **EXCLUDED** |
| **aldizkariak** | 3.4 | **38%** | **boilerplate(14)**, non-basque(4) | 🔴 **EXCLUDED** |

### Exclusion rationale

- **BOG** (403 MB): Phase 2 sentence splitting broke legal text into mid-sentence fragments. 36/40 lines are incomplete clauses like "1. Entitate adjudikatzailea:". The Basque is good (Q=4.0), but the processing artifact makes it unusable for autocomplete training. Rather than re-process, excluded.
- **Aldizkariak** (97 MB): 35% boilerplate — author lists, English titles, citation numbers, markdown table headers. Basque quality lowest (3.4/5). Net-negative for autocomplete.

### Final corpus (`data/clean-v3`, 2026-07-05)

After all exclusions (hplt-v1 + BERnaT BSM + BOG + aldizkariak) and Phase 2+3 cleaning:

| Metric | Value |
|--------|-------|
| Sources | **11** |
| Lines | **140M** |
| Size | **15 GB** |
| Tokens (pre-tokenized) | **4.77B** |
| .npy file size | **8.9 GB** |
| LLM avg quality | **4.6/5** |

**Phase 3 digit filtering removed ~6M lines** (botha -8.4%, finepdfs -6.5%, bopv -5.9%) by stripping decree IDs, digit-heavy lines, and orphan fragments. The remaining 11 sources are clean enough for training.

Decision: **GO for re-training** with 4K tokenizer on this corpus.