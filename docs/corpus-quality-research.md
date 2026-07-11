# Corpus Quality Research for Basque Autocomplete

> **Purpose:** define the next deep-research frontier for Morpheus after architecture, tokenization, and evaluation.
>
> **Focus:** corpus quality for **Basque predictive autocomplete**, not generic LM pretraining.
>
> **Date:** 2026-07-03

---

## Executive Summary

Tokenization and evaluation are no longer the main research bottlenecks for Morpheus. The remaining high-leverage research problem is **corpus quality**: what text should be kept, removed, downweighted, split, or re-mixed so that a small on-device model learns **clean, useful, context-appropriate completions** for Basque users.

The literature suggests five relevant subfields:

1. **Artifact taxonomy** — identify and quantify the exact kinds of noise that survive cleaning.
2. **Language-mixture analysis** — distinguish harmful contamination from authentic Basque–Spanish code-switching.
3. **Source-quality ablation** — determine which sources actually help autocomplete, rather than assuming “more curated” is always better.
4. **Deduplication / near-duplication** — remove repeated templates, boilerplate, and train–eval overlap that bias both training and evaluation.
5. **Prompt-distribution matching** — make training/eval inputs resemble real autocomplete prefixes rather than generic full-text language modeling.

### Bottom line

For Morpheus, the most important open questions are **not**:
- “Which architecture should we use?”
- “Should we use BPE or Unigram?”
- “How should we evaluate morphology?”

Those are already in good shape.

The most important open questions **are**:
- Which sources generate the cleanest and most useful Basque suggestions?
- Which mixed-language lines reflect real usage, and which are just crawl contamination?
- How much duplicate/templated text is inflating collocation quality and polluting suggestions?
- How far is the training distribution from the actual prefix distribution seen in an autocomplete UI?

This document synthesizes the relevant literature and turns it into a **Morpheus-specific research agenda**.

---

## 1. Why corpus quality matters more for autocomplete than for generic LM training

Several recent papers complicate the naive assumption that “cleaner text always yields better language models.”

### 1.1 Low-resource corpus quality does not map cleanly to downstream scores

Artetxe et al. (2022) studied Basque monolingual corpora and found **no clear correlation** between corpus size or human-perceived quality and downstream LM performance on their chosen tasks. Van Noord et al. (2024) generalized this across 11 lower-resourced European languages: MaCoCu and OSCAR looked better in human annotation, but CC100 often performed best in downstream encoder benchmarks.

This does **not** mean corpus quality is irrelevant for Morpheus. It means:

- quality effects may be **task-specific**;
- human quality labels like “publishable text” are only one axis;
- encoder-style downstream tasks may understate the cost of visible generation errors.

### 1.2 Generative autocomplete is unusually sensitive to visible noise

Autocomplete is not hidden representation learning. The model’s output is displayed directly to users, often after only a few typed characters. That makes the system more sensitive to:

- URLs,
- mentions/hashtags,
- broken encoding,
- duplicated boilerplate,
- punctuation spam,
- fragmentary text,
- mixed-language garbage,
- overlearned template continuations.

A user will tolerate some internal representation noise if NER improves by 0.3 F1; they will **not** tolerate suggestions like:
- `https://t.co/...`
- `@username`
- malformed Unicode,
- broken word fragments,
- repeated news boilerplate.

### 1.3 Basque adds a special difficulty: not all “contamination” is bad

Basque is a minority language in intense contact with Spanish and, in some regions, French. That means:

- some Spanish inside Basque text is **authentic user behavior**,
- some Spanish inside Basque corpora is **crawl contamination**,
- some mixed text is **valuable for autocomplete**,
- some mixed text is **poisonous for morphology and lexical prediction**.

So Morpheus cannot adopt a simplistic “remove all non-Basque text” policy.

---

## 2. What the literature already tells us

## 2.1 Curated and source-aware collection matters for Basque

The Latxa paper (Etxaniz et al., 2024) is highly relevant because it is the strongest recent Basque corpus-construction reference. Their design principle was explicit: **quality over quantity**.

Key choices from Latxa:

- They combined curated sources such as **EusCrawl**, **Wikipedia**, **Egunkaria**, and **Booktegi** with larger noisier web sources.
- They used a **priority order** during deduplication:
  1. Wikipedia / EusCrawl / Egunkaria / Booktegi
  2. CulturaX / Colossal OSCAR
  3. HPLT
- They report that **HPLT required the most aggressive filtering**, including large losses after language ID and document-quality scoring.
- They explicitly use normalization, deduplication, filtering, and language ID.

**Implication for Morpheus:** source provenance should not be treated as metadata fluff. It should be a first-class variable in training and ablation.

## 2.2 Human quality annotation is useful, but incomplete

Van Noord et al. (2024) propose a practical 5-level annotation scheme:

1. wrong language / not language
2. not running text
3. partially running text
4. running text but slightly non-standard
5. publishable text

This is an excellent starting point for Morpheus, but it misses autocomplete-specific phenomena such as:

- social-media markers that are syntactically valid but useless as suggestions,
- code-switched text that is useful for users but looks “impure,”
- templated commercial boilerplate that is grammatical but repetitive,
- prefix-distribution mismatch.

**Implication:** reuse the annotation idea, but extend it for autocomplete.

## 2.3 Language identification in low-resource web text is error-prone

Caswell et al. (2020) show that web-scale language identification becomes unreliable for many lower-resource languages. This is especially relevant for Basque, where:

- short strings are common,
- named entities often resemble Spanish/French,
- code-switching is real,
- noisy social text contains abbreviations, hashtags, dialect, and slang.

**Implication:** line-level language filtering should not rely on one hard LID threshold from one tool.

## 2.4 Crawl corpora contain the same kinds of junk again and again

CCNet (Wenzek et al., 2020), C4 documentation (Dodge et al., 2021), and Lee et al. (2022) all show recurring web-corpus pathologies:

- wrong-language passages,
- boilerplate,
- templated pages,
- repeated legal/disclaimer text,
- machine-generated pages,
- train–validation overlap,
- near-duplicate pages differing only in slots like names, dates, or places.

For autocomplete, these matter twice:

1. they bias the model toward unnatural visible outputs;
2. they inflate evaluation by repeating easy memorized continuations.

## 2.5 Deduplication is not just an efficiency optimization

Lee et al. (2022) show that deduplicating training data:

- reduces memorized emission by about **10×**,
- can reduce dataset size substantially,
- does not hurt perplexity and can improve it,
- reduces train–test leakage.

For Morpheus, the most important implication is qualitative: deduplication should reduce the model’s tendency to surface over-repeated junk and should make checkpoint comparisons more honest.

## 2.6 Real autocomplete distributions differ from synthetic prefix distributions

Smart Compose (Chen et al., 2019) emphasizes real-time, user-facing generation constraints. AmazonQAC (Everaert et al., 2024) makes the distributional point explicit: **synthetic prefixes are a poor proxy for real typed prefixes**.

AmazonQAC reports:

- real typed prefix sequences are often **non-linear** due to deletion and revision;
- **13%** of accepted completions do not literally continue the final typed prefix;
- context from past behavior significantly improves autocomplete;
- naturalistic prefix logs are much more informative than synthetic prefixes cut from final queries.

**Implication for Morpheus:** a corpus can be “good Basque text” and still be a poor fit for autocomplete if its prefix distribution is wrong.

## 2.7 Basque–Spanish code-switching is a real target behavior, not mere noise

Heredia et al. (2025) introduce **EuskañolDS**, a naturally sourced Basque–Spanish code-switching corpus. Important findings:

- code-switching is common in both formal and informal Basque usage;
- social and parliamentary sources both exhibit switching, but with different patterns;
- low-confidence language identification is a strong heuristic for discovering mixed-language instances;
- much naturally occurring mixed text is short, informal, and context-dependent.

**Implication:** mixed-language text must be separated into at least two classes:
- **authentic, user-relevant code-switching**;
- **irrelevant contamination**.

---

## 3. Subfield A — Artifact Taxonomy

## 3.1 Research question

What noise classes remain in the corpus after current cleaning, how frequent are they, and which ones most damage autocomplete quality?

## 3.2 Why this matters for Morpheus

Your own project notes already show the symptoms:

- URLs after questions,
- @mentions after greetings,
- hashtag/news spam,
- encoding artifacts,
- fragmentary whitespace/token boundaries,
- punctuation spam,
- malformed Unicode.

These are not theoretical. They are visible in inference.

## 3.3 Taxonomy proposed for Basque autocomplete

Start with the van Noord et al. (2024) annotation backbone, then refine it into autocomplete-specific categories.

### Tier 1: coarse quality labels

- **WL** — wrong language / not language
- **NR** — not running text
- **PR** — partially running text
- **RT** — running text but slightly non-standard
- **PT** — publishable text

### Tier 2: Morpheus-specific artifact labels

For every document/line, add zero or more of:

- **URL** — contains URL-like material
- **SOCIAL** — mentions, hashtags, repost markers, handles
- **MOJIBAKE** — broken encoding / replacement chars / malformed Unicode
- **HTML** — markup residue, CSS, scripts, navigation text
- **BOILERPLATE** — cookie banners, legal notices, templates, repetitive commercial text
- **LISTISH** — menus, bullet dumps, tag clouds, archive pages
- **TRUNCATED** — sentence fragments caused by bad extraction
- **NOISY_PUNCT** — repeated punctuation/emojis/symbol runs
- **TOKEN_BOUNDARY** — bad spacing, split compounds, run-together words
- **QUOTE_NEST** — quote/parenthesis corruption from extraction
- **OCRISH** — OCR-like substitutions or random caps/punctuation
- **LANG_MIX_AUTH** — authentic Basque-dominant mixed-language use
- **LANG_MIX_NOISE** — mixed-language contamination with no clear user value

## 3.4 What the literature suggests to expect

From C4, CCNet, and dedup papers, expect overrepresentation of:

- repeated site templates,
- commercial product slot-fill pages,
- page headers/footers,
- low-information structured fragments,
- malformed extraction from web DOMs.

From your own corpus history, expect overrepresentation of:

- social markers,
- news intros and headlines,
- code-switched snippets,
- encoding/whitespace artifacts.

## 3.5 Recommended research method

### Annotation sample

Sample at least **2,000 lines** stratified by source:
- curated web/news,
- Wikipedia,
- literature/book-like,
- social media,
- code-switching sources,
- any broad web dump source.

### Output

For each source, estimate:
- % WL / NR / PR / RT / PT
- % with each artifact label
- avg length
- proportion of lines likely safe for autocomplete

### Decision criterion

A source should not be judged only by “publishable text.” It should be judged by:

> **Autocomplete-safe yield** = percentage of lines that are both linguistically useful and unlikely to produce visibly bad suggestions.

---

## 4. Subfield B — Language-Mixture Analysis

## 4.1 Research question

How much non-Basque material exists in the corpus, and how much of it is genuine Basque-user behavior worth preserving?

## 4.2 Why Basque is special here

In English-centric cleaning pipelines, foreign-language spillover is usually a nuisance. In Basque, it is often part of the target usage.

Examples to distinguish:

### Useful mixed-language text
- Basque sentence with Spanish discourse marker
- Basque message with a short Spanish clause
- bilingual parliamentary discourse
- informal Euskañol from social media

### Harmful contamination
- full Spanish article accidentally kept in a Basque shard
- multilingual menus / category pages
- machine-translated mixed junk
- scraped metadata, tags, or quoted non-Basque fragments with no completion value

## 4.3 Literature-guided principles

### Principle 1: Don’t delete all mixed text

EuskañolDS shows that Basque–Spanish switching is a real, structured phenomenon.

### Principle 2: Low-confidence LID is a useful signal, not a final verdict

Heredia et al. (2025) use low-confidence FastText predictions as a way to surface code-switched instances. Caswell et al. (2020) warns that language ID is brittle in low-resource/noisy contexts.

Therefore:
- low confidence should trigger **inspection or special routing**,
- not automatic deletion.

### Principle 3: Source context matters

Mixed language in parliamentary transcriptions is different from mixed language in tweet replies, and both differ from mixed language in crawl residue.

## 4.4 Recommended label set

Each line/document should receive one of:

- **EU_CLEAN** — overwhelmingly Basque
- **EU_DOM_MIX** — Basque-dominant with authentic code-switching
- **BAL_MIX** — balanced Basque/Spanish or Basque/French
- **NON_EU_CONTAM** — mostly non-Basque and not valuable for autocomplete
- **UNKNOWN_NOISY** — too noisy for reliable language assignment

Also add:
- source id
- LID tool scores
- token ratio estimates
- presence of CS markers (switch boundary count, named entity ratio, punctuation ratio)

## 4.5 Research hypotheses worth testing

1. **EU_DOM_MIX helps informal autocomplete** more than pure curated news text.
2. **BAL_MIX is useful only up to a small fraction** and then starts hurting morphology.
3. **NON_EU_CONTAM harms more than it helps**, especially for suffix prediction.
4. Social-media mixed text may improve user realism but worsen output cleanliness unless filtered more aggressively.

## 4.6 Concrete experiment

Train small pilot models or fixed-budget adapter runs on:

- **A:** EU_CLEAN only
- **B:** EU_CLEAN + EU_DOM_MIX
- **C:** A + BAL_MIX capped at 5%
- **D:** full current mixture

Compare on:
- CSR
- MorphAcc
- visible artifact rate in generations
- human preference on 100 real typing prompts

---

## 5. Subfield C — Source-Quality Ablation

## 5.1 Research question

Which sources actually improve Basque autocomplete quality, and which only inflate size?

## 5.2 Why literature is not enough

The low-resource literature gives a cautionary lesson:
- better-looking corpora do not always win on generic downstream tasks.

But Morpheus is not training an encoder for POS/NER/COPA. It is training a **small generative autocomplete model** where visible errors are costly.

So source selection must be tested directly for **generation utility**, not borrowed from unrelated benchmark logic.

## 5.3 Strong prior from Latxa

Latxa’s source ordering is a valuable prior:

1. curated high-quality sources first,
2. massive but cleaner web sources second,
3. aggressively filtered broad web data last.

That suggests a practical Morpheus hypothesis:

> For a small autocomplete model, **high-trust sources likely dominate early quality**, while broad noisy web sources may only help after careful filtering and dedup.

## 5.4 Morpheus-specific source buckets

Use source buckets like:

- **CURATED_NEWS** — EusCrawl / Egunkaria-like
- **REFERENCE** — Wikipedia
- **LITERARY** — Booktegi / books
- **SOCIAL** — BERnaT / Twitter-like / short informal
- **CS_AUTH** — EuskañolDS / BasqueParl mixed segments
- **WEB_BROAD** — OSCAR / CulturaX / HPLT-like

## 5.5 Ablation matrix

A realistic research matrix:

| Mix | Description | Expected strength | Expected risk |
|---|---|---:|---:|
| M1 | CURATED_NEWS + REFERENCE | clean syntax, low visible junk | too formal |
| M2 | M1 + LITERARY | richer lexicon, longer syntax | less conversational |
| M3 | M2 + SOCIAL | more realistic user phrasing | more noise |
| M4 | M3 + CS_AUTH | better bilingual realism | more ambiguity |
| M5 | M4 + WEB_BROAD | more coverage/scale | template contamination |

## 5.6 Key insight to test

The main question is **not** “which source has the best average quality?”

It is:

> Which mixture gives the best tradeoff between
> - visible cleanliness,
> - conversational usefulness,
> - morphological accuracy,
> - and domain coverage?

## 5.7 Best evaluation targets for ablation

For each mixture, measure:

- **CSR** on formal prompts
- **CSR** on informal prompts
- **MorphAcc** on suffix-sensitive prompts
- **artifact suggestion rate**
- **code-switch acceptance quality**
- **human-rated naturalness**

---

## 6. Subfield D — Deduplication and Near-Duplication

## 6.1 Research question

How much duplicate and near-duplicate content remains, and how much is it distorting both training and evaluation?

## 6.2 Why this is especially dangerous for autocomplete

A duplicate-heavy corpus can make a model look good on:
- collocations,
- headlines,
- formulaic phrases,
- common templated continuations.

But that can be fake progress if the model is just overexposed to:
- site boilerplate,
- repeated article scaffolds,
- slogan/news intros,
- repeated social-media campaign posts,
- mirrored crawls,
- train–eval overlap.

For autocomplete, this creates two problems:

1. **Visible memorization:** suggestions become templatic and repetitive.
2. **Misleading metrics:** CSR may look high on easy memorized continuations while true generalization remains weak.

## 6.3 What recent work strongly supports

Lee et al. (2022):
- exact dedup is insufficient,
- near-duplication is common in web corpora,
- dedup reduces memorization sharply,
- train/test overlap can materially distort evaluation.

Common near-dup patterns include pages differing only by:
- location,
- product name,
- date,
- title,
- short slot-filled fields.

## 6.4 Recommended dedup stack for Morpheus

### Layer 1: exact document hash
Remove byte-identical or normalized-text-identical documents.

### Layer 2: line/paragraph exact repetition
Especially important for:
- repeated nav bars,
- headers/footers,
- repeated archive/menu text,
- copy-pasted social fragments.

### Layer 3: near-duplicate document detection
Use MinHash / fuzzy hashing / shingled Jaccard style methods.

### Layer 4: source-aware retention
When duplicates are found, keep according to a priority rule similar to Latxa:
- curated source > broad web source
- full running text > partial text
- Basque-dominant > mixed noisy

### Layer 5: eval contamination check
Ensure no exact or near duplicate from training leaks into:
- validation split,
- eval prompt targets,
- future human test sets.

## 6.5 Autocomplete-specific dedup metrics

In addition to generic dedup statistics, compute:

- **duplicate continuation rate**: how often identical long continuations recur after the same prefix context
- **top-prefix template concentration**: how concentrated are completions for frequent short prefixes
- **memorized boilerplate share**: fraction of generated suggestions that belong to high-frequency duplicate clusters

---

## 7. Subfield E — Prompt-Distribution Matching

## 7.1 Research question

How different is the training/evaluation distribution from actual autocomplete usage, and how should we correct for it?

## 7.2 Why this matters

A model trained on beautiful full documents may still perform poorly at autocomplete because the user task is different:

- the user stops in the middle of a phrase,
- the cursor is at the end of a line,
- the completion must be short and safe,
- the accepted suggestion may depend on discourse context,
- the typed prefix may contain revision, deletion, or misspelling.

## 7.3 What the literature says

### Smart Compose
Chen et al. (2019) shows that autocomplete is a real-time, user-behavior-sensitive product task, not just standard LM generation.

### AmazonQAC
Everaert et al. (2024) shows:
- synthetic prefixes miss real user typing behavior,
- many prefix trajectories are non-linear,
- context helps significantly,
- accepted completions are not always literal continuations of the last visible prefix.

### Context-sensitive QAC
Earlier QAC work also shows that recent context can materially improve completion quality even for short prefixes.

## 7.4 Implication for Morpheus

Even if Morpheus is trained as a causal LM, its **research protocol** should be autocomplete-shaped.

That means building datasets and evaluation slices around:

- realistic stopping points,
- sentence-internal prefix cuts,
- end-of-line cursor positions,
- short/medium/long prefix buckets,
- formal vs informal user contexts,
- code-switched contexts,
- typo/revision cases if available.

## 7.5 Research agenda here

### A. Prefix slicing from corpus text
Create synthetic prefix states from real Basque text, but do it intelligently:
- stop after 1–5 words,
- stop before case suffixes,
- stop at clause boundaries,
- stop after common discourse markers,
- stop before common MWEs.

### B. Real interaction logging
Once demo usage exists, log anonymized:
- typed prefix length,
- acceptance rate,
- rejection rate,
- revision patterns,
- contexts where users disable suggestions.

### C. Context-conditioned evaluation
Build eval sets where the same visible prefix is tested with different prior contexts.

### D. Safety/utility balancing
Measure not only “correctness” but:
- whether suggestions are too long,
- too risky,
- too boilerplate,
- too formal for informal contexts.

## 7.6 Key hypothesis

A corpus optimized for generic LM perplexity may still be mismatched for autocomplete because it underrepresents:
- unfinished thoughts,
- dialogue-like continuations,
- short pragmatic formulas,
- user-accepted shorter completions.

---

## 8. Recommended Morpheus Research Program

## Phase 1 — Audit what is already in `data/clean/`

### Deliverables
- source inventory
- per-source line counts / token counts
- 2,000-line manual quality sample
- artifact taxonomy report
- language-mixture report

### Key outputs
- `docs/corpus-quality-audit.md`
- `data/annotations/corpus_quality_sample.csv`

## Phase 2 — Build source-aware filtering labels

For every source shard, estimate:
- autocomplete-safe yield
- Basque dominance ratio
- authentic code-switching ratio
- duplicate burden
- boilerplate burden

### Decision outputs
- keep as-is
- keep but downweight
- keep only after stronger filtering
- remove from default training mix

## Phase 3 — Dedup and contamination study

- exact doc dedup
- near-dup clustering
- train/valid/eval overlap audit
- repeated-continuation concentration study

### Success criterion
A measurable drop in repeated boilerplate generations without harming CSR.

## Phase 4 — Small-scale source ablations

Run fixed-budget comparisons across 4–6 source mixtures.

### Evaluate with
- CSR
- MorphAcc
- Paradigm
- artifact rate
- bilingual realism checks
- human preference

## Phase 5 — Prompt-distribution correction

- construct prefix-sliced eval sets
- add context-conditioned prompt subsets
- if possible, begin opt-in demo logging for anonymized prefix statistics

---

## 9. Practical Recommendations for Morpheus Right Now

If the goal is to decide what to do **after resuming 4K training**, the best order is:

1. **Finish 4K training and establish the new baseline.**
2. **Audit corpus quality on the current cleaned corpus.**
3. **Run source-aware dedup and contamination checks.**
4. **Separate authentic Basque-dominant code-switching from contamination.**
5. **Run source-mixture ablations at small scale.**
6. **Only then decide whether a large recleaning / remix is worth a full retrain.**

### Highest-confidence immediate recommendations

- Do **not** treat all Spanish/French text as noise.
- Do **not** trust one LID score as a deletion oracle.
- Do **not** judge sources only by “publishable text.”
- Do **not** assume full-document LM quality transfers directly to autocomplete.
- Do **prioritize source-aware deduplication** and contamination checks.
- Do **measure autocomplete-safe yield**, not just generic cleanliness.

---

## 10. Morpheus-Specific Hypotheses Worth Testing

These are the best candidate hypotheses for the next empirical cycle:

### H1 — Curated Basque sources dominate syntax, but under-serve conversational autocomplete
Prediction: curated news/wiki/book mixtures produce cleaner but too-formal suggestions.

### H2 — A small amount of authentic mixed-language data helps user realism
Prediction: Basque-dominant code-switched data improves informal completions and bilingual acceptability.

### H3 — Broad web sources mostly help coverage after aggressive filtering and dedup
Prediction: unfiltered broad web data increases visible junk more than it helps morphology.

### H4 — Duplicate/template reduction improves visible generation more than it improves perplexity
Prediction: user-facing quality gains will exceed PPL gains.

### H5 — Prefix-distribution mismatch is currently under-measured
Prediction: models that look similar under LM-style eval will differ more strongly under realistic prefix-conditioned eval.

---

## 11. Suggested Files to Create Next

- `docs/corpus-quality-audit-plan.md`
- `scripts/pipeline/audit_corpus.py`
- `scripts/sample_corpus_by_source.py`
- `scripts/cluster_near_duplicates.py`
- `data/annotations/corpus_quality_guidelines.md`
- `eval/prefix-sliced-targets.json`

---

## References

- **Artetxe et al. (2022)**. *Does Corpus Quality Really Matter for Low-Resource Languages?* EMNLP 2022. https://aclanthology.org/2022.emnlp-main.499/
- **Bender & Friedman (2018)**. *Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science.* TACL 2018. https://aclanthology.org/Q18-1041/
- **Caswell et al. (2020)**. *Language ID in the Wild: Unexpected Challenges on the Path to a Thousand-Language Web Text Corpus.* COLING 2020. https://aclanthology.org/2020.coling-main.579/
- **Chen et al. (2019)**. *Gmail Smart Compose: Real-Time Assisted Writing.* arXiv. https://arxiv.org/abs/1906.00080
- **Dodge et al. (2021)**. *Documenting the English Colossal Clean Crawled Corpus.* EMNLP 2021. https://aclanthology.org/2021.emnlp-main.98/
- **Etxaniz et al. (2024)**. *Latxa: An Open Language Model and Evaluation Suite for Basque.* ACL 2024. https://aclanthology.org/2024.acl-long.799/
- **Everaert et al. (2024)**. *AmazonQAC: A Large-Scale, Naturalistic Query Autocomplete Dataset.* arXiv / EMNLP Industry 2024. https://arxiv.org/html/2411.04129v1
- **Heredia, Barnes, & Soroa (2025)**. *EuskañolDS: A Naturally Sourced Corpus for Basque-Spanish Code-Switching.* CALCS 2025. https://aclanthology.org/2025.calcs-1.1/
- **Lee et al. (2022)**. *Deduplicating Training Data Makes Language Models Better.* ACL 2022. https://aclanthology.org/2022.acl-long.577/ ; HTML mirror: https://ar5iv.labs.arxiv.org/html/2107.06499
- **van Noord et al. (2024)**. *Do Language Models Care About Text Quality? Evaluating Web-Crawled Corpora Across 11 Languages.* arXiv. https://arxiv.org/html/2403.08693v1
- **Wenzek et al. (2020)**. *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data.* LREC 2020. https://aclanthology.org/2020.lrec-1.494/

---

## Final Recommendation

If tokenization research answered **“how should Basque be segmented?”**, corpus quality research must answer **“what Basque should the model see in the first place?”**

For Morpheus, this is now the next deep-research problem most likely to change real user quality.