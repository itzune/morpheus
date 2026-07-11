# Evaluation Strategies for Predictive Autocompletion in Agglutinative Languages

> ## ⚠️ Eval Metric Versions — read before comparing numbers
>
> The evaluation code has two versions. **Numbers from v1 and v2 are NOT comparable.**
>
> | Version | Date | Status | Affected metrics |
> |---------|------|--------|------------------|
> | **v1** (buggy) | ≤ 2026-07-07 | Superseded | CSR (inflated), `boundary_prob_mass` (renormalized over top-K) |
> | **v2** (fixed) | 2026-07-07 | Current | CSR, MorphAcc, Paradigm — all correct |
>
> **What changed in v2** (`src/eval_utils.py`):
> 1. **CSR now charges 1 keystroke per accepted prediction** (Tab), per Trnka & McCoy (2008). v1 counted acceptance as free → CSR inflated.
> 2. **MorphAcc `boundary_prob_mass` uses true full-vocab probabilities**, not renormalized over top-K.
> 3. macro/micro CSR labels fixed (macro = per-test avg, micro = char-weighted).
>
> **What did NOT change** (v1 numbers still valid):
> - MorphAcc **hit-rate** (boolean: suffix in top-K) — unaffected by the prob bug.
> - Paradigm **Hit@K** (boolean: rank ≤ K).
> - Tokenizer fieldwork MorphAcc consistency (66.7% / 28.6%) — computed by a separate script.
> - All training loss / PPL curves.
>
> **⚠️ Important:** W&B CSR logs *during training* (e.g. `valid/csr=0.289` at step 6500) were computed by `train.py`'s inline eval, which **already charged for acceptance** — those were correct all along. The bug was only in `eval.py`'s standalone runs. This is why the two never agreed.
>
> **Action required:** Any CSR or `boundary_prob_mass` number dated before 2026-07-07 must be re-run with v2 before use in the paper or comparison tables. The hardcoded step-30K CSR of 0.836 in `scripts/experiments/eval_gpt2_baseline.py` is v1 and is flagged as such.

## Research Summary

### 1. Key Papers

**QuechuaTok: Morphological Boundary Accuracy as a Necessary Metric** (Contreras, 2026, arXiv:2606.23943)
- Introduces **MorphAcc**: proportion of tokenizer boundaries that align with morpheme boundaries from a morphological analyzer
- Key finding: BPE achieves lowest fertility (1.636) but only **6.67% MorphAcc** — it memorizes surface word forms, not morphological structure
- PRPE (morphology-aware) achieves **83.33% MorphAcc** with competitive fertility (1.797)
- **Fertility rate alone is insufficient** for agglutinative languages
- Unigram 4k: 66.67% MorphAcc — small vocab *forces* morpheme-aligned segmentation; larger vocabs overfit to surface frequency
- Applies directly to Basque: same suffixing agglutinative structure as Quechua

**Interactive Word Completion for Plains Cree** (Lane et al., ACL 2022)
- Morph-based autocompletion using finite-state morphological analyzer
- Evaluation: keystroke savings rate (KSR), number of completions accepted/rejected by users
- For agglutinative languages: morphological segmentation is both the input representation *and* the evaluation target
- Cree has similar complexity to Basque: rich verbal morphology, polysynthetic tendencies

**Evaluating Word Prediction: Framing Keystroke Savings** (Trnka & McCoy, ACL 2008)
- Classic framework for word prediction evaluation
- **Keystroke savings rate (KSR)**: percentage of keystrokes saved by accepting predictions
- Complications: KSR depends on UI design (list length, acceptance mechanism), user behavior, and input method
- Recommends: report KSR alongside top-N accuracy and list position

### 2. Established Evaluation Metrics

#### A. Intrinsic (model-only) metrics

| Metric | What it measures | Weakness for agglutinative |
|--------|-----------------|---------------------------|
| **Perplexity** | How well model predicts next token distribution | Full-sequence metric; doesn't isolate autocomplete-relevant positions |
| **Hit@K (next token accuracy)** | Is the correct next token in top-K predictions? | Single token only; 1 correct token ≠ coherent continuation |
| **Top-K token rank** | Position of the gold token in sorted predictions | Ignores multi-token coherence |
| **MorphAcc** | Do token boundaries align with morpheme boundaries? | Requires morphological analyzer; tokenizer-level, not generation-level |
| **Fertility rate** | Tokens per word | **Misleading for agglutinative** — lower = worse morphological awareness |

#### B. Extrinsic (task-level) metrics

| Metric | What it measures | Notes |
|--------|-----------------|-------|
| **Keystroke Savings Rate (KSR)** | % keystrokes saved by accepting completions | Gold standard but requires user study or simulation |
| **Character Savings Rate (CSR)** | Characters saved vs typing full text | Simpler than KSR, no UI model needed |
| **Word Completion Rate (WCR)** | % of words accepted from predictions | Platform-dependent |
| **Acceptance Rate** | How often users accept a suggestion | Requires user study |
| **Time to compose** | End-to-end typing time | Requires user study |

#### C. Task-specific for agglutinative autocomplete

| Metric | What it measures | Relevance to Morpheus |
|--------|-----------------|----------------------|
| **Morpheme-level Hit@K** | Accuracy at predicting the next morpheme (not token) | Direct: Basque suffix completion (etxe→tik) |
| **Suffix completion accuracy** | Can the model complete a partial word correctly? | Exact match for our B-category tests |
| **Multi-morpheme coherence** | Does greedy decoding produce a grammatically valid word? | Current gap: we check tokens but not morphological validity |
| **Boundary alignment** | Does the prediction respect morpheme boundaries? | Example: `etxe` → `tik` (correct) vs `etx` + `etik` (wrong split) |
| **Case paradigm coverage** | Across all 12+ Basque cases, how often is the right case predicted? | Systematic evaluation of morphological competence |

### 3. What's Wrong with Our Current Eval

| Problem | Detail |
|---------|--------|
| **Hit@K is token-level, not morpheme-level** | `etxe → tik` is ranked #22 because SentencePiece tokenizer treats `tik` as rare; a morphological tokenizer would rank it #1 |
| **No morphological boundary awareness** | We don't check if predictions align with morpheme boundaries — just if a specific token ID is top-ranked |
| **Multi-token coherence is too strict** | Exact string matching penalizes valid alternative continuations |
| **No keystroke-level metric** | We don't simulate actual typing: how many keys does user press before getting the right completion? |
| **No case paradigm coverage** | We test isolated suffixes but not the full paradigm (12+ cases, 4 numbers, definite/indefinite) |
| **Collocation tests reward memorization** | `Eskerrik → asko` will always be #1; it's a frequency artifact, not morphological skill |

### 4. Recommended Improvements for Morpheus Eval

**STATUS: IMPLEMENTED in `eval.py`** — see results below.

#### Priority 1: Fix the tokenizer-level problem

The SentencePiece Unigram tokenizer is fragmenting Basque morphology arbitrarily. This directly hurts scores:
- `etxe` + `tik` → tokenizer splits `etik` as a unit rather than `e` + `tik`
- Confirmed by MorphAcc results: top predictions for bare nouns are soft-hyphens (`\u00ad`), fragments of other words (`rik`, `en`, `ari`), not case suffixes
- Solution options:
  - **Morphological tokenizer** (like PRPE for Quechua): pre-segment with a Basque morphological analyzer, then train BPE/Unigram on morpheme-segmented text
  - **MorphAcc measurement**: use `euMor` or similar Basque morphological analyzer to score current tokenizer

#### Priority 2: Add morpheme-level evaluation
Replace or supplement Hit@K with:
- **Morpheme completion accuracy**: for a prompt ending at a morpheme boundary, does the model predict the correct next morpheme (regardless of tokenization)?
- **Case paradigm test**: for each noun root, test all 12+ case suffixes and measure accuracy
- **Suffix boundary test**: does the model place higher probability on suffix candidates that respect morphological boundaries?

#### Priority 3: Add simulated keystroke savings
- Simulate a user typing: for each character typed, check if the top-1 prediction matches the intended continuation
- Report **character savings rate (CSR)**: how many characters the user would not need to type
- This is the only metric that directly measures autocomplete utility

#### Priority 4: Keep the good parts
- Collocation tests (A-category): useful for regression detection — if `Eskerrik→asko` drops, something regressed
- Punctuation boundary tests (E-category): Basque-specific edge case, keep
- Long context test (D-category): SSM-specific, keep for detecting state degradation

### 5. Practical Implementation Plan

**Phase 1 (now): Character Savings Rate simulation**
```
def simulated_keystroke_savings(model, tok, prompt, target_completion):
    """
    Simulate keystroke-by-keystroke typing.
    For each character c in target_completion:
      - Current input = prompt + typed_so_far
      - Get model's top-1 prediction
      - If prediction starts with the next character(s) of target,
        count those as "saved"
      - Track minimum keystrokes needed to reach completion
    Returns: CSR = 1 - (keystrokes_needed / len(target_completion))
    """
```

**Phase 2 (requires morphological analyzer): MorphAcc + morpheme completion**
- Integrate a Basque morphological analyzer (euMor, Apertium, or Stanza Basque)
- For each eval test, decompose expected completion into morphemes
- Check: does the model's prediction respect those morpheme boundaries?
- Requires pre-tokenizing eval inputs with morphological segmentation

**Phase 3 (long-term): Re-tokenize with morphology-aware tokenizer**
- Train new SentencePiece model on morpheme-segmented Basque text
- Re-tokenize dataset and retrain Morpheus
- Compare MorphAcc and CSR before/after

### 6. Step 30k Baseline Results (2026-07-03) — ⚠️ EVAL v1 (pre-fix)

> **These numbers use eval v1 (buggy).** CSR is inflated (free acceptance) and `boundary_prob_mass` is renormalized over top-K. See the version note at the top of this file. MorphAcc hit-rate and Paradigm Hit@K are still valid. **Re-run with v2 before citing.**

Run on checkpoint `step_0030000.pt` (22% training, valid PPL 30.78).

#### Strategy 1: Character Savings Rate

| Prompt | Target | Chars | Typed | Saved | CSR |
|--------|--------|-------|-------|-------|-----|
| Kaixo, zer | moduz? | 6 | 0 | 6 | 100% |
| Egun on, zer | moduz? | 6 | 0 | 6 | 100% |
| Eskerrik | asko | 4 | 0 | 4 | 100% |
| Barkatu, ez dut | ulertzen | 8 | 0 | 8 | 100% |
| Zorionak eta urte | berri on! | 9 | 6 | 3 | 33% |
| Kaixo,zer | moduz? | 6 | 0 | 6 | 100% |
| Datorren astean, Eusko | Jaurlaritzak | 12 | 0 | 12 | 100% |
| Eusko Jaurlaritzak gaur | jakinarazi du | 13 | 1 | 12 | 92% |
| Gaur goizean...gure | bizitzaz | 8 | 4 | 4 | 50% |
| Etxe hau berria da eta | oso polita | 10 | 4 | 6 | 60% |

**Avg CSR: 83.6% | Macro CSR: 81.7%** ⚠️ v1 (inflated — free acceptance. v2 will be lower.)

Key insight: collocations and proper nouns are 100% (zero keystrokes needed). News phrases ~92-100%. Long context drops to 50-60%. The "Zorionak eta urte" test fails at 33% because `askotarako!` beats `berri on!` in the first token — but both are valid continuations; this is a normative judgment, not an error.

#### Strategy 2: Morpheme Boundary Accuracy

| Prompt | Boundary | MorphAcc@5 | Notes |
|--------|----------|------------|-------|
| etxe | etxe\|tik | ✗ | No suffix tokens in top-5. Hyphens, fragments of other words |
| etxeti | etxe\|tik | ✗ | Punctuation dominates: `. #1, ak #2, , #3` |
| lagun | lagun\|arekin | ✗ | Soft-hyphen #1, comma #2. Zero suffix probability |
| mendi | mendi\|ra | ✗ | Same pattern: `-` #1, `ak` #2, `lerro` #3 |
| Bihar...nire etxe | etxe\|ra | ✓ #2 | With context: `etara` #1, `raino` #2 (34.5%). Model needs disambiguation |

**MorphAcc@5: 20% (1/5)** (hit-rate still valid in v2; the `34.5%` prob-mass figure above is v1/renormalized)

Root cause: SentencePiece Unigram tokenizer fragments morphemes arbitrarily. The model never learns to associate root boundaries with case suffixes because the tokenizer cuts through morpheme boundaries. `etxe+ra → e|txe|ra` in tokenizer, but model sees `etxe` as one token and must predict a suffix that's been split into subword fragments.

#### Strategy 3: Case Paradigm Completion

6 nouns × 14 cases = 84 tests.

**Hit@1: 3.6% | Hit@3: 7.1% | Hit@5: 14.3%**

Per-case best performers:
- Absolutive (-a): 33% Hit@1 (avg rank #4) — best case, frequency-driven
- Genitive singular (-aren): 17% Hit@1 (avg rank #27)
- Ergative (-ak): 0% Hit@1 (avg rank #211)
- Causal (-arengatik): 0% Hit@1, never in top-10 (rank infinity)
- Allative (-ra): 0% Hit@1, avg rank #844 — despite being one of the most common suffixes

Key finding: the model has not learned Basque's case system. Only the absolutive (bare noun + article -a) has any signal, and that's from word frequency, not morphological knowledge. This is expected at 22% training — morphology emerges late in training for agglutinative languages.

### 7. References

1. Contreras, M. (2026). *QuechuaTok: Morphological Boundary Accuracy as a Necessary Metric for Tokenizer Evaluation in Agglutinative Low-Resource Languages*. arXiv:2606.23943.
2. Lane, W., Harrigan, A., & Arppe, A. (2022). *Interactive Word Completion for Plains Cree*. ACL 2022.
3. Trnka, K., & McCoy, K. (2008). *Evaluating Word Prediction: Framing Keystroke Savings*. ACL 2008.
4. Kosyak, S., & Tyers, F. (2022). *Predictive Text for Agglutinative and Polysynthetic Languages*. FieldMatters 2022.
5. Rust, P., et al. (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models*. ACL 2021.
