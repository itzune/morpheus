# Evaluation Reform Proposal

> **Status:** Proposal — synthesized from literature review (ChaI-TeA, Gmail Smart
> Compose, Trnka & McCoy, Kosyak & Tyers, WSTypist) and analysis of the current
> `eval.py` / `demo/csr_eval.py` / `demo/real_corpus_eval.py` implementations.
> The Basque-expert user must approve any metric that involves quality judgment.

## The Diagnosis (why we can't trust current eval)

The "eval crisis" is not a single bug. It is a stack of compounding weaknesses.
The symptoms:

| Path | 32K | 54K | Direction |
|------|-----|-----|-----------|
| GPU `eval.py` (f16) CSR macro | 0.2917 | 0.2682 | 54K **worse** (−2.35) |
| Demo `csr_eval.py` (Q4) CSR macro | 0.2780 | 0.2884 | 54K **better** (+1.04) |
| Human (expert) | — | — | 54K "much better" |

Two automated metrics disagree with each other **and** with the human. This means
none of them can be trusted to rank checkpoints. Root causes, in priority order:

### 1. The sample is far too small (the dominant problem)
`eval/targets.json` has **30 CSR sentences** (15 articles × 2 context depths).
At n=30, each sentence is worth ~3.3 CSR percentage points. The 32K→54K demo
gap is +1.04 — **one third of a single sentence**. This is pure noise.

For comparison: ChaI-TeA (the closest published analog) evaluates on
**26,000+** test prefixes. We are ~1000× underpowered. No statistical conclusion
is possible at n=30, and bootstrap CIs would span the entire 26–30% band.

### 2. Exact-match gold cannot credit valid Basque alternatives
Basque is agglutinative: a given prompt has many grammatically valid continuations.
CSR credits a prediction only if it is a **character-level prefix of the single
gold string** the author happened to write. If the model proposes an equally valid
but different word/suffix, CSR gives **zero**. This is a *lower bound* on quality.

ChaI-TeA hit exactly this wall: they tried LLM-as-judge (Claude 3) to credit
semantically-valid alternatives and found it "very challenging," deferring it as
future work. We have the same problem, compounded by agglutination (more valid
surface forms per slot than English).

Consequence: CSR can detect *catastrophic regression* (model emits garbage → CSR
crashes), but it **cannot reliably rank two competent models** that differ in
*which* valid continuation they prefer. This is precisely the 32K-vs-54K regime.

### 3. The two CSR implementations disagree by 3.4 points on the same data
`src/eval_utils.py::evaluate_csr` (GPU, f16, reference `sentencepiece`) and
`demo/csr_eval.py` (llama-server, Q4, llama.cpp SP) differ by ~3.4 points in
opposite directions across the two checkpoints. Confounded differences:

- **Quantization**: GPU=f16, demo=Q4_K_M. Expected ~1–2 pt CSR loss from Q4.
- **Output decoding**: the model predicts token *IDs* identically, but
  llama.cpp decodes those IDs with its **built-in SentencePiece**, which is
  known to diverge from the reference `sentencepiece` library on this 4K vocab
  (documented for long words). The `.lstrip()` + prefix-match logic itself is
  identical in both files.
- **Prompt path**: demo now sends token-ID prompts (BOS fix applied), so prompt
  tokenization is aligned — but output decoding is not.

Until these confounds are isolated, **no cross-path comparison is meaningful**.

### 4. The test set is saturated
`targets.json` has been evaluated **108 times** during training (every 500 steps
from 0→54K). It is a fixed, monitored target. Even though the *training-data*
leakage was fixed (68,755 validation lines excluded from pretokenization), the
*eval set itself* has been observed so often that we are overfitting our
attention to it, not the model's behavior.

### 5. CSR measures greedy single-token match, not fluency
CSR simulates top-1 greedy acceptance. It cannot distinguish "fluent Basque that
differs from gold" from "garbage." It is a keystroke-economy metric, not a
language-quality metric. PPL tracks the full distribution; CSR tracks one greedy
path. They measure different things and need not agree.

---

## What the Literature Says

### ChaI-TeA (2024, arXiv:2412.18377) — closest analog
- **`saved@k`**: `(len(accepted_text) − #acceptances) / (len(full_turn) − 1)`.
  Penalizes the *number* of acceptances (each Tab costs a keystroke). Our CSR is
  essentially `saved@1`. ChaI-TeA generalizes: how much is saved if the user can
  accept up to *k* suggestions? This separates model quality from greedy-decoding
  quality and reveals headroom.
- **Scale**: 26K+ prefixes vs our 30. Decisive.
- **Exact-match limitation**: explicitly acknowledged; LLM-as-judge attempted and
  abandoned as "very challenging." This validates our experience.
- **Mid-word penalty**: mid-word suggestions degrade acceptance ~60% in English.
  For Basque, **mid-word (suffix) completion is the core use case** — English
  findings do not transfer.
- **Perplexity insufficient for ranking**: large gap between perfect reranking
  (`k_max`) and small k. PPL alone won't optimize autocomplete.

### Trnka & McCoy (2008) — our CSR basis
- Keystroke Savings Rate framework; acceptance costs 1 keystroke (we implement
  this correctly). Recommends reporting KSR alongside top-N accuracy and list
  position — we report only CSR.

### Kosyak & Tyers (2022) — agglutinative predictive text
- FST-based approach for agglutinative/polysynthetic languages; eval = KSR with a
  user study. Confirms the multiple-valid-form problem is central for
  agglutinative languages.

### Gmail Smart Compose (Chen, 2019) — real-world gold standard
- Precision/recall of *accepted* suggestions + match rate, measured on real user
  data. The deployment metric, not a synthetic one. Relevant as the target
  product metric if we ever do a real user study.

### WSTypist (2026, arXiv:2602.06489)
- Simulation-based mobile typing with word suggestions. Confirms that
  simulation-based metrics (like our CSR) are accepted practice when user studies
  are infeasible — *provided* the simulation is at scale and the acceptance model
  is realistic.

---

## Reform Plan (tiered by tractability)

### Tier 0 — Immediate diagnostics (today, no new metrics)

**D0. Perplexity on fresh real-corpus text, both checkpoints.**
We already know training `valid_loss` fell 3.1→2.05 (PPL ~22→7.2) monotonically —
the model *is* learning by PPL. The open question is whether 54K < 32K on **genuinely
unseen** text. Compute mean cross-entropy / PPL of both checkpoints over the
real-corpus sentences (Wikipedia + Berria). PPL is a smooth, low-variance,
full-distribution metric with no exact-match artifact. If 54K < 32K in PPL *and*
the human says 54K is better, that is strong, convergent evidence that **CSR is
saturated, not the model**.

> ⚠️ Verify first that the real-corpus articles are not in the training data.
> The training corpus includes Wikipedia + Berria sources. If the same articles
> appear in `eval/real_corpus/`, PPL would reflect memorization, not
> generalization. (Relative 32K-vs-54K comparison is still informative — both see
> the same text — but absolute PPL would be optimistic.) A quick `grep`/hash
> overlap check between `eval/real_corpus/*.txt` and the training sources settles it.

**D1. Reconcile the GPU-vs-demo CSR divergence.**
Run the GPU `eval.py` CSR on **both** checkpoints over the **same** real-corpus
sentences (not just `targets.json`). Then run `demo/csr_eval.py` over the same
sentences. If the gap persists at scale (>200 sentences), it is a real
implementation/quantization difference, not noise. The likely culprit is llama.cpp's
SP decoding of output tokens — test by having the demo return raw token IDs and
decoding with the reference `sentencepiece` library.

### Tier 1 — Fix the eval we have (tractable, this week)

**R1. Scale the CSR test set to 200–500 sentences.**
The infrastructure exists: `demo/extract_real_prompts.py` extracts clean sentences;
`demo/csr_eval.py::build_pairs` cuts them into (prompt, target) pairs. Increase
`count` from 30 to 200+. This alone shrinks the noise floor from ±3 pt to ~±1 pt
and makes 32K-vs-54K actually decidable. Ensure the set is **held out from
training** (see D0 leakage check) and **frozen** (committed to git) so future
runs are comparable.

**R2. Report bootstrap 95% confidence intervals.**
Resample per-sentence CSRs 1000×. With n=30 the CI spans ~26–32% (explains
everything). With n=200 the CI is tight enough to rank. Trivial to add to
`csr_eval.py` and `eval.py` summaries. **No number should be reported without a CI
hereafter.**

**R3. Freeze a held-out eval set and stop re-running `targets.json`.**
`targets.json` (30 sentences, 108 eval passes) is retired as a ranking
instrument. Keep it only as a legacy regression smoke-test. The new canonical set
is the scaled real corpus (R1), committed and versioned.

### Tier 2 — Credit valid alternatives (the agglutinative core)

**R4. Multiple-reference CSR (expert-provided).**
For a subset of prompts, the user provides 2–4 valid alternative continuations
(not just the gold). CSR credits a match against *any* of them. This directly
fixes the "valid Basque but ≠ gold" penalty. Expensive in expert time → do for a
focused ~50-prompt set, not all.

**R5. Morphological-validity oracle via Apertium/euMor.**
Instead of exact gold match, accept a prediction if it is a **valid Basque word
form** (passes a morphological analyzer). This credits valid surface-form
alternatives automatically, without per-prompt expert alternatives. Reuses the
already-planned Phase-E Apertium integration. Caveat: the user has flagged that
surface-preserving morphology is required; the oracle must respect that.

**R6. LLM-as-judge (judgment call — user decides).**
Prompt a strong LLM in Basque: "is this a grammatically valid continuation?"
ChaI-TeA found this hard for free-form chat; for narrow autocomplete continuation
it may be more tractable. **The user is the Basque authority** — LLM judgments
would only be used to *triage* (flag clear garbage), never as the final quality
call, unless the user explicitly opts in.

**R7. N-gram overlap / partial-credit match.**
Replace binary prefix-match with n-gram overlap (BLEU-lite) between suggestion and
gold. Gives partial credit for "right root, wrong suffix" — common in agglutinative
languages. Cheap, no expert needed. Good as a *secondary* signal; should not
replace CSR (which has a clean keystroke interpretation).

### Tier 3 — Better metrics beyond CSR

**R8. Add `saved@k` (ChaI-Tea).**
Generalize CSR to allow up to *k* acceptances. Reveals how much headroom exists
between greedy (k=1) and a better ranking. Cheap extension to `evaluate_csr`.

**R9. Report CSR alongside top-N accuracy and list position (Trnka & McCoy).**
We currently report only top-1 CSR. Add: at each position, what rank is the
gold-matching prediction? How often is it in top-3 / top-5? This is the metric
Trnka & McCoy recommend pairing with KSR/CSR.

### Tier 4 — Human evaluation (gold standard)

**R10. Blind side-by-side A/B by the expert.**
20 fresh prompts, both checkpoints produce completions, labels hidden, user picks
better/worse/tie. This is the *only* metric that captures "valid Basque
alternative continuation" with authority. The user already does informal
qualitative testing; formalizing it with blinded labels removes bias. ~30 min of
expert time, definitive for the 32K-vs-54K question.

---

## Recommended immediate sequence

1. **D0** — PPL on real corpus, both checkpoints + leakage check. (~30 min GPU)
2. **D1** — Reconcile GPU-vs-demo CSR at scale. (diagnostic)
3. **R1 + R2** — Scale CSR to 200 sentences + bootstrap CIs. Re-rank 32K vs 54K.
4. **R10** — Expert blind A/B on 20 fresh prompts (definitive tiebreaker).

If D0 + R1 + R10 all agree 54K > 32K (PPL lower, CSR higher with CI, human
prefers), the eval crisis is resolved: **CSR-on-30 was saturated/noisy, the model
genuinely improved, and 54K is the better checkpoint.** That becomes the
publishable finding, with the eval-saturation caveat as a methodological note.

If they *disagree*, we have a real model problem to investigate.

---

## Open questions for the user (Basque expert)

- **R4/R6**: Are you willing to (a) provide alternative valid completions for a
  ~50-prompt subset, and/or (b) accept LLM-as-judge as a *triage* filter only?
- **R10**: Can you commit ~30 min for a blinded 20-prompt A/B?
- **D0 leakage check**: Do you know which Wikipedia/Berria articles are in
  `eval/real_corpus/` vs the training sources? (I can run an automated overlap
  check regardless.)
