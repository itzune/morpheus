# morpheus TODO

Engineering improvements identified from the real inference comparison against
futo-basque (`COMPARISON.md`). Ordered by ROI. Each item references the evidence.

---

## Priority 1: Beam search / top-2 sampling (highest ROI, proven upside)

**Evidence:** Real inference testing showed 5 correct answers sit at **rank 2** in
top-5 ("ikasten", "zait", "Unibertsitate" were all rank 2). Raw greedy gets 7/16
top-1; beam search or top-2 sampling would almost certainly get **12/16 = 75%**.

**Why it's #1:** The correct tokens are *already in the distribution* — we just
need to look one rank deeper. No retraining, no architecture change. morpheus
already has an inference engineering layer (retokenization fallback + sticky
merge in `src/eval_utils.py`); beam search is incremental to existing infra.

**Implementation sketch:**
- Add a beam-search mode to `_nw_keyboard_candidates()` in `src/eval_utils.py`
- Keep top-1 greedy as default (fastest), expose beam-width as a parameter
- Score candidates by cumulative log-prob (product of token probs, FUTO-style)
- Reuse existing retokenization fallback for each beam

**Expected result:** 43.8% → ~75% top-1 on the 16-prompt eval set.

**Status:** Not started.

---

## Priority 2: Test Mamba-2 + autocorrect format (strategic bet)

**Evidence:** futo-basque's `<XBU>…<CHAR_X>…<XEC>` keypress autocorrect format
works (82.5% top-1) but its transformer is format-contaminated (0% next-word).
morpheus's Mamba-2 SSM has no such contamination. **Open question:** can an SSM
learn the per-keystroke attention pattern autocorrect needs? (`COMPARISON.md`
Part 4.)

**Why it matters:** If yes → morpheus becomes a *complete* keyboard LM
(next-word + autocorrect + O(1) latency), strictly better than futo-basque in
every dimension except FUTO-app deployment. If no → confirms transformer is
necessary for autocorrect, validates the two-model architecture.

**Implementation sketch:**
- Reuse futo-basque's Phase 4a data (500K synthetic typo→correct triples)
  - Format: `<XBU><CHAR_X>…<CHAR_X><XBC>correctword<XEC>`
  - Source: `futo-transformer-basque/finetune/stage_a/` (synth.json)
- Convert to morpheus's 4K vocab (re-tokenize with morpheus's SP tokenizer)
- LoRA finetune (r=16, ~5K steps) on a single L40 — fast, reversible
- Eval: does morpheus produce valid `<XBU>…<XEC>` sequences? Does next-word
  survive (no format contamination like futo-basque)?

**Key risk:** SSM compresses history into fixed-size state — may struggle with
20+ keystroke words where a transformer can attend to each `<CHAR_X>` directly.

**Status:** Not started. Data available on GPU server at
`/root/futo-transformer-basque/finetune/stage_a/`.

---

## Priority 3: Adopt FUTO's logit banning + confidence modes

**Evidence:** morpheus still has date/number artifacts ("Bihar goizean"→comma,
"Non dago"→"?"). FUTO's `transform_logits()` bans symbol/punctuation tokens at
word boundaries and redirects their probability mass to SPACE — directly
suppressing these artifacts. (`COMPARISON.md` Strategy 1, 2.)

**Why it's #3:** Small code change in `src/eval_utils.py`, measurable
improvement, and sets up infrastructure for the two-engine dictionary+LM merge
(Strategy 3).

**Implementation sketch (from FUTO C++ source review):**
- **Contextual logit banning:** Before sampling, set symbol-token logits to
  `-inf` when the model should be predicting a word continuation. Redirect
  mass to the space token (renormalize).
- **Capitalization-aware:** If user is typing `AllCaps`, boost capital-first
  tokens; ban lowercase-first.
- **Confidence-ratio modes:** Compute `top1_prob / top2_prob`. If high →
  autocorrect mode (commit to top-1). If low → uncertain/clueless mode (show
  suggestions, don't auto-commit). FUTO uses 3.0 and 1.5 thresholds.
- **Two-engine merge (later):** Dictionary exact-match boost + LM rescoring.

**Status:** Not started.

---

## Priority 4: On-device LoRA personalization (already planned)

**Evidence:** FUTO's `AdapterTrainer` does on-device LoRA (r=16, 128 iters) for
personal vocabulary. morpheus writeup §7.3.5 already plans this.

**Status:** Already in morpheus roadmap. No additional action beyond confirming
the FUTO reference implementation details in `COMPARISON.md` Strategy 8.

---

## Completed (from this comparison session)

- [x] Real inference comparison vs futo-basque (`compare_inference.py`)
- [x] Verified morpheus deployment handles tokenizer divergence correctly
      (`demo/server.py` uses `sp.encode()` → token IDs, not string prompts)
- [x] Confirmed `add_bos_token=false` metadata works in GGUF (no BOS bug)
- [x] Documented real numbers: 43.8% top-1, 75% top-5 (raw greedy, no IE)
