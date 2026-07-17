# morpheus TODO

**Direction:** Desktop text-editor ghost-text autocompletion is now the primary
target — see [`docs/TRAJECTORY.md`](./docs/TRAJECTORY.md) for the full rationale.
This file is reordered around that direction: the Phase 6 FIM work leads,
mobile-IME items are moved to a deferred research track, and the FUTO
inference-engineering strategies are filtered to only what serves the editor.

Each item references the evidence. Ordered by ROI / dependency.

---

## Primary track: Desktop editor autocompletion (Phase 6)

### P1 — Continue.dev integration against the current AR model (fast path, today)

**Evidence:** The current 74K checkpoint already does prefix-only ghost-text.
TRaJECTORY §5 — Phase 6e′. Continue.dev is FIM-native and model-agnostic; it
works against any OpenAI-compatible `/v1/completions` endpoint, including a
prefix-only AR model (suffix ignored until FIM lands).

**Why it's #1:** Zero model work, zero retraining. Gets a working VS Code +
JetBrains ghost-text plugin immediately, surfacing editor-integration UX issues
(debounce, stop tokens, multiline boundaries) while the FIM model trains. The
fast-feedback path that runs in parallel with everything below.

**Implementation sketch:**
- Run morpheus behind `llama-server` (already done by `demo/server.py`), which
  exposes `/v1/completions`.
- Register a Continue.dev provider: `provider: openai`, endpoint
  `http://localhost:<port>/v1`, `roles: [autocomplete]`.
- Until FIM tokens exist, use a no-op template (plain prefix completion) with
  stop tokens `</s>`, `\n\n` (paragraph boundary).
- Tune `debounceDelay`, `prefixPercentage`, `maxSuffixPercentage`,
  `modelTimeout` per [`docs/TRAJECTORY.md`](./docs/TRAJECTORY.md) §4.2.

**Status:** Not started. No blockers.

---

### P2 — Phase 6a: FIM special tokens + embedding resize

**Evidence:** TRAJECTORY §2.3. Current 4K vocab has only `<unk>(0)`, `<s>(1)`,
`</s>(2)` — no FIM tokens. Bavarian et al. (2022) require `<PRE>/<SUF>/<MID>/<EOT>`.

**Implementation sketch:**
- Add 4 special tokens to the SentencePiece model: `<PRE>`, `<SUF>`, `<MID>`,
  `<EOT>` (Code Llama naming — simple to template).
- Vocab 4000 → 4004 → pad to `pad_vocab_size_multiple: 16` → 4016.
- Resize the embedding table; initialize new rows (mean init or small random).
- Resume-from-74K smoke test: confirm AR perplexity is unchanged by the resize
  (embeddings are ~3M of 91M params; new rows train fast).

**Status:** Not started. Gates P3–P5.

---

### P3 — Phase 6b: Basque FIM dataset (char-level PSM/SPM transform)

**Evidence:** TRAJECTORY §3. FIM data is a *transform* of existing text, not new
collection. Char-level splitting is robust against subword boundaries — critical
for the 4K UNIGRAM vocab where Basque agglutination splits irregularly
([`docs/tokenizer-fieldwork.md`](./docs/tokenizer-fieldwork.md)). Joint PSM/SPM
(50/50) gives positive transfer; 50% FIM / 50% AR mix preserves AR capability
("FIM-for-free").

**Implementation sketch:**
- `scripts/pipeline/` → add `build_fim.py`: char-level split each Latxa v2
  shard, emit PSM or SPM (50/50) wrapped with the new special tokens.
- Mix 50% FIM shards + 50% plain AR shards into the Phase 6 training stream.
- Bias ~20% of splits toward linguistic boundaries (sentence end, clause
  boundary, word boundary) to mimic real cursor positions; keep ~80% random-char
  for generalization.
- Pretokenize to `.npy` as the existing pipeline does.

**Status:** Not started. Depends on P2.

---

### P4 — Phase 6c: Continued pretraining (50% FIM / 50% AR, ~1–2B tokens)

**Evidence:** TRAJECTORY §2.2. Continued pretraining (not a finetune) is the
evidence-backed path — finetuning FIM onto an AR model is expensive and
loss-prone (high LR, catastrophic forgetting, ~50B tokens cited for 162M–13B
models). At 91M scale, 1–2B tokens is proportionate.

**Implementation sketch:**
- Resume from the 74K checkpoint (do *not* retrain from scratch).
- ~1–2B additional tokens, FIM rate 0.5 (per §3.2 sweet spot 50–90%; we use 50%
  to be conservative on AR preservation).
- Same optimizer/schedule as `config/small.yaml`; lower peak LR (continued
  pretrain, not warm start) — try `1.0e-3` → `min_lr 1.0e-5`.
- Monitor: AR perplexity on the existing valid set must not regress (the
  FIM-for-free property); add a FIM-span perplexity once P5 lands.

**Status:** Not started. Depends on P3. Needs GPU time.

---

### P5 — Phase 6d: Basque FIM eval (random-span infilling)

**Evidence:** TRAJECTORY §3.4 + §7.3. PPL does not measure FIM quality well
(Bavarian et al. emphasize sampling-based infilling benchmarks). Existing
CSR/MorphAcc ([`docs/eval-strategies.md`](./docs/eval-strategies.md)) are AR-only
and don't transfer to cursor-in-the-middle.

**Implementation sketch:**
- Held-out Basque sentences; mask a random span; measure whether `<MID>`
  generation reconstructs (exact-match) or plausibly continues (LLM-judge, per
  [`docs/llm-judge-eval-research.md`](./docs/llm-judge-eval-research.md)) the
  original.
- Report a **keystrokes-saved-in-the-middle** analogue: chars saved by accepting
  the FIM completion vs typing, for cursor positions mid-document. Mirrors the
  realistic-message methodology in [`COMPARISON.md`](./COMPARISON.md) §1.6 but at
  arbitrary cursor positions, not just word boundaries.
- Open question (TRaJECTORY §7.3): exact-match vs LLM-judge vs keystrokes —
  resolve by dogfooding in P6.

**Status:** Not started. Depends on P4.

---

### P6 — Phase 6e: Export FIM GGUF + Continue.dev FIM template + dogfood

**Evidence:** TRAJECTORY §4.2. Continue.dev ships per-model FIM templates
(`core/autocomplete/templating/AutocompleteTemplate.ts`); adding a morpheus
entry is the integration point.

**Implementation sketch:**
- Export the FIM checkpoint → HF → GGUF (reuse `scripts/export/`); confirm
  `add_bos_token=false` metadata still set.
- Add a morpheus FIM template to Continue.dev config:
  `<PRE>{prefix}<SUF>{suffix}<MID>`, stop tokens `<EOT> <PRE> <SUF> <MID>`.
- **Token-ID fidelity check (TRaJECTORY §4.3):** llama.cpp's SentencePiece
  rebuild diverges from reference on this 4K vocab. Test whether FIM (char-level
  splits, atomic FIM tokens) masks the divergence. If not, keep the thin
  token-ID proxy in `demo/server.py` (accept Continue's FIM string → SP-tokenize
  → forward token IDs to `/completion`).
- Dogfood in VS Code against the realistic-message set; iterate stop-token /
  debounce / multiline-boundary policy (TRaJECTORY §7.4).

**Status:** Not started. Depends on P5.

---

## Inference quality (no retraining)

### P7 — Decoding policy for ghost-text (was "beam search")

**Evidence:** Real inference testing showed 5 correct answers sit at **rank 2**
in top-5. Raw greedy gets 7/16 top-1; looking one rank deeper recovers most of
the gap. (`COMPARISON.md` §1.2.)

**What changed from the old TODO:** The old item proposed adding beam search to
`_nw_keyboard_candidates()` in `src/eval_utils.py`. That is the GPU *eval* path;
the editor deploys through `llama-server` `/v1/completions`, where sampling is a
config pass-through. So the lever is **sampling parameters + stop tokens**, not
custom beam code.

**Implementation sketch:**
- In Continue.dev: set a low-but-nonzero `temperature` (e.g. 0.2) and
  `top_k: 3–5` for ghost-text to recover rank-2/rank-3 tokens without
  wildness. (Pure greedy is too brittle; the rank-2 evidence justifies a small
  top-k.)
- Stop tokens: `</s>`, `<EOT>`, and paragraph boundary (`\n\n`) for prose.
- Server-side: confirm `llama-server` `/v1/completions` honors `top_k` /
  `temperature` / `stop` passthrough (it does); consider `min_p` sampling if
  available for better long-tail behavior.
- Multi-token ghost-text: Continue.dev manages multi-line logic; tune
  `maxPromptTokens` and suffix percentage.

**Expected result:** Recover much of the rank-2 upside (old TODO projected
43.8% → ~75% on the 16-prompt eval) via sampling config alone, no retraining.
Re-measure on the FIM eval (P5) once it exists.

**Status:** Not started. Partly folds into P1/P6 (Continue config).

---

## Downstream

### P8 — Desktop personalization (LoRA from the user's documents)

**Evidence:** Was "on-device LoRA" (old P4). FUTO's `AdapterTrainer` (r=16, 128
iters, n_ctx=64) is the reference ([`COMPARISON.md`](./COMPARISON.md) Strategy 8);
morpheus writeup §7.3.5 plans it; full feasibility in
[`docs/On-Device Model Adaptation Feasibility.md`](./docs/On-Device%20Model%20Adaptation%20Feasibility.md).

**What changed:** "On-device" = the desktop itself, and the training data is the
user's *documents* (richer than mobile typing logs). This is a Phase 7+ item,
**downstream of FIM** — personalize the FIM-capable model, not the AR-only one.

**Implementation sketch:**
- Background LoRA finetune (r=16) on recently-edited documents; merge + requantize
  to GGUF (Mamba-2 LoRA on in_proj/out_proj; no architectural barrier).
- Trigger: idle time + N new tokens typed.
- Privacy: fully local; no federated learning needed (contrast Gboard's
  federated LSTM — see On-Device doc).

**Status:** Not started. Depends on P4 (FIM model).

---

### P9 — Verb-agreement SFT (orthogonal model-quality fix)

**Evidence:** Three-way dative agreement (*nor-nori-nork*) failures persist
regardless of target platform. [`docs/verb-agreement-finetune-research.md`](./docs/verb-agreement-finetune-research.md)
— Strategy B (gap-fill SFT) is the primary fix.

**Why it's here, not higher:** This is a Basque-correctness issue, independent of
desktop vs mobile. It **composes** with FIM (FIM teaches *where* to fill; this
SFT teaches *what* Basque is correct). Track it in its own doc; listed here only
so it isn't lost from the roadmap.

**Status:** Not started. Independent of Phase 6; can run in parallel after P4.

---

## Deferred — Secondary research track (mobile IME)

### D1 — Test Mamba-2 + FUTO autocorrect format (`<XBU>…<CHAR_X>…<XEC>`)

**Evidence:** Was old Priority 2. `COMPARISON.md` Part 4 open question: can an
SSM learn the per-keystroke attention pattern autocorrect needs?

**Why deferred:** This is a **mobile-IME** capability. The desktop editor path
does not need autocorrect (it needs ghost-text/FIM). Per TRAJECTORY §1.2, the
mobile IME is a secondary track that depends on this experiment succeeding. It
is **not on the desktop critical path** and should not block P1–P6.

**If pursued later:** reuse futo-basque's Phase 4a data (500K synthetic
typo→correct triples) at `/root/futo-transformer-basque/finetune/stage_a/`,
re-tokenize to morpheus's 4K vocab, LoRA finetune (r=16, ~5K steps). Key risk:
SSM state compression may struggle with 20+ keystroke words.

**Status:** Not started. Gated by a decision to pursue mobile, not by desktop work.

---

## Explicitly not adopting (from the FUTO review)

Filtered out of the old TODO because they serve the mobile-keyboard UX, not the
desktop editor. Recorded here so the reasoning is explicit.

- **Logit banning of punctuation/symbol tokens (symbol→SPACE redirection).**
  FUTO bans punctuation at word boundaries to force mobile word-chip boundaries.
  For desktop ghost-text this is *harmful*: completions legitimately need
  sentence-ending periods, commas, newlines, and quotes. The punctuation
  artifacts it fixed ("Bihar goizean"→comma) are a non-issue for ghost text
  (visible as ghost text, user just doesn't accept) and are handled by
  Continue.dev's prefiltering + the confidence logic below.
- **Capitalization-aware logit banning.** Mobile-IME-specific (FirstCapital /
  AllCaps typing modes). Desktop editors handle capitalization via context;
  banning lowercase-leading tokens would break legitimate prose completions.
- **Confidence-ratio "autocorrect mode" (auto-commit when top1 >> top2).**
  Mobile-IME UX (auto-replace the typed word). Desktop ghost-text uses
  explicit accept (Tab); auto-commit is undesirable. The *other* half of the
  idea — suppress low-confidence suggestions ("clueless mode") — **is** useful
  and is already implemented by Continue.dev's confidence thresholding, so we
  don't reimplement it.
- **Chat-completions API for autocomplete transport.** Wrong shape for FIM
  (TRaJECTORY §4.1). FIM needs raw prompt + streaming + stop tokens, not a
  message-array/role abstraction.
- **Bespoke editor plugin.** Continue.dev already covers FIM templating,
  prefiltering, and VS Code + JetBrains support. Reusing it is strictly better.

---

## Completed

- [x] Real inference comparison vs futo-basque (`compare_inference.py`)
- [x] Verified morpheus deployment handles tokenizer divergence correctly
      (`demo/server.py` uses `sp.encode()` → token IDs, not string prompts)
- [x] Confirmed `add_bos_token=false` metadata works in GGUF (no BOS bug)
- [x] Documented real numbers: 43.8% top-1, 75% top-5 (raw greedy, no IE)
- [x] Keystrokes-saved eval on realistic messaging messages (`COMPARISON.md` §1.6)
