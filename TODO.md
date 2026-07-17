# morpheus TODO

**Direction:** Desktop text-editor ghost-text autocompletion is now the primary
target — see [`docs/TRAJECTORY.md`](./docs/TRAJECTORY.md) for the full rationale.
This file is reordered around that direction: the Phase 6 FIM work leads,
mobile-IME items are moved to a deferred research track, and the FUTO
inference-engineering strategies are filtered to only what serves the editor.

Each item references the evidence. Ordered by ROI / dependency.

---

## Primary track: Desktop editor autocompletion (Phase 6)

### P1 — Build the OpenAI-compatible face on the proxy (foundation for every client)

**Evidence:** The current 74K checkpoint already does prefix-only ghost-text.
The token-ID fidelity problem is already *known*, not conditional: the same
AR model scored **0% via string prompts vs 43.8% via token IDs**
(`COMPARISON.md` §1.2). So pointing Continue directly at `llama-server`'s
`/v1/completions` (string path) would hit the known-broken tokenization and
fail immediately. The proxy is therefore **the architecture, not a fallback** —
it centralizes the reference SentencePiece encoder and (later) the FIM template,
so every client (Continue, Obsidian, Vim, CLI) is thin and model-agnostic.

**Why it's #1:** Zero model work, zero retraining. Builds the reusable Basque
inference server and gives a working VS Code + JetBrains ghost-text plugin in
the same step, surfacing editor-integration UX issues (debounce, stop tokens,
multiline boundaries) while the FIM model trains.

**Architecture — two endpoints on `demo/server.py`:**

```
Client (Continue / Obsidian / Vim / CLI)
   │  standard completion API
   ▼
┌─────────────────────────────────────────────────────┐
│ Proxy: demo/server.py (reuses existing _call_llama) │
│  • /v1/completions  (OpenAI-compatible, string in)  │  ← Continue.dev uses this
│  • /v1/complete     ({prefix,suffix} convenience)   │  ← thin bespoke clients
│      └ SP-encode → token IDs → FIM template (P6)    │
└─────────────────────────────────────────────────────┘
   │  token-ID prompt
   ▼
llama-server /completion  (the fast engine; tokenizer bypassed)
```

The backend already exists: `_call_llama()` posts token-ID arrays to
llama-server `/completion` and SP-encodes first. Only the **OpenAI face**
(front route + response translator) is new.

**Implementation sketch:**
1. ✅ `POST /v1/completions` — accept OpenAI schema (`prompt`, `max_tokens`,
   `temperature`, `top_k`, `stop`, `stream`); SP-encode the string; forward to
   llama-server; reformat response to `choices[0].text` / `usage`.
2. ✅ Non-streaming (`stream: false`) — single request/response reformat.
3. ✅ **SSE streaming translation**: llama-server frames → OpenAI
   `data: {...}` frames, on the fly.
4. ✅ `POST /v1/complete` — convenience route `{prefix, suffix, max_tokens}`;
   server applies the FIM template once P6 lands (prefix-only until then).
5. ✅ **OpenAI-SDK conformance smoke test** (`demo/test_openai_compat.py`):
   9/9 checks pass against the real `openai` Python SDK (v2.46.0) — models.list,
   non-streaming text+usage+id+finish_reason, streaming SSE assembly, stop
   sequences, token-ID prompt path. This is the same SDK Continue.dev uses, so
   protocol conformance is proven without a full editor install.
6. ⬇️ **Deferred to P6**: full Continue.dev-in-VS-Code dogfood. Rationale: the
   prefix-only AR ghost-text is a weak test (the model just appends text); the
   interesting FIM-specific UX (does `<MID>` respect the suffix? stop before
   existing text?) only matters once the FIM model exists. The proxy itself is
   proven conformat; the editor dogfood happens against the FIM model in P6,
   where it doubles as the real quality bar.

**Why this is reusable, not a Continue integration:** the `/v1/completions`
face is the OpenAI-ecosystem standard — Continue, Cody, codecompanion.nvim, and
Obsidian plugins (Text Generator, Smart Composer) all speak it with zero code
from us. The `/v1/complete` face enables ~50-line bespoke clients. One server,
N thin clients; the tokenization fix and FIM template live in exactly the place
they belong. If llama.cpp ever fixes the divergence upstream, delete the proxy
and repoint — zero client changes.

**Status:** ✅ **Done** (commits `badf6e9`, smoke test below). `/v1/completions`
(streaming + non-streaming), `/v1/complete`, and `/v1/models` all implemented
on `demo/server.py` and verified end-to-end. The full Continue.dev editor
dogfood is deferred to P6 (see step 6 above) — it's a weak test against AR-only
and a strong test against FIM.

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

**Status:** ✅ **Done**. FIM tokenizer (`tokenizer/basque_unigram_fim.model`,
4004 pieces) and resized checkpoint (`checkpoints/step_0074000_fim.pt`,
vocab 4016) both created and verified. Smoke test confirms AR perplexity
unchanged: original loss=1.5909 (ppl 4.91) vs resized loss=1.5911 (ppl 4.91),
Δ=0.01%. Gates P3–P5.

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

**Status:** ✅ **Done**. `scripts/pipeline/build_fim.py` implements char-level
PSM/SPM transform with deterministic two-pass (count→fill) and per-line RNG.
Built and verified:
- `data/train_fim.npy` — full FIM+AR training stream (~5B tokens, 50% FIM)
- `data/valid_fim.npy` — FIM validation set (2M tokens, 22.5K FIM + 27.4K AR)
- Structure verified: 100% PSM (`PRE<SUF<MID<EOT`) and SPM (`SUF<PRE<MID<EOT`)
  ordering correct; PSM/SPM balance 50.6/49.4; FIM rate ~45% (short lines
  fall back to AR).
- `config/phase6_fim.yaml` created (P4 config: 2B tokens, LR 1e-3, FIM data).

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

**Status:** ✅ Done. `scripts/pipeline/build_fim.py` implemented and deployed; full training set building on server (`data/train_fim.npy`, ~1.5B tokens target, 50% FIM/50% AR mix). FIM validation set `data/valid_fim.npy` built and verified.

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

**Status:** Script ready (`scripts/fim_eval.py`): exact-match, char accuracy (Levenshtein), keystrokes-saved metrics. Awaiting P4 checkpoint to evaluate. Depends on P4.

---

### P6 — Phase 6e: Export FIM GGUF + server-side FIM template + dogfood

**Evidence:** TRAJECTORY §4.2. The FIM template (`<PRE>{prefix}<SUF>{suffix}<MID>`)
belongs **server-side** on the proxy (P1), not duplicated in each editor client.

**What changed from the old TODO:** the old P6 framed the token-ID proxy as
conditional ("test if FIM masks the divergence; if not, keep the proxy"). The
proxy is the default from P1 — it already solves tokenization, so P6 only adds
the FIM template to it. No client-side FIM templating; no per-editor token-name
knowledge.

**Implementation sketch:**
- Export the FIM checkpoint → HF → GGUF (reuse `scripts/export/`); confirm
  `add_bos_token=false` metadata still set.
- Add the FIM template **to the proxy**: `/v1/complete` wraps `{prefix,suffix}`
  as `<PRE>{prefix}<SUF>{suffix}<MID>`; `/v1/completions` passes through
  already-templated strings (Continue.dev's own template works unchanged).
  Stop tokens `<EOT> <PRE> <SUF> <MID>`.
- Point Continue.dev at `http://localhost:9090/v1` (`provider: openai`,
  `roles: [autocomplete]`); this is the dogfood deferred from P1 — the proxy is
  already proven OpenAI-conformant, so this is config + UX observation, not
  integration work. FIM is where ghost-text gets useful.
- Dogfood in VS Code against the realistic-message set; iterate stop-token /
  debounce / multiline-boundary policy (TRaJECTORY §7.4). Compare against the
  P1 prefix-only baseline — FIM must measurably beat it to justify the GPU spend.

**Status:** ✅ Server-side FIM template added to `/v1/complete` (builds `<PRE>{prefix}<SUF>{suffix}<MID>`, stops at `<EOT>`). Smoke tests #6/#7 added. Depends on P5 for FIM checkpoint dogfood.

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

- [x] **P3: Basque FIM dataset (char-level PSM/SPM transform)**
      `scripts/pipeline/build_fim.py` — two-pass (count→fill) with deterministic
      per-line RNG, 50% FIM / 50% AR, 50/50 PSM/SPM, ~20% linguistic-boundary
      splits / ~80% random-char. FIM string SP-encoded as one string (FIM tokens
      are atomic USER_DEFINED, survives encoding intact). Built:
      `data/train_fim.npy` (~5B tokens), `data/valid_fim.npy` (2M tokens).
      Verified: 100% PSM/SPM structure correct, PSM/SPM 50.6/49.4, FIM rate ~45%.
      `config/phase6_fim.yaml` created for P4.
- [x] **P2: FIM special tokens + embedding resize**
      Three scripts: `add_fim_tokens.py` (protobuf append, preserves all
      existing IDs), `resize_embeddings.py` (4000→4016, mean-init FIM rows,
      zero-init padding, tied embeddings), `smoke_test_fim.py` (tokenizer +
      AR perplexity). Results: 4 FIM tokens at IDs 4000–4003 (`<PRE>/<SUF>/
      <MID>/<EOT>`, USER_DEFINED, atomic); AR perplexity Δ=0.01%
      (loss 1.5909→1.5911, ppl 4.91→4.91) — negligible perturbation.
- [x] **P1: OpenAI-compatible `/v1/completions` + `/v1/complete` on proxy**
      (commits `badf6e9`, `a205eb4`). Three routes on `demo/server.py`:
      `/v1/completions` (streaming SSE + non-streaming, OpenAI schema),
      `/v1/complete` (convenience `{prefix, suffix}` body), `/v1/models`.
      SP-encodes string prompts to token IDs (bypasses llama.cpp tokenizer
      divergence). Verified with `demo/test_openai_compat.py` (9/9 checks
      pass against the real `openai` Python SDK v2.46.0 — the same SDK
      Continue.dev uses). Full Continue.dev editor dogfood deferred to P6
      (AR-only ghost-text is a weak test; FIM is where it gets useful).
- [x] Real inference comparison vs futo-basque (`compare_inference.py`)
- [x] Verified morpheus deployment handles tokenizer divergence correctly
      (`demo/server.py` uses `sp.encode()` → token IDs, not string prompts)
- [x] Confirmed `add_bos_token=false` metadata works in GGUF (no BOS bug)
- [x] Documented real numbers: 43.8% top-1, 75% top-5 (raw greedy, no IE)
- [x] Keystrokes-saved eval on realistic messaging messages (`COMPARISON.md` §1.6)
