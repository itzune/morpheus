# FUTO-Basque vs Morpheus: Comparative Analysis & FUTO Engineering Review

> **Purpose:** (1) Compare futo-basque and morpheus for next-word/token prediction.
> (2) Deep-review FUTO Keyboard's transformer engineering strategies and identify
> what could be applied to morpheus.
>
> Date: 2026-07-15. Updated 2026-07-17 with **re-run inference testing** on the
> fixed futo-basque v2.0.0 model (pretrain bug fixed + multitask finetune).
> Both GGUF models loaded via llama-cpp-python, same 16-prompt eval set.
> Based on FUTO source (`JNI_LanguageModel.cpp`, `LanguageModel.kt`,
> `ModelMeta.cpp`), the morpheus writeup, and real inference results.

---

## Part 1 — Head-to-Head: Which is Better at Next-Word/Token Prediction?

### 1.1 The two models at a glance

| Dimension | futo-basque | morpheus |
|-----------|-------------|----------|
| **Architecture** | Llama transformer (MHA, no GQA) | Mamba-2 SSM |
| **Parameters** | 25M | 91M (3.6×) |
| **Tokenizer** | SP UNIGRAM, vocab=4096 | SP UNIGRAM, vocab=4000 |
| **Context** | 2048 max (trained @ seq_len 1024) | 1024 (SSM state = fixed ~16KB/layer) |
| **Training data** | 3B tokens (Latxa v2) + finetune | ~10B tokens (~2.16 epochs of 4.62B corpus) |
| **Tokens:params ratio** | ~120:1 | ~110:1 |
| **Deployment format** | GGUF v2, Q6_K, 49MB | GGUF Q4_K_M 55MB / Q5_K_M 66MB |
| **Deployment target** | FUTO Keyboard app (Android) | Custom demo server (Docker/WASM) |
| **Primary capability** | **Autocorrect** (keypress format) + next-word | **Next-word / ghost-text** continuation |
| **Inference layer** | FUTO app's C++ (banned tokens, beam, confidence) | Custom (retokenization fallback, sticky merge) |

### 1.2 Real inference testing (not docs)

We ran both GGUF models through the **same inference engine** (llama-cpp-python)
on the **same 16-prompt eval set**, with SentencePiece tokenization fed as token
IDs (bypassing llama.cpp's string tokenizer, which diverges from SP on morpheus's
4K vocab). Full script: `compare_inference.py`,
raw results: `notes/comparison_results.json`.

**Critical tokenizer-divergence finding:** llama.cpp's rebuilt tokenizer does NOT
match SentencePiece for morpheus's 4K UNIGRAM vocab. Example: SP tokenizes
`▁zer` as a single token (515), but llama.cpp splits it into `▁`+`z`+`er`
(tokens 261, 277, 372). The model never saw that 3-token sequence in training,
so string-prompt inference produces garbage (0%). Passing SP token IDs directly
fixes it → 43.8%. This is **exactly the issue morpheus's README and demo server
already document and handle** (`demo/server.py` uses `sp.encode(prompt)` →
token IDs, never string prompts). So this is NOT a deployment bug — only our
initial test script (using string prompts) was affected. Confirmed:
`add_bos_token=false` in the GGUF metadata works correctly; BOS is NOT prepended.

#### Head-to-head results (real inference, same eval set, same engine)

| Metric | morpheus | futo-basque | Winner |
|--------|----------|-------------|--------|
| **Next-word top-1 (greedy)** | 43.8% (7/16) | **56.2%** (9/16) | **futo-basque** |
| **Next-word top-5** | 75.0% (12/16) | 75.0% (12/16) | tie |
| **Latency (ms/token)** | 2.5 | 0.8 | **futo-basque** (3× faster) |

#### What both models get right (shared 7/16 top-1 hits)

Both models correctly predict the same 7 prompts:

| Prompt | Both predict | Gold |
|--------|-------------|------|
| `Egun on, zer` | **moduz** ✓ | moduz, berri, da |
| `Zer` | **da** ✓ | da, esan, egin |
| `Gaur ezin` | **da** ✓ | dut, naiz, da |
| `Barkatu, ez` | **da/dut** ✓ | dakit, nahi, dut |
| `Gaur eguraldi` | **ona/txarra** ✓ | ona, txarra, politikoa |
| `Atzo etxera` | **joan** ✓ | joan, etorri, heldu |
| `Lagun batek` | **esan** ✓ | esan, egin, idatzi |

#### Where futo-basque pulls ahead (2 additional top-1 hits)

futo-basque correctly predicts 2 prompts where morpheus fails:

| Prompt | morpheus | futo-basque | Gold |
|--------|----------|-------------|------|
| `Ni euskara` | ri ✗ | **ikasten** ✓ | ikasten |
| `Bai, gustatu` | ko ✗ | **zait** ✓ | zait, zaizu |

These are high-value conversational patterns ("I Basque → am learning",
"Yes, liked → it"). futo-basque's multitask finetune on conversational data
likely helps here.

#### Where both models fail (7/16 shared misses)

| Prompt | morpheus | futo-basque | Gold |
|--------|----------|-------------|------|
| `Ez dut` | uste | uste | ahaztu, dakit, maite |
| `Zein da zure` | iritzia | ametsa | izena, adina, etxea |
| `Bihar goizean` | , | hasiko | etorriko, joango, izango |
| `Eskerrik asko` | , | zure | guztiaz, laguntzagatik |
| `Non dago` | ? | zure | etxea, trena, garagardoa |
| `Nola` | nahi | egin | zaude, dago, da |
| `Euskal Herriko` | Unibertsitateko | Unibertsitateko | Unibertsitatea |

Both models produce "uste" for `Ez dut` (a plausible but non-gold continuation)
and both truncate `Euskal Herriko → Unibertsitateko` (close but genitive form).

#### What changed since v1.0.0 (root cause postmortem)

The previous comparison (Jul 15) showed futo-basque at **0% top-1** — every
prediction was wrapped in `<XBU>` autocorrect format tokens. This was diagnosed
as **two compounding bugs**:

1. **Pretrain double causal-shift** (root cause): the pretrain script used
   `input_ids=ids[:-1]`, `labels=ids[1:]`, but HF Trainer already shifts
   internally — so the model learned skip-1 prediction `P(token[i+2]|token[i])`
   instead of next-token `P(token[i+1]|token[i])`. Diagnostic confirmed:
   skip-1 loss 3.8 < next-token loss 7.6 (inverted). Fixed in `a377081`.

2. **Format contamination from sequential finetune**: the old 4a→4b→4c pipeline
   mixed `<XBU>` control tokens inline with plain text, causing 100% format
   bleed. Fixed by replacing with unified multitask finetune (4m): 60% plain
   text + 40% isolated triples, strictly segregated at sequence level.

After both fixes, futo-basque v2.0.0 scores **56.2% top-1** — a complete reversal
from 0%.

### 1.3 Documented stats (for reference)

The morpheus writeup documents higher numbers with its full inference engineering
layer (retokenization fallback + sticky merge):

| Metric | morpheus (documented) | morpheus (real, raw greedy) |
|--------|----------------------|-----------------------------|
| Next-word top-1 | 50.3% (75/149) | 43.8% (7/16) |
| Next-word top-3 | 85.2% (127/149) | — |
| NW-CSR (with IE) | 0.40 | — |
| Word accuracy (no IE) | 60.4% (90/149) | — |
| Latency | 97ms (2017 laptop) | 2.5ms (L40 server) |

The real 43.8% (raw greedy, no IE, different eval set) is consistent with the
documented 50.3% (with IE, CSR sentences). The gap is explained by: (1) no
inference engineering, (2) different eval set, (3) fewer prompts (16 vs 149).

futo-basque v2.0.0's real inference score (56.2% top-1) is from the same eval
set and engine as morpheus's 43.8% — a fair comparison.

### 1.4 Autocorrect

| Metric | futo-basque v2.0.0 | morpheus |
|--------|-------------|----------|
| **Autocorrect top-1** (standalone GGUF eval) | 0% (0/29) | N/A (doesn't do autocorrect) |

**Note:** The previous comparison reported 82.5% autocorrect for futo-basque
v1.0.0 — that number was from the model's own eval script which extracted words
from inside `<XBU>` format tokens, and the model was trained on the broken
skip-1 objective. The v2.0.0 model, evaluated honestly in FUTO control-token
format (`keyboard.py`), scores 0% standalone — the 40% triple ratio in the
multitask finetune wasn't sufficient to master the autocorrect format.

However, in the real FUTO app, autocorrect works via a **hybrid architecture**:
the classical dictionary engine (AOSP LatinIME) proposes real-word candidates,
and the transformer re-ranks them. The transformer doesn't need to generate
correct words from scratch — it just needs to score them. A Basque dictionary
wordlist (`eu_wordlist.combined.gz`) is still needed for the dictionary half.

**Morpheus has no autocorrect capability at all.** It does word *completion*
(prefix → full word), not word *correction* (typo → correct word).

### 1.5 Verdict

| Task | Winner | Margin |
|------|--------|--------|
| Next-word top-1 (real inference) | **futo-basque** | 56.2% vs 43.8% (+12.4pp) |
| Next-word top-5 (real inference) | **tie** | 75.0% vs 75.0% |
| Autocorrect (standalone) | **neither** | futo 0% (needs dictionary engine), morpheus N/A |
| Inference latency (server) | **futo-basque** | 0.8ms vs 2.5ms (3× faster) |
| Deployment reach | **futo-basque** | Runs inside FUTO Keyboard (real Android app) |

**For next-word prediction, futo-basque v2.0.0 is now the winner** — 56.2% top-1
vs morpheus's 43.8%, on the same eval set and engine. This is a complete reversal
from the previous comparison (Jul 15), where futo-basque scored 0% due to a
pretrain double causal-shift bug (now fixed) and format contamination from the
sequential finetune pipeline (replaced by unified multitask finetune).

futo-basque achieves this with **3.6× fewer parameters** (25M vs 91M) and **3×
faster inference** (0.8ms vs 2.5ms). The two models are now tied on top-5
(75.0% each), meaning both have similar near-miss recovery — but futo-basque
converts more of those into top-1 hits.

The remaining gap: futo-basque's standalone autocorrect is 0% (the multitask
finetune's 40% triple ratio wasn't enough to master the control-token format).
In the real FUTO app, the hybrid dictionary engine should compensate, but a
Basque dictionary wordlist is still needed. morpheus has no autocorrect at all.

### 1.6 Keystrokes-saved on realistic messaging

The top-1/top-5 accuracy on 16 hand-picked prompts (§1.2) has a methodological
problem: the "gold" answers are often debatable (`Ez dut uste` = "I don't think
so" is perfect Basque, but gold lists `ahaztu/dakit/maite`), and "accuracy on
prompts" doesn't measure the actual keyboard user experience.

A better metric: **keystrokes saved**. We simulate typing 20 realistic Basque
messaging messages (WhatsApp/Telegram-style one-person messages) word by word.
After each completed word + space, we ask the model for the next-word suggestion.
If the intended next word appears, we count its character length as keystrokes
saved. This directly measures "how much typing does the model eliminate?"

Script: `futo-transformer-basque/scripts/eval/keystrokes.py`.

#### Results

| Metric | futo-basque v2.0.0 | morpheus | Winner |
|--------|:---:|:---:|:---:|
| **Top-1 keystrokes saved** | **8.4%** (61/726 chars) | 6.2% (45/726) | futo-basque |
| **Top-5 keystrokes saved** | **28.9%** (210/726 chars) | 23.8% (173/726) | futo-basque |
| Top-1 words correct | 14/137 | 11/137 | futo-basque |
| Top-5 words correct | 45/137 | 39/137 | futo-basque |

futo-basque wins on every metric. On a typical 40-character message, its top-1
suggestion saves ~4 keystrokes (tap the first suggestion instead of typing
those chars); the full top-5 suggestions bar saves ~12 characters (~29% of the
message).

Both models are honestly modest — the first word is never predictable (no
context), and many words are content words that are genuinely unpredictable.
But futo-basque's edge shows on the predictable patterns: `Eskerrik → asko`,
`Euskara ikasten → ari`, `Ongi → etorri`, `Non → dago`, `Zein filma → ikusi`.

#### Example: `Eskerrik asko denagatik oso ondo pasa nuen`

```
  ✓ ...Eskerrik ▎                → asko      (model: asko)   saved: +4 chars
    ...Eskerrik asko ▎           → denagatik (model: zure)   saved:  0
    ...Eskerrik asko denagatik ▎ → oso       (model: eta)    saved:  0
    ...rrik asko denagatik oso ▎ → ondo      (model: pozik)  saved:  0
  ✓ ...asko denagatik oso ondo ▎ → pasa      (model: pasa)   saved: +4 chars
    ...denagatik oso ondo pasa ▎ → nuen      (model: duzuen) saved:  0
  total: 8/28 predictable chars saved = 29%
```

The model nails the formulaic openings (`Eskerrik asko`, `oso ondo pasar`)
where the corpus has strong collocations, and misses the content/variable
words (`denagatik`, `nuen`) that are genuinely open-ended.

#### Updated verdict (keystrokes-saved)

The keystrokes-saved metric confirms the §1.2 conclusion: **futo-basque is the
better next-word model**, and does so with 3.6× fewer parameters. The advantage
is modest in absolute terms (8.4% vs 6.2% top-1) but consistent across all four
metrics. For a keyboard where every keystroke matters, futo-basque eliminates
~29% of typing via the suggestions bar.

---

## Part 2 — FUTO Engineering Strategies: Deep Review

FUTO's C++ inference layer (`JNI_LanguageModel.cpp`, ~1300 lines) is a masterclass in
production keyboard-LM engineering. Below, each strategy is analyzed for applicability
to morpheus. Strategies are ordered by impact.

### Strategy 1: Contextual Logit Banning (`transform_logits`) ⭐ HIGH IMPACT

**What FUTO does:** Before sampling, FUTO modifies the raw logits based on context:

- **Word-separator banning:** All tokens containing symbols (`.!@#$%^&*()…`) are banned
  (`-999.0f`), and their probability mass is **redirected to the SPACE token**. This
  forces the model to end words at spaces, not punctuation.
- **First-token-of-word banning:** Tokens starting with `'` or `-` are banned at word
  start (prevents `'-word` predictions).
- **Capitalization-aware banning:** If the user is typing `FirstCapital` or `AllCapitals`,
  lowercase-leading tokens are banned. This prevents suggesting `etxea` when the user
  typed `Etxe`.
- **General banning:** Known-bad tokens (e.g. `"-▁"`) are always banned.

**Applicability to morpheus:** **HIGH.** Morpheus's inference engineering (§5.4) handles
the tokenization trap with retokenization fallback, but does **no logit-level
constraining**. Adding capitalization-aware banning and symbol→space redirection would
directly improve candidate quality — morpheus currently can suggest lowercase
continuations of capitalized words, and punctuation-terminated tokens pollute the
candidate pool. This is a low-effort, high-yield addition to `src/eval_utils.py` and
the demo server.

### Strategy 2: Confidence-Based Prediction Modes ⭐ HIGH IMPACT

**What FUTO does:** FUTO classifies every prediction into one of three modes based on
the **ratio** between top-1 and top-2 probability:

```
if top1 > threshold × top2:        → "autocorrect" (auto-replace)
else if top1 > (threshold×0.1)×top2: → "uncertain" (show as suggestion)
else:                               → "clueless" (don't suggest)
```

This is a **calibration-free** heuristic — it uses relative probability ratios, not
absolute thresholds, so it doesn't need temperature tuning per model. "Clueless" mode
gets a massive score penalty (`probMult: 500000→10, probOffset: 100000→-100000`),
effectively hiding bad predictions.

**Applicability to morpheus:** **HIGH.** Morpheus's demo shows 3 chips but has no
notion of "should I even suggest?" — it always shows candidates. The confidence-ratio
heuristic would let morpheus suppress low-confidence predictions (reducing noise) and
potentially enable an autocorrect mode (auto-accept when top-1 >> top-2). This directly
addresses morpheus's future-work item about repetition loops and noisy completions
(§7.1.2) — "clueless" mode would hide them.

### Strategy 3: Two-Engine Merge (Dictionary + LM Rescoring) ⭐ HIGH IMPACT

**What FUTO does:** FUTO runs **two prediction engines in parallel** and merges:

1. **Classical AOSP dictionary + bigram engine** — high recall, knows the full
   wordlist, handles OOV via edit distance.
2. **Transformer LM** — high precision, context-aware.

The LM can **rescore the dictionary's candidates** (`rescoreSuggestions`): it takes the
dictionary's word list + scores, normalizes them, computes the LM probability of each
candidate's first token (÷ token count for length normalization), and reweights. This
combines dictionary recall with LM precision.

**Applicability to morpheus:** **HIGH — and this is morpheus's biggest gap.** Morpheus
is a pure LM with no dictionary fallback. For rare words, proper nouns, or OOV items the
LM doesn't know, morpheus simply fails. A lightweight Basque wordlist (frequency
dictionary) + LM rescoring would:
- Rescue OOV/rare words the LM misses (high recall)
- Let the LM's context re-rank them (high precision)
- Provide a fallback when the LM is "clueless"

This is arguably more valuable than morpheus's planned distillation (§7.2.3) for
improving real-world autocomplete hit rate. FUTO's rescoring formula
(`transformedScore × logits[first_token] / n_tokens`) is a simple, effective pattern.

### Strategy 4: The Keypress Autocorrect Format (`<XBU>…<XBC>…<XEC>`) ⭐ HIGH IMPACT (new capability)

**What FUTO does:** This is FUTO's signature innovation. Instead of word completion
(prefix → full word), FUTO does word **correction** (typed chars → correct word):

```
<context> <XBU> <CHAR_T><CHAR_E><CHAR_H> <XBC> The <XEC>
                   ↑ typed keystrokes          ↑ model predicts correction
```

Each typed character is encoded as a discrete `<CHAR_X>` token (accent-stripped,
uppercased via NFD). The model sees *exactly what was typed* and predicts what was
*meant*. This elegantly handles typos, transpositions, fat-finger errors.

**Applicability to morpheus:** **HIGH — would add a capability morpheus entirely lacks.**
Morpheus does word completion, not correction. If morpheus adopted this format via a
finetune (synthetic typo→correct pairs, like futo-basque's Phase 4a), it could do
*both* next-word prediction AND autocorrect. The 91M model has enough capacity to learn
both registers. This would make morpheus a complete keyboard LM, not just a predictor.

**Caveat:** This requires the `<CHAR_A>…<CHAR_Z>` structural tokens in the vocabulary
and a finetune dataset of typo→correct pairs. futo-basque's `typo_synthesis.py` and
`generate_triples.py` are directly reusable references.

### Strategy 5: Beam Search with Probability Products ⭐ MEDIUM IMPACT

**What FUTO does:** FUTO's `Sample()` function does proper beam search for multi-token
word candidates:

1. Keep `NUM_RESULTS=3` parallel sequences (each gets its own KV cache slot via
   `llama_kv_cache_seq_cp`).
2. At each step, extend each sequence with top-k tokens, multiply probabilities
   (`P(seq) = P(tok₁) × P(tok₂) × …`), re-sort, keep top-3.
3. Stop when a sequence hits `<XEC>` (end of correction) or a word boundary (space token).
4. Handles the case where multiple children come from the same parent (reassigns seq_id
   + copies KV cache).

**Applicability to morpheus:** **MEDIUM.** Morpheus uses greedy decoding + retokenization
fallback (parallel prefix queries). Beam search would produce better multi-token
completions (proper probability scoring vs. greedy), but:
- Mamba-2 has no KV cache to parallelize across beams (each beam needs its own SSM state)
- The retokenization fallback already handles the main failure mode (tokenization trap)
- Beam search adds latency (multiple forward passes)

For morpheus, a **cheaper variant** — top-k sampling at each step with early stopping at
word boundaries — would capture most of the benefit without full beam search.

### Strategy 6: Char Embedding Mixing (`char_embed_mixing_v1`) ◻ MEDIUM (mobile-only)

**What FUTO does:** FUTO doesn't just feed `<CHAR_T>` for a "T" keystroke — it takes the
**exact (x,y) screen coordinates** of the tap, decomposes them into proximity weights
across the 4 nearest keys (`ProximityInfo::decomposeTapPosition`), and **mixes the
embeddings** of those 4 `<CHAR_X>` tokens weighted by proximity. An `encoder_weight`
matrix can also map raw (x,y) → embedding (for swipe typing). This handles imprecise
touch typing — if you tap between T and R, the model sees a mix of both.

**Applicability to morpheus:** **LOW (current), MEDIUM (if mobile).** Morpheus's demo is
a text editor / browser keyboard, not a touch keyboard with coordinate data. But if
morpheus targets mobile (its §7.2.3 distillation goal), this is a brilliant strategy
for noisy touch input. The `encoder_weight` matrix approach (2D coordinates → embedding)
is more elegant than discrete token mixing.

### Strategy 7: Context Safeguarding (`safeguardContext`) ◻ LOW IMPACT

**What FUTO does:** Aggressively trims context before feeding the model:
- Max 70 chars or 16 words
- Trim to last sentence boundary (`.?!`)
- Then to last comma
- Then to last 5 words
- "Longest match" optimization: reuse previous context via suffix matching

**Applicability to morpheus:** **LOW.** Morpheus's Mamba-2 SSM state is **fixed-size**
(~16KB/layer) regardless of context length — there's no latency penalty for long
context. This is a key Mamba advantage (§5.2: "decode speed is context-length
independent"). FUTO needs context trimming because transformer KV cache grows linearly;
morpheus doesn't. However, the "recent context matters most" intuition is still valid
for quality — very old context can mislead. A soft version (trim to last 1-2 sentences)
could help quality without being a latency necessity.

### Strategy 8: On-Device LoRA Personalization (`AdapterTrainer`) ◻ MEDIUM (already planned)

**What FUTO does:** Fine-tunes the model **on the phone** from the user's typed text:
- LoRA r=16, α=16, 1 epoch, 128 iterations, 6 threads, n_ctx=64
- Training examples = user's typed text (trimmed + space-appended)
- Writes a full merged GGUF (LoRA applied + saved)
- Requires `lora_finetunable_v1` metadata flag

**Applicability to morpheus:** **MEDIUM — already in morpheus's future work (§7.3.5).**
Morpheus explicitly plans "on-device personalization from completion logs." FUTO's
implementation is a concrete reference: the hyperparameters (r=16, 128 iters, n_ctx=64)
are sensible defaults for a keyboard. Morpheus's completion logs (§5.4.6) already
capture (context, target) pairs — they're a ready-made personalization dataset. The
91M scale is large enough to meaningfully shift toward a single user's distribution
(15-40× larger than Gboard's federated LMs).

**Mamba-2 + LoRA feasibility:** LoRA adapts linear projections (q/k/v in transformers,
in_proj/out_proj in Mamba). Mamba-2's `in_proj` and `out_proj` are standard linears —
LoRA applies directly. No architectural barrier.

### Strategy 9: Personal Dictionary as Glossary Prompt (`addPersonalDictionary`) ◻ LOW IMPACT

**What FUTO does:** If the user has a personal dictionary, FUTO prepends a fake context:
```
(Glossary: word1, word2, word3)

<actual context>
```
The model sees these words and can suggest them. Zero-shot personalization — no
training needed.

**Applicability to morpheus:** **LOW-MEDIUM.** Simple and clever. For user-specific
terminology (names, jargon), prepending a glossary is a zero-cost way to bias
predictions. morpheus could add this to its demo server in ~10 lines. Less powerful
than LoRA personalization but zero-effort.

### Strategy 10: Exact-Match Boost + Length Sanity Check ◻ LOW IMPACT

**What FUTO does:**
- **Exact-match boost:** If a correction candidate exactly matches the typed partial word,
  boost it (subtract 1.0 from non-exact matches). Respects "user typed it right, keep it."
- **Length sanity:** If prediction is < half the typed word's length, force "clueless."
  Prevents absurd short predictions for long typed words.
- **Banned words:** User-configurable banned predictions, with hash-based wildcard
  sequence matching.

**Applicability to morpheus:** **LOW.** Simple sanity guards. The exact-match boost is
relevant for morpheus's word completion (if the model's continuation matches what the
user already typed, prefer it). The length sanity check prevents garbage. Trivial to add.

### Strategy 11: KV Cache Fast-Forward + Mix Caching ❌ N/A (architecture-specific)

**What FUTO does:** Caches the decoded prompt's KV state and only re-processes the delta
when context changes (`transformer_context_fastforward`). Also caches mixed embeddings
(`GetCachedMixAmount` checks which char-mixes haven't changed).

**Applicability to morpheus:** **N/A.** Mamba-2 has no KV cache — its SSM state is
fixed-size and updated incrementally. The concept ("don't recompute what hasn't
changed") is already inherent in Mamba's recurrent state. However, morpheus's demo
server could cache the SSM state across queries (if llama.cpp supports state
checkpointing for Mamba) to avoid re-prefilling the context each time.

---

## Part 3 — Summary: What Morpheus Should Adopt

### Tier 1: High-impact, low-effort (do first)

| # | Strategy | Effort | Why |
|---|----------|--------|-----|
| 1 | **Contextual logit banning** (capitalization, symbol→space) | Small | Directly improves candidate quality in `eval_utils.py` + demo |
| 2 | **Confidence-ratio modes** (autocorrect/uncertain/clueless) | Small | Suppresses noise, enables autocorrect mode, fixes repetition-loop UX |
| 3 | **Two-engine merge** (wordlist + LM rescoring) | Medium | Biggest gap — rescues OOV/rare words; FUTO's rescoring formula is reusable |

### Tier 2: High-impact, medium-effort (new capabilities)

| # | Strategy | Effort | Why |
|---|----------|--------|-----|
| 4 | **Keypress autocorrect format** (`<XBU>…<XEC>` + `<CHAR_X>`) | Large | Adds autocorrect capability morpheus entirely lacks; needs vocab slots + finetune |

### Tier 3: Already planned or situationally useful

| # | Strategy | Effort | Why |
|---|----------|--------|-----|
| 5 | On-device LoRA personalization | Medium | Already in morpheus §7.3.5; FUTO's hyperparameters are a reference |
| 6 | Char embedding mixing | Medium | Only if morpheus targets mobile touch keyboard |
| 7 | Beam search (lightweight variant) | Medium | Better multi-token completions; Mamba makes full beam costly |
| 8 | Glossary prompt / exact-match boost / length sanity | Small | Trivial sanity guards and zero-shot personalization |

### The reverse: what futo-basque should adopt from morpheus

For completeness — futo-basque's broken next-word (0% top-1, 100% format overfit) could be fixed by adopting
morpheus's **inference engineering** (§5.4):

1. **Retokenization fallback** — query from progressively shorter prefixes to escape
   the tokenization trap (the reason futo-basque emits `<XEC>` is format overfit, but
   retokenization would also help genuine completion failures).
2. **Sticky merge** — carry forward previous candidates when switching from next-word
   to completion mode.
3. **A clean-text recovery finetune** — a small finetune on plain text (no `<XBU>`
   format) after Phase 4c, to teach the model the register shift between autocorrect
   mode and next-word mode.

---

## Part 4 — The Deeper Architectural Insight

The comparison reveals a fundamental **architecture-task alignment** question:

- **FUTO chose transformer + KV cache** because the keypress autocorrect format needs
  to attend to each `<CHAR_X>` token individually (the model must see *which keys were
  pressed*). The KV cache lets it incrementally process keystrokes. The cost: KV cache
  grows with context, requiring aggressive context trimming.

- **Morpheus chose Mamba-2 SSM** because next-word/ghost-text prediction needs constant
  latency over long sessions. The SSM state is fixed-size — context length doesn't
  affect decode speed. The cost: no easy way to do the per-keystroke attention that
  autocorrect needs (SSM compresses history into a fixed state, can't "look back" at
  individual keystrokes).

**The key question for morpheus:** Can Mamba-2 learn the `<XBU>…<CHAR_X>…<XEC>`
autocorrect format? Theoretically yes — the SSM state can encode "these keystrokes were
typed" — but it may struggle with long words (20+ keystrokes) where a transformer's
attention can directly attend to each `<CHAR_X>`. This is an **empirical question worth
testing** before committing to adding autocorrect to morpheus. A small experiment
(finetune morpheus on futo-basque's Phase 4a data) would answer it.

If Mamba-2 can learn the format, morpheus could become a **complete keyboard LM**
(next-word + autocorrect + O(1) latency) — strictly more capable than futo-basque in
every dimension except deployment reach (FUTO app vs custom demo).
