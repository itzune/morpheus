# Morpheus Trajectory: Desktop Text-Editor Autocompletion

> **Date:** 2026-07-17
> **Purpose:** Define the strategic direction for Morpheus — pivoting from a
> general on-device Basque autocomplete to a **desktop text-editor
> autocompletion model** as its primary target — and record the key technical
> decisions that follow.
> **Status:** Direction approved; implementation not started.
> **Related:** [`../COMPARISON.md`](../COMPARISON.md) · [`copilot-inline-completion-research.md`](./copilot-inline-completion-research.md) · [`verb-agreement-finetune-research.md`](./verb-agreement-finetune-research.md) · [`On-Device Model Adaptation Feasibility.md`](./On-Device%20Model%20Adaptation%20Feasibility.md)

---

## TL;DR

Morpheus was built as an on-device Basque autocomplete and evaluated across
mobile-keyboard and desktop-ghost-text scenarios. After a full comparison
against the sibling futo-basque transformer project (`COMPARISON.md`), the
evidence points to a clear product direction:

**The desktop text editor is Morpheus's primary deployment target.** The model's
strengths — Mamba-2's constant-latency decoding, a morphology-aware 4K tokenizer,
and a 55 MB quantized footprint — align best with the editor-autocomplete use
case (ghost-text / inline completion in a document the user is actively writing),
*not* the mobile IME (where the futo-basque transformer's per-keystroke
attention and FUTO-app integration win, and where the market is already served).

To fully serve the editor target, three technical decisions define the roadmap:

1. **Adopt Fill-in-the-Middle (FIM) via continued pretraining** (not a
   finetune), to make the model cursor-position-aware.
2. **Generate a Basque FIM dataset** as a transform of the existing corpus
   (PSM/SPM char-level splitting), not new data collection.
3. **Integrate via the OpenAI-compatible `/v1/completions` protocol through
   Continue.dev**, with a custom FIM template — not a bespoke editor plugin,
   and not the chat-completions API.

Each is justified below.

---

## 1. Why desktop text-editor autocompletion

### 1.1 The comparison settled the target question

The head-to-head with futo-basque (`COMPARISON.md` §1.5–§1.6) showed that, for
pure next-word prediction, the two models are **more similar than different** —
on the 16-prompt eval they get the *same* 7 right and the *same* 7 wrong, and the
keystrokes-saved gap is modest (8.4% vs 6.2% top-1). Neither architecture is a
clear winner on accuracy.

The architectures diverge on a different axis: **task alignment**.

| Axis | Mamba-2 (morpheus) | Transformer (futo-basque) |
|------|:---:|:---:|
| Single-token latency | 2.5 ms | 0.8 ms (smaller model) |
| Long-context decode latency | **O(1)** (fixed SSM state) | grows with KV cache |
| Per-keystroke attention (autocorrect) | weak (state compression) | **strong** (attention) |
| Editor ghost-text fit | **strong** | adequate |
| Mobile IME autocorrect fit | weak | **strong** (in FUTO app) |

The key insight (`COMPARISON.md` Part 4): the transformer's attention is
structurally suited to the FUTO keypress-autocorrect format (`<XBU>…<CHAR_X>…`),
where each keystroke must be attended to individually. Mamba-2's fixed-state
compression is structurally suited to **constant-latency next-word/ghost-text
prediction over long sessions** — exactly the editor use case.

futo-basque also has a concrete deployment channel (the FUTO Keyboard Android
app) that morpheus lacks. Competing there would mean building an Android IME
against a transformer that's already integrated. That is a poor use of morpheus's
differentiating architecture.

### 1.2 The mobile IME path is de-prioritized, not abandoned

The Android-keyboard research ([`android-keyboard-research.md`](./android-keyboard-research.md))
remains valid — morpheus *could* ship as an Android IME via HuoziIME-style
forking, and its O(1) latency is a genuine advantage there. But:

- The autocorrect-format experiment (`TODO.md` Priority 2) is an open empirical
  question with real SSM-compression risk for long words. It is a research bet,
  not a product bet.
- The editor path requires no such gamble — ghost-text is pure next-token
  generation, which morpheus already does well.

**Decision:** desktop editor autocomplete is primary; Android IME is a
secondary research track that depends on the autocorrect-format experiment
succeeding.

### 1.3 What "desktop text-editor autocompletion" means concretely

The target UX is **inline ghost-text completion** (like GitHub Copilot's
suggestions, or Gmail Smart Compose) inside a desktop text editor — VS Code,
JetBrains, Neovim, or a plain text editor — while a user writes Basque prose.

This differs from the mobile-keyboard UX in three ways that matter for the model:

1. **Cursor-in-the-middle.** The user's cursor sits inside an existing document.
   The model has both *prefix* (text before cursor) and *suffix* (text after).
   An autoregressive-only model can use only the prefix; the suffix is wasted.
2. **Longer, richer context.** Sessions are long documents, not one-line chat
   bubbles. Mamba-2's O(1) decode is a real advantage here.
3. **Multiline completion.** Editors accept multi-line ghost text, not just the
   next word. The model should be able to complete a phrase or sentence.

Point 1 is the gap. The model is currently AR-only — it cannot see the suffix.
Closing that gap is decision #1.

---

## 2. Decision 1 — Fill-in-the-Middle via continued pretraining

### 2.1 The problem: AR-only models are cursor-blind

Today the demo server ([`demo/server.py`](../demo/server.py)) does prefix-only
ghost-text: `smart_context` trims trailing subword fragments, and the model
generates *forward* from the cursor. The suffix (everything after the cursor)
is never seen. For a user editing mid-sentence, this discards half the
available signal and produces completions that ignore the sentence's actual
ending.

**Fill-in-the-Middle (FIM)** fixes this. Bavarian et al. (2022), *"Efficient
Training of Language Models to Fill in the Middle,"* showed that an
autoregressive model can be made infill-capable purely by *rearranging the
training data* — no architecture change:

```
Original:  "Euskara ikasten ari naiz"
PSM:       <PRE> Euskara ikas <SUF> naiz <MID> ten ari <EOT>
SPM:       <SUF> naiz <PRE> Euskara ikas <MID> ten ari <EOT>
```

At inference, the editor wraps the real prefix/suffix with the FIM tokens and
the model generates the `<MID>` content, conditioned on *both* sides.

### 2.2 Continued pretraining, not a finetune — and why

The literature is emphatic on this point (GoPenAI guide; Bavarian et al.):

- **"FIM-for-free":** mixing FIM-transformed data with plain AR data during
  *pretraining* preserves AR performance while adding infill capability — no
  trade-off. Optimal FIM rate is **50–90%**.
- **Retrofitting FIM via finetuning is expensive and loss-prone:** high learning
  rates needed to overcome ingrained AR patterns, catastrophic-forgetting risk,
  and the original paper cites ~50B extra tokens for 162M–13B models.

For morpheus, this means **a continued-pretraining stage** (call it **Phase 6**)
that resumes from the current 74K-step checkpoint and trains ~1–2B additional
tokens with a 50% FIM / 50% AR mix. The 91M scale is small enough that the
"50B tokens" figure does not apply; a 1–2B-token pass is proportionate.

**Why not a separate FIM model:** the FIM-for-free property means one checkpoint
does both. At inference, the editor's FIM template (with `<PRE>/<SUF>/<MID>`)
selects infill mode; a bare prompt selects plain AR mode. No model switching.

### 2.3 Tokenizer change required

The current 4K SentencePiece vocab has only `<unk>(0)`, `<s>(1)`, `</s>(2)`.
Four FIM special tokens must be added:

| Token | Role |
|-------|------|
| `<PRE>` | marks start of prefix |
| `<SUF>` | marks start of suffix |
| `<MID>` | marks start of (generated) middle |
| `<EOT>` | end of FIM sequence (generation stop token) |

This expands vocab to 4004 (then padded to a multiple of 16 → 4000 is already
a multiple of 16, so 4004 → pad to 4016). The embedding table must be resized
and the new rows initialized. Because embeddings are a small fraction of the
91M params (~3M), the new tokens train quickly.

**Token-name choice:** the `<PRE>/<SUF>/<MID>/<EOT>` names match the original
FIM paper and Code Llama, and are simple to template. The Qwen-style
`<|fim_prefix|>` names are equivalent; the choice is cosmetic and only affects
the editor-side template string.

### 2.4 What FIM does *not* solve

FIM makes the model cursor-aware; it does not by itself improve Basque
morphological correctness. The verb-agreement failures
([`verb-agreement-finetune-research.md`](./verb-agreement-finetune-research.md))
— three-way dative agreement (*nor-nori-nork*) — are orthogonal and still
require the gap-fill SFT (Strategy B) or a hybrid re-ranker (Strategy C). FIM
and verb-agreement SFT compose: FIM teaches *where* to fill, SFT teaches *what*
Basque is correct there.

---

## 3. Decision 2 — Generate a Basque FIM dataset (transform, not collect)

### 3.1 FIM data is free

The decisive practical point: **FIM training data is generated by transforming
existing text, not by authoring new examples.** Every sentence in the Latxa v2
corpus becomes a FIM training example via the split-and-rearrange operation.
No new data collection, no labeling, no Basque-specific gap-fill authoring is
required for the *pretraining* stage.

### 3.2 Recipe (from the research)

```
def make_fim(text, mode="joint"):
    prefix, middle, suffix = char_level_split(text)   # NOT token-level
    if mode == "joint":
        mode = "PSM" if random() < 0.5 else "SPM"     # 50/50, positive transfer
    if mode == "PSM":
        return [PRE] + enc(prefix) + [SUF] + enc(suffix) + [MID] + enc(middle) + [EOT]
    else:  # SPM
        return [SUF] + enc(suffix) + [PRE] + enc(prefix) + [MID] + enc(middle) + [EOT]
```

Key parameters, all evidence-backed:

- **Char-level splitting** (not token-level): robust against subword boundaries.
  This matters *a lot* for morpheus's 4K UNIGRAM vocab, where Basque's
  agglutinative morphology splits irregularly (`etxea` → `▁etx` + `ea` or
  `▁etxe` + `a` depending on context). Token-level splits would land on
  arbitrary subword seams; char-level splits generalize.
- **Joint PSM/SPM** (50/50): empirical evaluation shows SPM has a slight edge
  (better KV-cache efficiency at inference) but joint training gives positive
  transfer and deployment flexibility. Use both.
- **50% FIM / 50% AR mix**: preserves AR capability (FIM-for-free). Do not go
  above 90% — AR performance degrades.

### 3.3 Basque-specific consideration: mid-morpheme splits

Because Basque is agglutinative, a random char-level split will frequently land
*inside a suffix chain* (`etxe`|`koak`, `ikas`|`ten`). This is actually
desirable — it exercises exactly the morphological boundary completion that
matters for Basque — but it means the model's FIM quality is tightly coupled to
the tokenizer's morphological consistency (the 4K vocab was chosen for this,
[`tokenizer-fieldwork.md`](./tokenizer-fieldwork.md)).

A fraction (~20%) of splits should be biased toward **linguistically natural
boundaries** — sentence end, clause boundary, word boundary — to mimic where
editors' cursors actually sit. The majority stays random-char to preserve
generalization.

### 3.4 A Basque infilling eval is needed

Perplexity does not measure FIM quality well. The research emphasizes
**sampling-based infilling benchmarks** (Bavarian et al. introduced
"random-span infilling" with unit tests for code; for natural language, exact-
match and LLM-judge are the analogues). We should add a **Basque FIM eval**:
held-out sentences, mask a random span, measure whether the model's `<MID>`
generation reconstructs (or plausibly continues) the original. This complements
the existing CSR/MorphAcc metrics, which are AR-only.

---

## 4. Decision 3 — Editor integration via OpenAI-compatible completions + Continue.dev

### 4.1 The transport question: which API?

The options for wiring a local model into an editor:

| Option | Fit | Notes |
|--------|-----|-------|
| OpenAI `/v1/completions` (text completion) | ✅ **best** | Raw prompt + streaming + stop tokens. Implemented by llama.cpp, Ollama, vLLM, LM Studio. |
| OpenAI `/v1/chat/completions` | ❌ wrong shape | Message-array/role abstraction adds nothing for FIM and gets in the way. |
| MCP (Model Context Protocol) | ❌ wrong layer | For tools/context providers, not autocomplete transport. |
| Bespoke editor plugin | ❌ avoid | Duplicates Continue.dev's work; narrows editor support. |
| llama.cpp native `/completion` | ⚠️ low-level | What the demo server already proxies; fine as a backend, not as the editor-facing API. |

**The key clarification:** "OpenAI deprecated the completions API" refers to
OpenAI's *hosted* endpoint (retired Jan 2024). The *protocol* is alive — it is
the de-facto lingua franca for local/self-hosted text completion, implemented
by every local inference server. "Deprecated by OpenAI" ≠ "dead standard."

For FIM, chat completions is structurally wrong: FIM needs a raw prompt string
(the templated `prefix+suffix+<MID>`), streaming, and stop tokens. Every
production FIM system (Copilot, Codestral, Continue) uses the raw
text-completion shape.

### 4.2 The editor: Continue.dev

[Continue.dev](https://docs.continue.dev) is the open-source AI coding
assistant for VS Code and JetBrains. Critically, its **autocomplete** subsystem
is FIM-native and model-agnostic: it ships per-model FIM templates
(`core/autocomplete/templating/AutocompleteTemplate.ts`) for Qwen-Coder
(`<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`), Codestral
(`[SUFFIX]…[PREFIX]`), Code Llama (`<PRE>…<SUF>…<MID>`), Granite, StableCode,
and others.

**The integration path for morpheus:**

1. Run morpheus behind **llama-server** (already done by `demo/server.py`),
   which exposes `/v1/completions`.
2. Register a Continue provider pointing at `http://localhost:<port>/v1`,
   `provider: openai`, with `roles: [autocomplete]`.
3. Add a **morpheus FIM template** entry mapping morpheus's `<PRE>/<SUF>/<MID>`
   tokens, with stop tokens `<EOT>`, `<PRE>`, `<SUF>`, `<MID>`.
4. Set `prefixPercentage` / `maxSuffixPercentage` / `debounceDelay`.

This gives **VS Code + JetBrains** ghost-text autocomplete with *zero* editor
plugin code of our own, and inherits Continue's prefiltering, snippet
formatting, and multiline logic.

### 4.3 The token-ID fidelity caveat (the one real wrinkle)

Morpheus has a known issue ([`../README.md`](../README.md) GGUF model card;
[`demo/server.py`](../demo/server.py) `_prompt_to_ids` docstring): llama.cpp's
rebuilt SentencePiece tokenizer **diverges** from the reference `sentencepiece`
library on morpheus's 4K UNIGRAM vocab, costing ~7× autocomplete quality if
string prompts are used. The demo server solves this by sending **token IDs**
(`sp.encode(prompt, out_type=int)`) to llama-server's `/completion`, not
strings.

The OpenAI-compatible `/v1/completions` path that Continue speaks takes a
**string** prompt, which would re-trigger the divergence. Two mitigations:

- **Likely sufficient:** the divergence is worst on long agglutinative words.
  FIM splits at char boundaries, and the FIM tokens themselves are atomic, so
  the prefix/suffix pieces are shorter and less divergence-prone. This needs
  empirical confirmation, not assumption.
- **If fidelity matters:** keep a thin proxy (the existing `demo/server.py`
  already is one) that accepts Continue's FIM string, SP-tokenizes it, and
  forwards token IDs to llama-server `/completion`. ~50 lines on top of what
  exists. This preserves training/inference tokenization identity.

Start with the direct `/v1/completions` path; add the proxy only if the
divergence measurably hurts FIM quality.

### 4.4 What this gets us immediately

Even before the FIM model lands, Continue.dev works against the *current*
AR-only model (prefix-only completion, suffix ignored). This means **step 3 of
the roadmap can start today** against the existing 74K checkpoint, giving a
working editor plugin to dogfood while the FIM model trains.

---

## 5. Roadmap

| Phase | Goal | Depends on | Effort |
|-------|------|-----------|--------|
| **6a** | Add `<PRE>/<SUF>/<MID>/<EOT>` tokens; resize embeddings; resume-from-74K smoke test | nothing | small |
| **6b** | FIM data transform (char-level PSM/SPM) on Latxa v2; pretokenize FIM shards | 6a | medium |
| **6c** | Continued-pretrain Phase 6: ~1–2B tokens, 50% FIM / 50% AR | 6b | GPU time |
| **6d** | Basque FIM eval (random-span infilling on held-out sentences) | 6c | medium |
| **6e** | Export FIM checkpoint → GGUF; add morpheus FIM template to Continue.dev; dogfood in VS Code | 6d | small |
| **(parallel) 6e′** | Wire Continue.dev against the *current* AR checkpoint (prefix-only) for early dogfooding | nothing | small |
| **(research) R** | Test Mamba-2 + FUTO autocorrect format (`TODO.md` Priority 2) | nothing | medium — *separate track, gates the mobile-IME decision* |

**Critical path:** 6a → 6b → 6c → 6d → 6e.
**Fast-feedback path:** 6e′ (today) in parallel.

Phases 6a–6e are coupled: FIM tokens must exist before FIM data can be
tokenized, which must exist before Phase 6 training, which must finish before
the FIM eval/model is meaningful. Phase 6e′ is fully independent and should
start immediately to surface editor-integration UX issues while the model work
proceeds.

---

## 6. What we are explicitly *not* doing

- **Not switching architectures.** Mamba-2's O(1) decode is the differentiator
  for long-session editor autocomplete. The comparison showed the transformer
  wins only on autocorrect (a mobile-IME capability) and single-token latency
  (a function of morpheus being larger, not of architecture). The editor target
  plays to Mamba-2's strength.
- **Not competing with futo-basque on mobile.** That space is served by a
  transformer integrated into a real keyboard app. morpheus's value is
  elsewhere.
- **Not building a bespoke editor plugin.** Continue.dev already solves
  editor-side FIM templating, prefiltering, and multi-editor support. Reusing
  it is strictly better than reimplementing.
- **Not adopting the chat-completions API** for autocomplete transport. It is
  the wrong abstraction for FIM.
- **Not treating FIM as a finetune.** Continued pretraining is the
  evidence-backed path; finetuning FIM onto an AR model is the documented
  expensive/loss-prone alternative.

---

## 7. Open questions

1. **Mamba-2 + FIM quality at scale.** The FIM-for-free result is established
   for transformers. Mamba-2's fixed SSM state compresses prefix+suffix into a
   bounded representation; whether this degrades long-span infilling quality
   (vs a transformer's ability to attend back into the prefix/suffix) is an
   **empirical question** that only Phase 6c–6d answers. If degraded, the
   mitigation is shorter FIM spans (bias splits toward nearer boundaries),
   not an architecture change.
2. **Tokenizer divergence under FIM.** Does the llama.cpp↔SentencePiece
   divergence (§4.3) measurably hurt FIM, or is it masked by char-level
   splitting and atomic FIM tokens? Needs measurement before committing to
   the proxy.
3. **Basque infilling eval design.** What's the right metric — exact-match
   reconstruction, LLM-judge plausibility, or a keystrokes-saved analogue
   applied to mid-document cursor positions? The existing CSR/MorphAcc
   ([`eval-strategies.md`](./eval-strategies.md)) are AR-only and don't
   transfer.
4. **Multiline completion boundaries.** Editors accept multi-line ghost text.
   When should the model stop — at sentence end, at line end, at a fixed token
   budget? This is a stop-token/debounce policy question as much as a model
   question, and is best answered by dogfooding (Phase 6e′).

---

## 8. Summary

Morpheus's architecture is best suited to **desktop text-editor ghost-text
autocomplete**: long sessions, constant decode latency, morphology-aware
tokenization. The missing capability is **cursor-position awareness**, which
FIM provides. The plan is a continued-pretraining stage that adds FIM tokens
and trains on a 50/50 FIM+AR mix of the existing corpus, evaluated with a new
Basque infilling benchmark, and deployed through Continue.dev via the
OpenAI-compatible completions protocol. Every step is evidence-grounded in the
FIM literature and in the comparison findings; none requires an architecture
change or new data collection.
