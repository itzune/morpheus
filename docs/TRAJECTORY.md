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

### 1.4 Server-side baselines: Kimu 2B and Latxa 8B set the quality ceiling

While morpheus targets on-device O(1) latency, a larger server-side model
defines the quality bar the on-device model asymptotes toward — and the
baseline any FIM fine-tune must beat. We deployed **HiTZ/Latxa-Llama-3.1-8B**
(the HiTZ continued-pretraining of Llama-3.1-8B on 4.2B Basque tokens; the
*base* model, not the instruct variant) on the L40 GPU server via the same
`llama-server` + demo proxy, using the existing `llama-fim` backend
([`demo/backends.py`](../demo/backends.py)) — zero client changes, just an env
var swap (`MORPHEUS_BACKEND=llama-fim`).

**Head-to-head on the same 30-test CSR set** (`eval/targets.json`, free-acceptance,
1 token/step, same hardware, same day), against the two most popular Basque
LLM base models — Orai NLP's Kimu 2B (Gemma-2 CPT) and HiTZ's Latxa 8B
(Llama-3.1 CPT):

| Model | Params | Macro CSR | Accept rate | Avg conf |
|-------|-------:|----------:|------------:|---------:|
| Morpheus v3_fim Q5_K_M (on-device) | 91M | 24.83% | 27.5% | 0.35 |
| **Kimu 2B Q6_K** (server) | 2B | **34.10%** | 35.5% | 0.42 |
| Latxa 8B Q6_K (server) | 8B | 33.18% | **40.2%** | **0.49** |

Both Basque LLMs beat Morpheus by **~9 CSR points** (33–34% vs 24.8%) — the gap
a 2B+ pretrained backbone opens over a 91M from-scratch model. Notably, Kimu
2B *edges out* Latxa 8B (+0.9 pts) despite being 4× smaller: on this
autocomplete task, a 2B Basque-pretrained model is sufficient to reach the
8B quality ceiling. But the quality lead comes with a hard hardware cost that
fixes the deployment split:

| | Morpheus (91M, 64 MB) | Kimu 2B (2.1 GB) | Latxa 8B (6.6 GB) |
|---|---|---|---|
| **GPU (L40)** latency | 76 ms (105 tok/s) | 95 ms (84.5 tok/s) | 115 ms (70.4 tok/s) |
| **GPU (L40)** memory | 602 MiB VRAM | 3036 MiB VRAM | 6988 MiB VRAM |
| **CPU laptop** latency | 196 ms (40.7 tok/s) | 1439 ms (5.6 tok/s) | **2869 ms (2.8 tok/s)** |
| **CPU laptop** memory | 266 MiB RAM | 2357 MiB RAM | 6648 MiB RAM |
| Autocomplete-viable on CPU? | **yes** | no (9.6× over) | **no** (19× over budget) |

On the L40 all three clear the 150 ms threshold and the choice is a
quality/VRAM trade. Latency scales cleanly with model size: 76 / 95 / 115 ms.
Kimu is the efficiency frontier — it matches Latxa's CSR at 43% less VRAM and
19 ms lower latency. The GPU itself is not the bottleneck — utilization across
all three processes on the single card is 31% mean / 78% peak. The host-CPU
signal runs *counter* to model size: Latxa's and Kimu's model servers idle at
0% host CPU (the `llama-fim` backend hands llama-server a plain string and the
GPU does all the work), whereas morpheus's `morpheus-sp-fim` backend burns 2%
host CPU on SentencePiece encoding + retokenization-fallback in Python — the
smaller model is cheaper to *run* but costlier to *serve*. On a CPU laptop
Latxa 8B collapses to ~2.9 s/request and 6.6 GB RAM — not a real-time model
off-Gpu — Kimu 2B is 2× faster (~1.4 s, 5.6 tok/s) but still 9.6× over budget,
while morpheus stays at 40.7 tok/s. So the two tiers are not a
preference but a constraint: **morpheus is the only one that runs on the edge;
Kimu and Latxa are the server-side ceiling** (and the FIM fine-tune
candidates, see point 3 below). Full benchmark:
[`comparison.md`](../eval/demo-results/20260719_latxa_vs_morpheus/comparison.md#latency--resource-footprint).

#### Domain examples — where the quality difference is visible

The CSR gap is real but abstract. The difference is clearer in side-by-side
continuations across three writing domains (greedy, 12 tokens):

**Email writing** — `Egun on! Astelehenean bilera bat egitea proposatzen`
("Good morning! I propose holding a meeting on Monday")

| Model | Continuation | Conf |
|-------|-------------|-----:|
| Latxa 8B | `dizuet, 18:00etan. Bilera` *("to you [pl.], at 18:00. Meeting…")* | 0.45 |
| Kimu 2B | `dizut. -Bai, noski. Noiz` *("to you [sg]. -Yes, of course. When…")* | 0.42 |
| Morpheus | `dizugu, eta, ondoren, egutegia eta` *("we propose, and, then, the calendar and")* | 0.34 |

All three pick a valid dative verb form. Latxa commits to a concrete meeting
time; Kimu drifts into a dialogue response; morpheus drifts into generic
`eta, ondoren` connective filler.

**Essay / article** — `Adimen artifizialak Hezkuntzan izango duen eragina`
("The impact AI will have on Education")

| Model | Continuation | Conf |
|-------|-------------|-----:|
| Latxa 8B | `aztertuko dute Euskal Herri` *("will examine [in] the Basque Country")* | 0.56 |
| Kimu 2B | `aztertuko dute bihar, Elkargu` *("will examine tomorrow, the Council")* | 0.55 |
| Morpheus | `aztertzeko, EHUko ikertzaile talde batek,` *("to examine, a UPV/EHU research team,")* | 0.33 |

All three correctly continue with *aztertu* (examine) — the strongest
collocation. Kimu and Latxa are near-identical in confidence (0.55 vs 0.56);
morpheus lags.

**Educational / technical** — `Suhesia sareko komunikazio guztiak`
("The firewall [?] all network communications")

| Model | Continuation | Conf |
|-------|-------------|-----:|
| Kimu 2B | `kontrolatzeko eta kudeatzeko erabiltzen` *("to control and manage [used]")* | 0.45 |
| Latxa 8B | `zifratuta daude, eta ez dago er` *("are encrypted, and there is no…")* | 0.42 |
| Morpheus | `, 100.000 biztanletik gora` *(", over 100,000 inhabitants")* | 0.35 |

Here the models diverge sharply. Both Kimu and Latxa stay on-topic (network
security → control/encryption). Kimu's continuation is arguably more natural
(firewalls control traffic) and carries higher confidence. Morpheus drifts to
an unrelated demographic filler — a hallmark of an underpowered model latching
onto a high-frequency statistical pattern instead of the sentence's semantic
thread.

#### What this establishes

1. **The two-tier positioning is validated.** Morpheus (91M, 55 MB Q4) remains
   the on-device/edge model; Kimu 2B (2.1 GB Q6) and Latxa 8B (6.6 GB Q6) are
   the server-side models. They are complementary, not competing — the +9 CSR
   points and cleaner output justify the server deployment where latency budget
   allows. Kimu 2B is the efficiency frontier: it matches Latxa's CSR at 4×
   smaller size.
2. **This is the *base* model with no FIM training.** AR append (cursor at
   end-of-buffer) works well; FIM infill (cursor mid-sentence) is non-functional
   — Latxa emits EOS immediately on the `<|fim_begin|>` sentinels it has never
   seen. That gap is precisely what a FIM continued-pretraining stage would fill.
3. **Kimu 2B is the leading FIM fine-tune candidate, Latxa 8B the fallback.** The original
   plan (Option B: Qwen3.5-2B-Base) preserved Mamba-family architectural
   continuity. But Kimu 2B (Gemma-2 CPT on Basque) is already Basque-adapted (no
   CPT needed), already beats morpheus by +9 CSR points as-is (34.10% — even
   edging out Latxa 8B's 33.18%), and its 2B size makes full-FT on one L40 fast
   and memory-feasible (~24–30 GB). Latxa 8B remains a fallback if a higher
   quality ceiling is worth the longer training time. (The 2B Qwen path is now
   the lowest-priority fallback, superseded by Kimu.)

Full comparison + raw result files: [`eval/demo-results/20260719_latxa_vs_morpheus/`](../eval/demo-results/20260719_latxa_vs_morpheus/comparison.md).

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

### 4.2 A reusable Basque inference server, not a Continue integration

The architecture is the **inference-server pattern** (how vLLM, TGI, Ollama,
and LM Studio all work): one server, many thin clients. The model-specific
knowledge — the reference SentencePiece encoder and (later) the FIM template —
lives server-side. Every client speaks the standard OpenAI completion API and
stays generic.

The server is `demo/server.py`. Its backend already exists: `_call_llama()`
SP-encodes a string to token IDs and forwards them to llama-server's native
`/completion` endpoint. The missing piece is an **OpenAI-compatible face** on
the front — two routes:

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

- **`/v1/completions`** — the OpenAI-ecosystem standard. Takes an
  already-templated string (`prompt`), `max_tokens`, `temperature`, `top_k`,
  `stop`, `stream`; SP-encodes; forwards token IDs; reformats the response to
  `choices[0].text` / `usage`. Continue.dev, Cody, codecompanion.nvim, and
  Obsidian plugins (Text Generator, Smart Composer) all speak this with **zero
  code from us**.
- **`/v1/complete`** — a convenience route taking `{prefix, suffix, max_tokens}`
  as separate fields. The server applies the FIM template (`<PRE>{prefix}<SUF>{suffix}<MID>`)
  once P6 lands. This is the endpoint ~50-line bespoke clients use — an Obsidian
  plugin or a Vim function sends raw prefix/suffix and renders the returned text
  as ghost, never needing to know morpheus's token names.

Putting the FIM template on the server (not in each client) is the key
decision for reuse: if the model's FIM token names ever change, only the server
changes; every client keeps working.

[Continue.dev](https://docs.continue.dev) is the **first client**, chosen
because its autocomplete subsystem is FIM-native and model-agnostic (it ships
per-model FIM templates — Qwen-Coder, Codestral, Code Llama, Granite,
StableCode). Continue points at `http://localhost:<port>/v1`, `provider: openai`,
`roles: [autocomplete]`, and uses its own FIM template against
`/v1/completions`. This gives VS Code + JetBrains ghost-text with zero editor
plugin code, inheriting Continue's prefiltering, snippet formatting, and
multiline logic.

### 4.3 The tokenization fix lives server-side (not conditional)

Morpheus has a known issue ([`../README.md`](../README.md) GGUF model card;
[`demo/server.py`](../demo/server.py) `_prompt_to_ids` docstring): llama.cpp's
rebuilt SentencePiece tokenizer **diverges** from the reference `sentencepiece`
library on morpheus's 4K UNIGRAM vocab, costing ~7× autocomplete quality if
string prompts are used. The same AR model scored **0% via string prompts vs
43.8% via token IDs** (`COMPARISON.md` §1.2).

This is not a maybe — it is already measured. So the proxy is **the
architecture, not a fallback**: the reference SentencePiece encoder lives in
the server, llama.cpp's broken tokenizer is bypassed, and every client gets
faithful tokenization for free. There is no "test the string path first" step;
we already know it produces garbage.

The divergence is tokenizer-type × vocab-size × language (UNIGRAM Viterbi
search × 4K small vocab × Basque agglutinative long words), **not the
architecture** — a transformer with this exact tokenizer would hit the same
issue. The clean fix is to put the correct encoder where the wrong one lives,
which is exactly what the proxy does. If llama.cpp ever fixes the divergence
upstream, delete the proxy and repoint clients at llama-server directly — zero
client-side changes.

### 4.4 What this gets us immediately

The proxy face works against the *current* AR-only model (prefix-only
completion, suffix ignored until FIM lands). This means the integration can
start today against the existing 74K checkpoint: build the `/v1/completions`
route, point Continue at it, dogfood prefix-only ghost-text, and record the
quality baseline that FIM (Phase 6) must beat — all before any GPU work.

---

## 5. Roadmap

| Phase | Goal | Depends on | Effort | Status |
|-------|------|-----------|--------|--------|
| **6e′** | Build the OpenAI-compatible face on the proxy (`/v1/completions` + `/v1/complete`); point Continue.dev at it; dogfood prefix-only against the 74K AR checkpoint; record the baseline FIM must beat | nothing | small — **first step** | ✅ Done (P1) |
| **6a** | Add `<PRE>/<SUF>/<MID>/<EOT>` tokens; resize embeddings; resume-from-74K smoke test | nothing (parallel with 6e′) | small | ✅ Done (P2) |
| **6b** | FIM data transform (char-level PSM/SPM) on Latxa v2; pretokenize FIM shards | 6a | medium | ✅ Done (P3) |
| **6c** | Continued-pretrain Phase 6: ~1–2B tokens, 50% FIM / 50% AR | 6b | GPU time | Next (P4) |
| **6d** | Basque FIM eval (random-span infilling on held-out sentences) | 6c | medium | Script ready (P5) |
| **6e** | Export FIM checkpoint → GGUF; add FIM template **server-side** to the proxy; dogfood in VS Code | 6d | small | P6 |
| **(research) R** | Test Mamba-2 + FUTO autocorrect format (`TODO.md` D1) | nothing | medium — *separate track, gates the mobile-IME decision* | Deferred |

**Critical path:** 6e′ ✅ → 6a ✅ → 6b ✅ → 6c (next) → 6d → 6e.

Phases 6a–6e are coupled: FIM tokens must exist before FIM data can be
tokenized, which must exist before Phase 6 training, which must finish before
the FIM eval/model is meaningful. **6e′ is the unambiguous first step** — it
builds the reusable Basque inference server (which every later phase and every
client depends on), de-risks the integration surface for free, and establishes
the dogfooding loop — all with zero model work. 6a can run in parallel since it
touches only the tokenizer + embeddings, not the server.

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
2. **Proxy SSE streaming translation.** llama-server's native `/completion`
   streams in its own SSE frame format; the OpenAI `/v1/completions` face must
   translate frames on the fly. Non-streaming is trivial; the streaming path
   is mechanical but unverified. Resolved as part of 6e′.
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

The comparison against the base HiTZ/Latxa-Llama-3.1-8B (§1.4) settles the
deployment architecture as a **two-tier split, fixed by hardware rather than
preference**:

- **Morpheus (Mamba-2, 91M, 64 MB) is the on-device tier.** It is the only
  model that runs on the edge — 40.7 tok/s on a 2017 consumer laptop CPU — but
  its quality ceiling is real: on essay and technical prose it gives *naive*
  predictions, drifting to high-frequency connective filler or unrelated
  statistical patterns instead of holding the semantic thread. Its sweet spot
  is **formulaic completion** (email openings/closings, fixed collocations,
  common administrative phrasing) and, given its parameter budget, **domain-
  specialized fine-tunes** where a narrow distribution raises the hit rate on
  the patterns it can actually learn. Mamba-2's O(1) decode and constant memory
  remain the right architecture for long-session, constant-latency edge use —
  but scoped to what a 91M model can genuinely predict, not general expository
  writing.

- **Kimu 2B and Latxa 8B (base) are the server-side tier.** Both are +9 CSR
  points better than morpheus (34.10% / 33.18% vs 24.83%), cross-domain *without*
  specialization (both stayed on-topic across email, essay, and technical
  prompts), but GPU-bound — Latxa collapses to 2.9 s/request (2.8 tok/s) and 6.6
  GB RAM on the same laptop, ~19× over the latency budget; Kimu 2B is 2× faster
  (~1.4 s, 5.6 tok/s, 9.6× over) but still not viable on CPU. Kimu 2B is the
  efficiency frontier (matches Latxa's CSR at 4× smaller size) and the leading
  candidate for **FIM continued pretraining** (§2) to close the infill gap. A
  practical advantage of both is that they ride on **standard LLMs** (Gemma-2 /
  Llama-3.1): the surrounding ecosystem — quantization, serving, editor plugins,
  eval harnesses — is mature and shared with mainstream work, lowering the
  engineering burden relative to the bespoke Mamba-2 toolchain.

The plan is therefore a continued-pretraining stage that adds FIM tokens and
trains on a 50/50 FIM+AR mix — applied to the Latxa 8B base on the GPU side —
evaluated with a new Basque infilling benchmark, and deployed through a
**reusable Basque inference server** (the proxy with an OpenAI-compatible
face) that centralizes the tokenization fix and FIM template, letting any
editor client (Continue.dev, Obsidian, Vim, CLI) connect with zero model-
specific code. Every step is evidence-grounded in the FIM literature and the
comparison findings; none requires an architecture change or new data
collection.
