# Verb-Agreement Fine-Tuning Research for Basque Autocomplete

> **Purpose:** define a concrete research and engineering path to fix a class of model failures — **Basque auxiliary verb agreement** — that no amount of continued pretraining, more data, or a larger model will resolve.
>
> **Focus:** **Strategy B — supervised fine-tuning on gap-fill exercises** — as the primary fix, with Strategies A (FST bypass) and C (hybrid re-ranker) as complements.
>
> **Date:** 2026-07-13

---

## Executive Summary

Morpheus produces fluent Basque completions and correctly resolves the **easy** two-way ergative alignment contrast (`dugu` after a transitive frame, `gara` after an intransitive; §6.11). But it **fails on three-way dative agreement** — the *nor-nori-nork* (absolutive–dative–ergative) auxiliary paradigm that encodes subject, indirect object, and direct object simultaneously. A representative failure:

> `Nik hari eskuminak eman` → model proposes **`dizkit`**
> correct: **`dizkiot`** (present) / **`nizkion`** (past) / etc.

The model confused a form meaning *"he/she has them to me"* (`dizkit`) with one meaning *"I have them to him/her"* (`dizkiot`). This is **not** a fluency failure and **not** the CSR paradox (§6.12). It is a **grammatical-feature mapping failure** inside the polysynthetic auxiliary — exactly the case the write-up flags as the hardest part of Basque verbal morphology (§4.4.4.1, "The Verbmorph Gap") and the one §6.9 admits "emerges late and requires more context than bare-root prompts provide."

### Why this matters

1. **It is a real product failure, not a benchmark artifact.** In free Basque prose, auxiliaries appear in nearly every finite clause. Getting `dizkiot`/`dizkit`-class confusions wrong degrades every multi-sentence completion.
2. **It is fixable.** Unlike the CSR paradox (which is structural to Basque word length), agreement mapping *can* be taught into the weights with the right supervised signal.
3. **The training data already exists, at scale, with gold answers and automatic correction.** Basque language-learning infrastructure (HABE/ikasbil, EEP exams, aditzak.eus) is built around exactly the "bete hutsak / aditza jokatu" (fill-the-blank / conjugate-the-verb) exercise that isolates this skill.
4. **The current training infrastructure cannot do this yet.** `train.py` supports resume-from-checkpoint (continuing the same run) but no fine-tuning-specific features: no separate fine-tune dataset, no layer freezing, no differential learning rates, no parameter-efficient methods (LoRA/PEFT). This is confirmed in §7.3.4 and verified in the source.

### Bottom line

The highest-leverage research direction for Morpheus after tokenization, evaluation, and corpus quality is **a supervised fine-tuning stage on Basque verb-agreement gap-fill exercises**, delivered as a LoRA (or full) adapter on the 91M base model. This is the only strategy that (a) fixes the failure in the model weights, (b) generalizes to free-text autocomplete (not just the exercise format), and (c) has a ready-made, gold-labeled data source. An Apertium-based FST re-ranker (Strategy C) can complement it at inference time, but cannot replace it.

---

## 1. The Problem: Auxiliary Agreement Failure

### 1.1 The failure case

The motivating example, provided from live model behavior:

| Input context | Model output | Correct (present) | Correct (past) |
|---|---|---|---|
| `Nik hari eskuminak eman ...` | `dizkit` | `dizkiot` | `nizkion` |

Gloss: *Nik* (I, ergative) *hari* (to him/her, dative) *eskuminak* (object, plural absolutive) *eman* (given). The finite auxiliary must agree with all three arguments. The model produced the auxiliary for *"he/she has them **to me**"* instead of *"**I** have them **to him/her**."*

### 1.2 Morphological decomposition (why this is hard)

Under the standard analysis (Hualde & Ortiz de Urbina, 2003), the *nor-nori-nork* present auxiliary with a 3rd-person plural absolutive decomposes as:

```
dizkiot = d-    + -zki-  + -o-   + -t
          present  3pl.abs   3sg.dat  1sg.erg

dizkit  = d-    + -zki-  + -t            (+ zero-marked 3sg.erg)
          present  3pl.abs   1sg.dat
```

The critical ambiguity: the **`-t` morpheme encodes "1st person" but its grammatical role (ergative vs. dative) depends on whether the `-o-` (3sg dative) slot is filled.** The model dropped `-o-` and reinterpreted `-t` as 1sg dative rather than 1sg ergative.

This is the **portmanteau problem**: a single surface morpheme carries fused person+role information, and its interpretation is positionally and paradigmatically conditioned. It cannot be solved by memorizing surface forms — there are hundreds of such combinations, and productive Basque generates ones unseen in training.

### 1.3 This is NOT the CSR paradox

These are commonly conflated but are **different failure modes with different fixes**:

| | CSR paradox (§6.12) | Agreement failure (this doc) |
|---|---|---|
| **Root cause** | Agglutinative words are long; user must type many chars before suggestion | Model maps the wrong person/role combination to the auxiliary morpheme sequence |
| **Model quality** | Model is already *good* (79.5% Top-3, best of all languages) | Model is *wrong* on a grammatical category |
| **Fixable by training?** | ❌ No — word length is structural | ✅ Yes — agreement is learnable |
| **Fixable by FST?** | ⚠️ Only as a prefix-pruning re-ranker | ✅ Deterministically, when arguments are explicit |
| **Affects free text?** | Indirectly (perceived keystrokes saved) | ✅ Directly (every finite clause) |

The CSR paradox is a *metric* problem masquerading as a model problem. The agreement failure is a *genuine model* problem. Confusing them leads to the wrong fix: more pretraining data will not help agreement (the model has seen `dizkiot` thousands of times — it just hasn't learned the combinatorial rule), and FST morphology will not help CSR (Basque words stay long regardless).

---

## 2. Why the Model Fails (Grounded in the Write-Up)

### 2.1 The tokens are available; the grammar is not learned

§4.4.4.1 shows that at the 4K vocabulary, the auxiliary morphemes **are** independently accessible tokens:

| Verb form | Gloss | 4K tokenization |
|---|---|---|
| `dizkizut` | *I have them to you* | `▁di` `zki` `zu` `t` |
| `dakizkioke` | *he can know them to him* | `▁da` `ki` `zki` `o` `ke` |
| `zitzaizkidan` | *they were to me* | `▁zitzai` `zki` `dan` |

So the failure is **not a tokenization failure**. The model *can* represent `zki`, `o`, `t` as separate tokens. What it has not learned is the **combinatorial mapping**: which (erg person, dat person, abs number, tense) tuples license which morpheme *sequence and role-assignment*. This is a learning problem, not a representation problem.

### 2.2 The model has only been validated on the easy case

§6.11 claims the model "resolves ergative alignment contrasts correctly (`dugu` after a transitive frame, `gara` after an intransitive frame)." But this is the **two-way, first-person-plural, present-tense** case — the single easiest cell of the paradigm (1pl subject, no dative, present). It tells us nothing about:

- 3-way agreement (with dative argument) — the `dizkiot` class
- Cross-person combinations (1sg erg + 3sg dat, 2sg erg + 1sg dat, etc.)
- Past tense and other paradigms (`nizkion`, etc.)

### 2.3 The paradigm test already admits the weakness

§6.9 reports a paradigm test (84 tests, bare-root prompts) where "the absolutive case (-a) is well-learned (83% Hit@5), the ergative (-ak) and inessive (-an) show partial learning, [and] most other cases remain below threshold." The explicit conclusion: **"morphology emerges late and requires more context than bare-root prompts provide."** The `dizkiot` failure is the verbal analogue of this — the hardest cell, learned last, and not yet learned.

### 2.4 Why more pretraining will not fix it

The base model was trained on ~10B tokens of Basque prose (§6.7), which contains abundant auxiliary tokens. The problem is **signal density**: in free prose, the (context → auxiliary) mapping is diluted across millions of unrelated language-modeling targets. The agreement rule is a *tiny fraction* of the loss signal and competes with fluency, collocation, and world-knowledge objectives. Pretraining optimizes for the average next-token; agreement is a sparse, high-precision sub-skill that gets under-served. **This is precisely what supervised fine-tuning is designed to fix**: concentrate the loss signal on exactly the skill you want to teach.

---

## 3. Three Strategies

| Strategy | Mechanism | Fixes gap-fill? | Fixes free-text autocomplete? | Improves model weights? | Build effort |
|---|---|:---:|:---:|:---:|---|
| **A. FST bypass** (Apertium analyze → generate) | Deterministic morphological generation from analyzed arguments | ✅ near-100% | ⚠️ only when arguments are explicit & adjacent | ❌ bypasses model | Medium |
| **B. Supervised fine-tuning on gap-fill** ⭐ | Teach agreement mapping into the weights via gold (context, blank, form) triples | ✅ | ✅ | ✅ | Medium-High |
| **C. Hybrid FST re-ranker** (at inference) | FST enumerates valid paradigm; LM picks contextually | ✅ | ✅ | ❌ (inference-time) | Medium |

**Strategy B is the focus of this document** because it is the only one that improves the model itself and therefore the only one that fixes free-text autocomplete (where the product lives). Strategies A and C are valuable complements but cannot substitute — see §5 and §6.

---

## 4. Strategy B (Deep Dive): Supervised Fine-Tuning on Gap-Fill Exercises

### 4.1 Why gap-fill exercises are ideal training data

Basque language-learning pedagogy is organized around a single exercise type — **"bete hutsak" / "aditza jokatu"** (fill the blank / conjugate the verb) — that is almost perfectly shaped as a supervised signal for this exact weakness:

1. **It isolates the target skill.** Each item presents a sentence with the arguments *and* a blank where the finite auxiliary goes. There is no competing objective — the only thing being tested is agreement mapping.
2. **Gold answers are baked in.** Because these are exam/exercise items with automatic correction, the correct surface form is already encoded. No labeling required.
3. **It covers the full paradigm systematically.** Graded exercises (A1–C2) deliberately walk through every person/number/tense combination, including the hard 3-way dative cells the model gets wrong.
4. **It is abundant.** Decades of HABE exams, euskaltegi materials, and online exercise platforms exist in this exact format.
5. **It is distribution-aligned with the failure.** The model fails precisely on the constructions these exercises were designed to teach. There is no train/eval distribution mismatch on the target skill.

In short: the Basque-language-teaching community has already built the dataset we need, for a different purpose, and packaged it with gold labels.

### 4.2 Data sources

| Source | Format | Volume (est.) | Gold labels? | License/Access |
|---|---|---|---|---|
| **ikasbil.eus (HABE)** — "Gramatika gaitasuna lantzen" | Moodle quiz, automatic correction, A1–C2, dedicated *Aditza* sections (Indikatiboa, Baldintza, Ahalera, Subjuntiboa, Agintera, Aditz bereziak); exercise types include "hutsuneak bete," "esaldiak moldatu" | Large (hundreds of exercise sets) | ✅ (auto-corrected) | Public web; Moodle extraction needed |
| **ikasbil Azterketa-ereduak** (HABE exam models, B1/B2) | PDF + online, writing/grammar sections | Medium | ✅ (model answers) | Public |
| **azterketak.eus (EEP — Euskararen Erakunde Publikoa)** | Public proficiency exams | Medium | ✅ | Public |
| **aditzak.eus** | Tool that conjugates the Basque auxiliary (*"Euskal aditz laguntzailea jokatzeko tresna"*) | Effectively a full paradigm generator | ✅ (generative) | Public web; can synthesize cells on demand |
| **Urbizu, Zulaika & Saralegi (2024)** — *"How Well Can BERT Learn the Grammar of an Agglutinative and Flexible-Order Language? The Case of Basque"* (LREC-COLING 2024) | Grammar-probing benchmark for Basque BERT | Research set | ✅ | Published; **must read before building** — may already cover part of the agreement paradigm and save redundant work |

**Action item before any collection:** read Urbizu et al. (2024) in full. It is the only existing Basque grammar-probing benchmark we found, and it may already provide (a) a reusable probe dataset, (b) a methodology for constructing minimal pairs, and (c) baseline numbers on BERT-class models that contextualize Morpheus's results. Building on it is both more rigorous and less work than starting from scratch.

### 4.3 Dataset construction

**Target schema** — each item a JSON record:

```json
{
  "id": "habe-b2-0042",
  "source": "ikasbil/gramatika-gaitasuna",
  "level": "B2",
  "context": "Nik hari eskuminak eman [...] ",
  "blank_position": "end",
  "arguments": {
    "erg": {"person": "1", "number": "sg"},
    "dat": {"person": "3", "number": "sg"},
    "abs": {"person": "3", "number": "pl"}
  },
  "tense": "present",
  "mood": "indicative",
  "gold_form": "dizkiot",
  "alternative_correct": ["nizkion"],
  "full_sentence": "Nik hari eskuminak eman dizkiot."
}
```

The `arguments`, `tense`, and `mood` fields are the **supervision signal that free prose lacks**. They can be obtained three ways, in increasing order of effort:

1. **Extraction from auto-corrected exercises** (ikasbil Moodle) — the platform already knows the intended agreement features because it grades against them. Extraction = parse Moodle quiz XML/JSON, map each item to the schema. This is the bulk path.
2. **FST-assisted annotation** — run Apertium-eu's analyzer over the context to extract argument features automatically (§5.1). This lets us *generate* labeled items from any Basque text containing a finite auxiliary, not just exercise items — scaling beyond the exercise corpus.
3. **Synthetic generation via aditzak.eus / Apertium generator** — enumerate paradigm cells programmatically and wrap each in a minimal context. Highest coverage of rare cells; lowest ecological validity. Use to *balance* the dataset, not as the primary source.

**Recommended mix:** ~60% extracted exercises (ecological), ~30% FST-annotated prose (scale + real contexts), ~10% synthetic (paradigm coverage for rare cells). This balances realism against paradigm completeness and avoids over-fitting to exercise-style sentences.

**Target size:** 5,000–20,000 labeled items. The agreement paradigm has on the order of a few hundred cells (persons × numbers × tenses × moods × transitivity); a few thousand diverse examples per difficult cell is ample for a 91M model. This is small by LM standards — the point is signal density, not scale.

### 4.4 Task formulation

Two formulations, with different tradeoffs. **Recommend running both and comparing.**

**Formulation 1: Causal-LM completion (preferred starting point).**
Format each item as a standard completion the model already knows how to do:

```
Nik hari eskuminak eman dizkiot.
```

Mask the auxiliary during loss computation (or simply include it in the sequence and let standard next-token loss apply, optionally upweighted). This requires **no architectural change** — it reuses the exact inference path (`/api/autocomplete/greedy`) and the exact token-ID-prompting pipeline that the write-up (§5.4) shows is quality-critical. The model learns "given these arguments and this tense, produce this auxiliary" as a next-token skill, which transfers directly to free-text autocomplete.

**Formulation 2: Masked/infill objective.**
Present `Nik hari eskuminak eman ___ .` and train the model to fill the blank. This is closer to the exercise format but requires a (minor) training-loop change to handle the mask token and a corresponding inference change. It is more faithful to the gap-fill task but *less* aligned with the causal autocomplete product. Reserve for a second experiment.

**Why prefer Formulation 1:** the product is a causal LM doing next-token prediction. Fine-tuning in the same paradigm maximizes transfer. The write-up's central deployment lesson is that token-ID-prompted causal completion is the faithful path (string prompting "gave ~4% vs ~28% with token IDs," §6.13); fine-tuning should respect that.

### 4.5 Fine-tuning approach and infrastructure gaps

**Current state of `train.py` (verified):**
- Supports: `--config`, `--batch-size`, `--seq-len`, `--resume` (loads optimizer + scheduler + step, **continues the same run**)
- Cosine LR schedule (`learning_rate: 2e-3`, `min_lr: 1e-5`)
- Checkpoint save/load (atomic)
- **Does NOT support:** a separate fine-tune dataset, layer freezing (`requires_grad` toggling), differential learning rates, parameter-efficient methods (LoRA/PEFT), or any notion of "adapt to a new distribution"

`--resume` is the closest existing mechanism, but it is **not fine-tuning**: it restores the optimizer state and step counter and continues the original objective on the original data distribution. Using it on a new dataset would (a) keep the pretraining LR schedule (wrong for fine-tuning), (b) train all layers uniformly (risks catastrophic forgetting), and (c) not allow a smaller, adapter-scale update.

**What needs to be built (in priority order):**

1. **Separate fine-tune dataset loader.** A path in `dataset.py` for the gap-fill JSON schema (§4.3), independent of the pretraining corpus. Minimal: a new data source selectable via config.
2. **Lower learning rate + fresh schedule.** Fine-tuning typically needs 10–100× smaller LR than pretraining (e.g., `1e-4` to `1e-5`) with its own warmup. The existing cosine scheduler can be parameterized for this; the gap is a config preset, not new code.
3. **Layer freezing / selective unfreezing.** For a 91M model, start by freezing embeddings and early layers, training only the upper Mamba-2 blocks. This protects general fluency while updating the agreement-relevant representations.
4. **LoRA / PEFT adapter (recommended).** This is the cleanest path for an on-device model:
   - Train only low-rank adapter matrices (~0.1–1% of parameters)
   - Distribute the adapter as a small file (tens of KB to low MB) separate from the base GGUF
   - Hot-swap at runtime via the existing `/api/model/reload` endpoint (§5.3) — no re-quantization of the base model
   - Avoids catastrophic forgetting by construction (base weights untouched)
   - Aligns with the §7.3.5 on-device personalization roadmap, where the *same* LoRA path enables per-user adaptation from completion logs
   - **Caveat (must verify):** PEFT/LoRA support for Mamba-2 (SSM) architectures is less mature than for Transformers. Confirm `peft` library compatibility with the Mamba-2 block before committing. If unsupported, full fine-tuning at 91M is entirely feasible on a single GPU and is the fallback.
5. **Eval-gated stopping.** Stop fine-tuning when the held-out agreement eval (§7) stops improving — do not use training loss, which will keep dropping while the model over-fits to exercise style.

### 4.6 Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Catastrophic forgetting** — fine-tuning degrades general fluency/PPL | High (full FT) / Low (LoRA) | Prefer LoRA; if full FT, freeze early layers + low LR; monitor held-out PPL on `eval/real_corpus/` (14 files) as a forgetting guardrail |
| **Exercise-style over-fitting** — model learns "exam sentence" artifacts, not agreement | Medium | Mix 30% FST-annotated real prose (§4.3); hold out a free-prose agreement eval, not just exercise items |
| **Distribution shift at inference** — exercise arguments are explicit & adjacent; real arguments are implied/distant | Medium | Formulation 1 (causal completion) over masked-infill; include items with non-adjacent arguments in the dataset |
| **Paradigm imbalance** — rare cells under-represented even in exercises | Medium | Synthetic generation (aditzak.eus / Apertium) to balance rare cells; report per-cell accuracy, not just aggregate |
| **Metric inversion** — agreement fine-tuning could move autocomplete metrics opposite to PPL (as §6.13 documents) | Medium | Do not optimize on CSR/Top-1; optimize on the held-out agreement eval + guardrail PPL. The write-up already argues PPL is the only reliable checkpoint-ranking metric. |
| **GGUF deployment drift** — fine-tuned weights may not quantize identically | Low | Re-run the quant comparison pipeline (`eval/benchmarks/`) on the fine-tuned model; the existing Q4/Q5 sim harness catches this |

### 4.7 Why this is the right scale for the 91M model

§7.3.4 notes the 91M parameter scale is "small enough that full fine-tuning on a single GPU is feasible." This is a strategic advantage: at 91M, we do not face the compute or distribution constraints that make fine-tuning 8B+ models expensive. We can afford to run the full experimental matrix (LoRA vs. full FT, frozen vs. unfrozen, several LR values, both task formulations) cheaply, and to ship multiple domain/agreement adapters as separate small files. The constraint that forces large models toward LoRA is, for us, a choice rather than a necessity — though LoRA remains preferable for the hot-swap and forgetting-resistance properties.

---

## 5. Strategy A: FST Bypass (Apertium) — Complement, Not Replacement

### 5.1 What it can do

Apertium-eu is a **bidirectional** finite-state morphological analyzer/generator for Basque (§7.2). For the gap-fill task it enables a fully deterministic pipeline:

```
context (Nik hari eskuminak eman __)
  → Apertium analyze each NP → (1sg.erg, 3sg.dat, 3pl.abs)
  → rule: tense from "eman" + discourse → present
  → Apertium generate surface form → "dizkiot"
```

This would get **explicit-argument gap-fill near-100% correct**, because it is rule-based morphological generation — exactly what FSTs are built for. It also enables **FST-assisted annotation** of arbitrary prose (§4.3, method 2), which is what makes the Strategy B dataset scalable beyond the exercise corpus.

### 5.2 What it cannot do (hence not a replacement for B)

- **It does not change the model.** It is a bypass. Every free-text completion still goes through the un-improved LM.
- **It needs explicit, adjacent arguments.** In real autocomplete, arguments are often implied, distant, or not yet typed. The deterministic pipeline breaks when the FST cannot see all arguments.
- **It cannot choose among valid alternatives.** `dizkiot` (present) vs. `nizkion` (past) both satisfy the agreement features; choosing between them requires discourse context — an LM job.

**Verdict:** Strategy A is essential as a *dataset annotation tool* for Strategy B, and as the engine of the Strategy C re-ranker. As a standalone product fix, it is too narrow.

---

## 6. Strategy C: Hybrid FST Re-Ranker (Inference-Time)

At inference, constrain the LM's auxiliary predictions to the morphologically-valid paradigm:

```
typed prefix + context
  → LM proposes top-k candidate continuations
  → FST filters to candidates that are valid inflections for the context's arguments
  → LM (re-)ranks the survivors by full-context probability
```

This turns an open-ended generation problem ("guess `dizkiot` from `di`") into a ranking problem over a handful of valid candidates — drastically easier, and the mechanism by which FST morphology can also attack the CSR paradox's prefix-length problem (rare valid continuations surface earlier).

This is **incremental engineering on existing machinery**: the retokenization fallback (§5.4.2) already queries progressively shorter prefixes and filters by the typed prefix; the sticky-merge (§5.4.3) already maintains a candidate pool across renders. Adding an FST validity filter to that pool is a localized change to `demo/server.py`'s candidate pipeline.

**Verdict:** ship Strategy C as a product-level safety net for agreement, *after* Strategy B has done the weight-level fix. C without B means the model is still wrong whenever the FST can't see the arguments; B without C means occasional valid-but-contextually-wrong paradigm choices. Together they cover each other's gaps.

---

## 7. Evaluation: A Basque Auxiliary-Agreement Benchmark

There is currently **no standard Basque verb-agreement benchmark for autocomplete LMs** in the repo. `eval/benchmarks/` contains quant comparisons and typing simulations; `eval/ab_eval2/` has blinded A/B results; none isolate auxiliary agreement. Building this benchmark is a contribution in its own right and a prerequisite for measuring Strategy B.

### 7.1 Benchmark design

- **Held-out split** from the §4.3 dataset (never used in fine-tuning), stratified by:
  - transitivity (intransitive / transitive / ditransitive)
  - person combinations (especially the hard 3-way dative cells)
  - tense × mood
  - argument adjacency (explicit+adjacent vs. implied+distant)
- **Metrics:**
  - **Exact-match accuracy** per cell (did the model produce the gold auxiliary?)
  - **Paradigm-valid accuracy** (did it produce *a* valid inflection for the arguments, even if wrong tense/mood?) — measured via Apertium analysis of the output
  - **Top-k agreement accuracy** (is the correct form in the top-3?) — analogous to the §6.12 Top-3 metric, which is the honest one for agglutinative prediction
  - **Per-cell breakdown** — the whole point is to see the `dizkiot`-class cells specifically, not an aggregate that hides them
- **Baselines:**
  - Morpheus base (pre-fine-tuning) — establishes the failure rate this doc is about
  - HiTZ/gpt2-eus-euscrawl and Latxa (§6.6 baselines) — do larger Basque models get this right? If yes, the failure is scale/curriculum; if no, it is structural to how Basque is taught to LMs
  - Urbizu et al. (2024) BERT results, if their benchmark overlaps — external grounding

### 7.2 Guardrail metric

Continue reporting held-out PPL on `eval/real_corpus/` (14 files, the §6.6 set) throughout fine-tuning. **If agreement accuracy rises but PPL degrades beyond a threshold, stop** — this is the forgetting signal. The §6.13 metric-inversion finding means we must not trust a single metric; the pair (agreement-accuracy ↑, PPL stable) is the success criterion.

---

## 8. Research Agenda (Ordered)

1. **Read Urbizu et al. (2024) in full.** Determine whether a reusable agreement probe already exists. *(Blocks: avoids redundant dataset work.)*
2. **Build the benchmark (§7) and measure the base model.** Quantify the `dizkiot`-class failure rate per cell. We currently have one anecdotal example; we need the distribution. *(Blocks: cannot claim improvement without a baseline.)*
3. **Prototype the Apertium annotation pipeline (§5.1).** Even if Strategy A is not the product fix, FST-assisted annotation is required to scale the Strategy B dataset. Validate Apertium-eu's analysis accuracy on the target constructions.
4. **Confirm PEFT/LoRA compatibility with Mamba-2.** If supported, this is the preferred fine-tuning path. If not, fall back to full fine-tuning (feasible at 91M).
5. **Extend `train.py` with fine-tune features (§4.5):** separate dataset loader, lower-LR config preset, layer freezing, (optionally) LoRA.
6. **Construct the dataset (§4.3)** at the 5–20k scale, with the 60/30/10 exercise/prose/synthetic mix.
7. **Fine-tune and ablate:** LoRA vs. full, frozen vs. unfrozen, both task formulations. Select on the held-out agreement eval + PPL guardrail.
8. **Re-quantize and re-run `eval/benchmarks/`** on the fine-tuned model to verify no GGUF deployment drift.
9. **Optionally add Strategy C re-ranker** to `demo/server.py` as a product-level safety net.

---

## 9. Open Questions

- **Does the failure persist at scale?** If Latxa (8B+) or Kimu (9B) get `dizkiot`-class agreement right, the failure may be partly a capacity/curriculum issue, and the fine-tuning signal may need to be richer. If they also fail, it confirms the structural "Basque is hard to teach LMs by pretraining" thesis and makes the supervised fine-tuning case stronger. The §7 baselines answer this.
- **Is the portmanteau ambiguity (`-t` = 1sg.erg vs. 1sg.dat) the dominant failure, or are there other confusion classes?** Only the per-cell benchmark (§7.1) reveals this. Hypothesis: the cross-person dative cells (1↔3, 2↔3) dominate; same-person cells are easier.
- **How much does FST-assisted annotation cost in precision?** Apertium-eu is a shallow-transfer analyzer; its accuracy on complex verbal morphology needs validation before relying on it for dataset labels (§8 step 3).
- **Does agreement fine-tuning transfer to free text, or only to exercise-style sentences?** The distant-argument held-out split (§7.1) answers this. If transfer is poor, Strategy C (inference-time FST) becomes more important.
- **Can a single LoRA adapter cover both agreement correction and domain adaptation (§7.3.4), or do they interfere?** If they compose, one adapter mechanism serves multiple §7.3 roadmap items.

---

## 10. References

- **Hualde, J. I. & Ortiz de Urbina, J. (2003).** *A Grammar of Basque.* Mouton de Gruyter. — standard reference for the auxiliary morpheme decomposition in §1.2.
- **Urbizu, G., Zulaika, M., & Saralegi, X. (2024).** *How Well Can BERT Learn the Grammar of an Agglutinative and Flexible-Order Language? The Case of Basque.* LREC-COLING 2024. https://aclanthology.org/2024.lrec-main.731/ — existing Basque grammar-probing benchmark; required reading (§4.2, §8.1).
- **Contreras, A. (2026).** QuechuaTok — MorphAcc and the fertility/MorphAcc tradeoff (write-up §4.4).
- **Lane, A. et al. (2022).** Plains Cree word completion with FST segmentation — +15–30% KSR precedent for morphology-aware prediction (write-up §2.2).
- **Etxaniz, J. et al. (2024).** *Latxa: An Open Language Model and Evaluation Suite for Basque.* ACL 2024. https://aclanthology.org/2024.acl-long.799/ — baseline Basque LLM + eval suite (§7.2).
- **Apertium-eu** — finite-state Basque morphological analyzer/generator (write-up §7.2).
- **Internal:** Morpheus write-up §4.4.4 (verbal morphology / fertility paradox), §5.4 (inference engineering), §6.8–6.9 (ergative alignment + paradigm tests), §6.11–6.13 (cross-lingual, CSR paradox, metric inversion), §7.2–7.3.5 (Apertium + fine-tuning + on-device personalization roadmap). `train.py` (verified capabilities, §4.5). `eval/benchmarks/`, `eval/real_corpus/` (eval infrastructure, §7).
- **Data sources:** ikasbil.eus (HABE, "Gramatika gaitasuna lantzen"), azterketak.eus (EEP), aditzak.eus.
