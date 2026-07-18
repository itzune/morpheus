# Phase 6 v2 training complete — FIM quality improved but `<EOT>` emission unreliable

Training completed and we have eval numbers. The tokenization fix + packed
dataset worked (char accuracy +40%, keystrokes penalty halved), but we've hit a
wall with `<EOT>` emission reliability. Looking for your guidance on the highest-
leverage next step before we commit more GPU time.

---

## What changed since your GO

Per your review, we launched Phase 6 v2 with:
1. **Token-level FIM splitting** (BigCode/StarCoder) — fixed string-level
   fragmentation (words now keep `▁` markers after FIM tokens, ~30% fewer tokens).
2. **Packed dataset** (`PackedFimDataset`) — greedily packs whole examples into
   1025-token windows, 0 broken FIM structures, 3.1% padding. Your #1 concern.
3. **FIM perplexity monitored** every 500 steps (your Q4, already wired up).

Setup unchanged from what you reviewed: 91M Mamba-2, 500M tokens, LR 1e-3 → 1e-5
cosine, 50% FIM / 50% AR, loss on all tokens (FIM-for-free, `ignore_index=0`).

---

## Training results (completed)

| Metric | Value |
|--------|-------|
| Duration | 2.8 hours, 500M tokens, 3,814 steps |
| Best AR valid loss | 1.9965 → **ppl 7.4** (unchanged from 74K baseline) |
| FIM valid loss | 2.0770 → **ppl 8.0** |
| FIM ppl trajectory | 8.5 (step 500) → 8.2 (step 1000) → 8.1 (step 1500) → 8.0 (step 2000) → **plateau at 8.0** |

**FIM-for-free confirmed again** — AR perplexity did not regress (7.4 → 7.4).
Next-word quality also held (Top1 ~54%, Top5 ~87%).

The FIM ppl curve flattened around step 2000 (~260M tokens). It was still
*technically* descending at step 3500 (2.0799 → 2.0770) but the delta is tiny.

---

## Eval results (200 examples, full FIM eval)

| Metric | v1 (string-level, broken) | v2 (token-level + packed) | Δ |
|--------|---------------------------|---------------------------|---|
| Exact-match | 2.5% | 2.7% | ≈ same |
| Char accuracy | 22.3% | **31.2%** | **+40%** |
| Keystrokes saved | -58.8% | **-26.5%** | **halved** |

The tokenization fix + packing produced real improvement: coherent Basque
output instead of repetitive garbage, and the penalty roughly halved.

---

## The core remaining issue: `<EOT>` emission is unreliable

This is the blocker. We tested with 4 fill-in-the-blank style examples (strong
contextual constraint, single correct answer):

```
1. Ni atzo amonaren etxera joan ___ bazkaltzera     (target: nintzen)
2. Zein da zure ___? Ni Xabi naiz                    (target: izena)
3. Urrutiko ___ hamalau                              (target: intxaurrak)
4. Zuek animatuko ___ etxera ostiralean bazkaltzera? (target: zarete)
```

| # | Target | Model output | `<EOT>`? | Verdict |
|---|--------|--------------|----------|---------|
| 1 | `nintzen` | **`nintzen`**, eta han, nire amona eta nire amona... | ❌ | Right first token, didn't stop → degenerated |
| 2 | `izena` | `laguna da.` | ✅ | Wrong but clean/coherent |
| 3 | `intxaurrak` | `10.1.1.1.1.1...` | ❌ | Degenerate |
| 4 | `zarete` | `gara, eta animatu!` | ✅ | Wrong, grammatically close |

**The pattern is clear:**
- When `<EOT>` fires → output is short, clean, coherent.
- When `<EOT>` doesn't fire → the model keeps generating and *eventually
  degenerates into repetition* (e.g. "eta nire amona, eta nire amona...").

**Example 1 is the most revealing:** the model generated `nintzen` (the exact
correct answer) as its very first token — so it *learned the infill capability*.
But it couldn't reliably signal "I'm done" and ran off into repetition.

This is consistent across the 200-example eval: the model knows the format and
often knows the infill, but the stop behavior is unreliable. Negative keystrokes-
saved comes almost entirely from cases where `<EOT>` didn't fire and the model
generated 64 tokens (our cap) of runaway text.

---

## Our hypotheses (ranked)

1. **Insufficient training signal for `<EOT>`** (most likely): At 50% FIM / 50%
   AR, the `<EOT>` token appears in only ~50% of examples and is 1 of ~40 tokens
   per FIM example. The loss signal for "emit `<EOT>` here" is weak relative to
   the strong AR continuation signal. The model defaults to continuing.

2. **FIM ppl plateau at 8.0**: The curve flattened early (~step 2000, 260M
   tokens). More tokens at the same 50/50 ratio may not help if the bottleneck is
   signal strength, not data volume. But it's also possible 500M was just barely
   enough and the stop behavior is the last thing to crystallize.

3. **Model size (91M params)**: A small model may lack capacity to reliably learn
   the meta-behavior of "when to stop." We can't easily test this without a
   larger model.

4. **No explicit stop-token loss weighting**: Currently `<EOT>` has the same loss
   weight as any other token. If hypothesis #1 is right, upweighting `<EOT>` in
   the loss could directly address the problem.

---

## Options we're considering (need your call)

| Option | Cost | Hypothesis it tests |
|--------|------|---------------------|
| A. Continue training to 1B tokens (same 50/50 ratio) | ~5.6h more | Is it a data-volume problem? |
| B. Raise FIM ratio to 70/30, train 500M more | ~2.8h | Is it a signal-density problem? |
| C. Add `<EOT>` loss weighting (e.g. 5–10×), train 500M more | ~2.8h + code change | Directly targets the stop-token signal |
| D. Combine B + C (70/30 ratio + EOT weighting) | ~2.8h + code change | Max signal for stop behavior |
| E. Ship AR-only now, defer FIM | 0 | AR quality is solid; FIM is experimental |

We can do A–D without a data rebuild (we'd just need a new config / a small
`train.py` change for C/D). The checkpoint at `best.pt` (step 3500) is our
resume point.

---

## Specific questions

1. **Given the FIM ppl plateaued at 8.0 by step 2000 but the `<EOT>` problem
   persists, do you read this as a signal-strength issue (hypothesis #1) or a
   data-volume issue (hypothesis #2)?** I.e. is more data at 50/50 likely to fix
   the stop behavior, or do we need to change the training signal?

2. **Is `<EOT>` loss weighting a standard/recommended technique for FIM training?**
   We haven't seen it called out explicitly in the FIM literature (Bavarian et al.
   use plain cross-entropy on all tokens). If you'd recommend it, what weight
   range is sane (2×? 5×? 10×?) and are there failure modes to watch for (e.g.
   premature stopping)?

3. **If we adjust the FIM/AR ratio, is 70/30 a reasonable next step, or would you
   push harder (e.g. 90/10)?** Concern with going too high: AR regression
   (currently FIM-for-free is holding at ppl 7.4).

4. **Is example 1 (correct first token `nintzen`, then runaway repetition) a
   known failure mode with a name in the FIM/infill literature?** It looks
   distinct from "model doesn't know the answer" — the model *does* know, it just
   can't stop. Any targeted fixes for this specific pattern?

5. **Any red flags in our setup that could be causing the `<EOT>` unreliability
   that we haven't considered?** E.g. the way `<EOT>` is positioned in the data
   (always followed by `</s>`), the tied embeddings, the Mamba recurrence not
   "seeing" a clear end-of-middle signal, etc.

Happy to share the eval JSON, training log, or the `PackedFimDataset` code if
any of that would help your assessment.
