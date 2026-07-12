# Copilot Research → Morpheus Action Items

**Date:** July 11, 2026
**Purpose:** Identify which insights from the GitHub Copilot architecture research are genuinely applicable to Morpheus's pipeline or evaluation, and which are not.

---

## Summary

Of the 10+ practices Copilot uses, **3 are genuinely actionable** for Morpheus, **2 are already done**, **2 validate existing decisions** (worth citing in the paper), and the rest are either cloud-specific (irrelevant to on-device) or code-specific (irrelevant to natural language).

---

## ✅ Actionable Improvements

### 1. Fill-in-the-Middle (FIM) Training — PIPELINE

**What Copilot does:** Trains the model with prefix + suffix + middle format, so the model can complete code at the cursor while respecting text after the cursor (closing braces, function signatures, etc.).

**Why it matters for Morpheus:** Our model currently does **prefix-only continuation** — it predicts what comes *after* the cursor, with zero awareness of what comes *before the end of the document*. In a real text editor, the cursor is often in the **middle** of a paragraph or document. The model's continuation should be contextually appropriate given both the text before AND after the cursor.

**Current state:** Training is pure next-token prediction (`train.py` lines 388-400: standard causal LM, `x` → `y` shift, `F.cross_entropy`). No FIM anywhere. The greedy demo (`index-greedy.html` line 438) even explicitly disables suggestions when the cursor is not at the end: `if (editor.selectionStart !== editor.value.length) { clearGhost(); return; }`.

**Recommendation:** This is the single most impactful pipeline improvement. FIM can be added as a **fine-tuning stage** on the existing checkpoint (no retraining from scratch needed). The approach:
1. Take the trained 91M checkpoint as the starting point
2. Format a portion of training data as FIM: split sentences/documents at random positions, format as `<prefix> + <suffix> → predict <middle>`
3. Fine-tune for a short period (1-2 epochs) with FIM-formatted data mixed with standard continuation data
4. The model learns to condition on both prefix and suffix

**Effort:** Moderate. Requires: FIM data formatting script, fine-tuning training loop (we don't have one yet — noted in §7.3.4 of the paper), re-evaluation. ~2-3 days of work.

**Paper impact:** This is a significant future-work item. It would make Morpheus more practical for real text editor integration (cursor in the middle of a document).

**Caveat:** FIM is more critical for code (structured, closing braces) than for prose. For natural language, prefix-only continuation is less wrong — but in a real editor, mid-paragraph completion that ignores the rest of the paragraph is still suboptimal.

---

### 2. Accepted-and-Retained Characters Metric — EVALUATION

**What Copilot does:** Their primary production metric is not just "did the user accept the suggestion?" but "did the accepted text **stay** in the final code?" This captures the retention dimension — a suggestion accepted but later deleted provided no real value.

**Why it matters for Morpheus:** Our completion logging (`/api/log`) currently logs `accept` events with the accepted text, probability, candidates, and context. But it does NOT track whether the accepted text was later **deleted or modified**. A user might accept a suggestion, realize it's wrong, and delete it — our current logging counts this as a successful acceptance.

**Current state:** 
- `predictive-keyboard.html` line 616: logs `event: 'accept'` with text, prob, candidates, context
- No `event: 'delete'` or `event: 'edit'` or retention tracking exists
- The typing simulation (`simulate_typing.py`) doesn't model deletion either

**Recommendation:** Extend completion logging to track retention:
1. Add a `event: 'delete'` log when previously-accepted text is removed (detect via comparing editor content before/after, tracking the last accepted text span)
2. Add a `event: 'edit'` log when previously-accepted text is modified
3. Add a `retained: true/false` field that can be computed post-hoc by correlating accept events with subsequent delete/edit events within a time window
4. In the replay script (`replay_completions.py`), compute a **retention rate** = (accepted text that was not deleted within N seconds) / (total accepted)

**Effort:** Small. ~half a day of frontend + replay script changes. The logging infrastructure already exists.

**Paper impact:** This is a more mature version of completion logging (§5.5.6). It directly addresses the acceptance rate trap (which we already identified as the CSR paradox). Can be mentioned as an improvement to the evaluation methodology.

**Caveat:** Retention tracking requires real user sessions. Our simulation doesn't model deletion, so this metric can only be populated from real usage logs, not from simulation. This means it's a deployment-time metric, not a pre-deployment eval.

---

### 3. Time-to-First-Token (TTFT) Latency Metric — EVALUATION / DEMO

**What Copilot does:** Tracks time-to-first-token separately from total latency. This matters because streaming means the user sees the first token quickly even if the full completion takes longer.

**Why it matters for Morpheus:** We currently track only total `latency_ms` (time from request to full response). For a multi-token ghost text suggestion, the user cares about when they **see** the suggestion, not when the full generation completes. If we ever add streaming (see below), TTFT becomes the primary latency metric.

**Current state:** `server.py` measures total latency only (`t0 = time.perf_counter()` at start, `latency = (time.perf_counter() - t0) * 1000` at end). No TTFT breakdown.

**Recommendation:** Add TTFT measurement:
1. In `_call_llama()` and `_keyboard_candidates()`, record the timestamp when the first token is generated (before the full completion finishes)
2. Return both `latency_ms` (total) and `ttft_ms` (time to first token) in the API response
3. Display both in the demo UI (the greedy demo already shows `latency_ms`)
4. Log TTFT in completion logs for offline analysis

**Effort:** Trivial. ~1 hour. Just add a second timestamp.

**Paper impact:** Minor, but makes the latency analysis more precise. TTFT is the metric that matters for user perception.

---

## ✅ Already Done (No Action Needed)

### 4. LLM-as-a-Judge Evaluation
Copilot uses an independent LLM to score completions on Quality, Relevance, Helpfulness. We already researched this extensively in `docs/llm-judge-eval-research.md` and it's in our next steps (item #2 in Next Steps). Our design is actually more rigorous than Copilot's: we plan reference-guided pairwise with position swap + human validation against our existing 30 A/B judgments. **No change needed — just implement it.**

### 5. Streaming
Copilot streams tokens to reduce perceived latency. Our `llama-server` backend already supports streaming (`"stream": False` is currently set in `server.py` line 389). We currently disable it and wait for the full response. Enabling streaming is a one-line change (`"stream": True`), but the frontend would need to handle progressive ghost text display. **Low priority** — our on-device latency is already <50ms, so streaming provides less benefit than it does for Copilot's 200ms cloud round-trip.

---

## ✅ Validates Existing Decisions (Cite in Paper)

### 6. The Acceptance Rate Trap
Copilot explicitly learned that optimizing for acceptance rate alone is harmful: "a heavy focus on acceptance rates could lead to incorrectly favoring a high volume of simple and short suggestions." This is **independent validation** of our CSR Paradox finding (§6.14) and our conclusion that CSR is a fragile metric (§6.8). **Cite Copilot in the paper** as a production-scale confirmation that acceptance-rate-style metrics are structurally biased.

### 7. Language-Specific Expert Evaluation
Copilot: "we find language-specific evaluations lead to better outcomes along quality and style preferences." This validates our decision to use **expert-authored Basque evaluation prompts** and the finding that assistant-authored ad-hoc prompts scored 13.3% vs expert's 60.0% (4.5× gap). **Cite Copilot in the paper** as independent validation that language experts are essential for quality evaluation.

---

## ❌ Not Applicable (Cloud-Specific or Code-Specific)

### 8. HTTP/2 Multiplexing, Request Cancellation, Global Proxy Fleet
These exist *because* Copilot's model is remote. Morpheus runs on-device — there is no network latency to fight, no connections to multiplex, no requests to cancel over HTTP. Our entire value proposition is eliminating this infrastructure. **Not applicable.**

### 9. Execution-Based Benchmark (Unit Tests)
Copilot tests if generated code compiles and passes tests. For natural language, there is no equivalent of "compilation." We could check grammatical correctness, but we've explicitly noted we are not Basque linguistics authorities. **Not applicable.**

### 10. Reinforcement Learning for Completions
Copilot uses RL (quality, relevance, helpfulness rewards). This is a significant training pipeline addition. For a 91M model and a research project, RL is feasible but complex. **Note as long-term future work** (§7.4), but not immediately actionable.

### 11. Mid-Training on Domain Corpus
Copilot does mid-training on a code corpus before fine-tuning. Morpheus is already trained from scratch on domain-specific (Basque) data. Mid-training as a concept is about adapting a general model to a domain — we already ARE the domain model. **Not applicable.**

---

## Prioritized Action List

| Priority | Item | Type | Effort | Paper Section |
|----------|------|------|--------|---------------|
| **1** | FIM fine-tuning | Pipeline | 2-3 days | §7 (future work) |
| **2** | Accepted-and-retained characters logging | Evaluation | 0.5 days | §5.5.6 improvement |
| **3** | Time-to-first-token metric | Evaluation/Demo | 1 hour | §6 latency analysis |
| **4** | LLM-as-a-Judge (already planned) | Evaluation | 1-2 days | §6 new metric |
| **5** | Cite Copilot as validation (acceptance trap, expert eval) | Paper | 30 min | §6.8, §6.14 |
| **6** | Streaming (low priority, on-device already fast) | Demo | 2 hours | §5.4 |
| **7** | RL for completions (long-term) | Pipeline | 1-2 weeks | §7.4 |

**Recommendation:** Do #5 immediately (30 min, strengthens the paper). Do #3 next (1 hour, easy win). Do #2 when we next touch the demo (half day, improves eval maturity). Do #1 and #4 as the next major work items. #6 and #7 are lower priority.
