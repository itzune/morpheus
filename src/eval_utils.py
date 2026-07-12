"""
Shared evaluation utilities for Morpheus autocomplete evaluation.

Single source of truth for CSR and MorphAcc metrics. Used by:
  - train.py  (inline W&B logging during training)
  - eval.py   (full CLI evaluation with JSON output)

This module ensures both entry points use identical metric definitions.
"""

import math

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────
#  Core prediction helpers
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_next_token_logits(model, context_ids, device):
    """Run model forward and return logits for the next token position."""
    ctx = torch.tensor([context_ids], device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(ctx)
    return out.logits[0, -1, :].float()


def get_top_k_predictions(model, sp, context_ids, device, k=10, logits=None):
    """Get top-K token predictions with true probabilities.

    Args:
        logits: Pre-computed logits for the next position. If None, runs
                a model forward pass. Used by paradigm eval to avoid
                redundant forward passes across multiple case suffixes.

    Returns list of dicts: id, piece, decoded, logprob, prob.
    """
    if logits is None:
        logits = get_next_token_logits(model, context_ids, device)

    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k)

    results = []
    for tid, prob in zip(topk_ids.tolist(), topk_probs.tolist()):
        piece = sp.id_to_piece(tid)
        decoded = piece.replace("\u2581", " ")
        results.append({
            "id": tid,
            "piece": piece,
            "decoded": decoded,
            "logprob": math.log(prob) if prob > 0 else float("-inf"),
            "prob": prob,
        })
    return results


# ──────────────────────────────────────────────────────────────────────
#  Strategy 1: Character Savings Rate (CSR)
# ──────────────────────────────────────────────────────────────────────
#
#  Simulates keystroke-by-keystroke typing of target_completion.
#
#  At each position, gets the model's greedy top-1 prediction. If the
#  prediction matches a prefix of length L of the remaining target, the
#  user accepts it with 1 keystroke (Tab) and advances L characters.
#  If no match, the user types one character (1 keystroke).
#
#  Per Trnka & McCoy (2008), acceptance costs a keystroke.
#
#  CSR = 1 - (keystrokes_needed / total_chars)
#
#  Example: prompt="Kaixo, zer", target="moduz?"
#    typed ""   → model predicts "moduz?" → full match → 1 keystroke (Tab) → CSR = 5/6 = 0.83
#    typed ""   → model predicts "mo"     → L=2 match  → 1 keystroke, advance 2, continue
#    typed "mo" → model predicts "duz?"   → full match → 1 keystroke (Tab) → total 2 keystrokes → CSR = 4/6 = 0.67

def evaluate_csr(model, sp, tests, device="cuda"):
    """Character Savings Rate evaluation.

    Returns list of per-test result dicts with csr, keystrokes_needed,
    keystrokes_saved, and a detailed predictions trace.
    """
    model.eval()
    results = []

    for t in tests:
        prompt = t["input"]
        target = t["target_completion"]
        category = t.get("category", "unknown")

        typed_so_far = ""
        keystrokes_typed = 0
        chars_auto_completed = 0
        predictions_made = []
        finished = False

        while len(typed_so_far) < len(target) and not finished:
            remaining_target = target[len(typed_so_far):]

            current_full_text = prompt + " " + typed_so_far if typed_so_far else prompt
            full_ids = sp.encode(current_full_text, out_type=int)

            if len(full_ids) > 1024:
                full_ids = full_ids[-1024:]

            top_k = get_top_k_predictions(model, sp, full_ids, device, k=1)
            if not top_k:
                keystrokes_typed += 1
                typed_so_far += target[len(typed_so_far)]
                predictions_made.append({
                    "pos": len(typed_so_far),
                    "typed": typed_so_far,
                    "pred": "(error)",
                    "remaining": remaining_target,
                    "match_len": 0,
                    "match": False,
                })
                continue

            pred_text = top_k[0]["decoded"].lstrip()

            if not pred_text:
                keystrokes_typed += 1
                typed_so_far += target[len(typed_so_far)]
                predictions_made.append({
                    "pos": len(typed_so_far),
                    "typed": typed_so_far,
                    "pred": "(empty)",
                    "remaining": remaining_target,
                    "match_len": 0,
                    "match": False,
                })
                continue

            # Find longest prefix match between prediction and remaining target
            max_len = min(len(pred_text), len(remaining_target))
            match_len = 0
            for j in range(1, max_len + 1):
                if pred_text[:j] == remaining_target[:j]:
                    match_len = j
                else:
                    break

            predictions_made.append({
                "pos": len(typed_so_far),
                "typed": typed_so_far,
                "pred": pred_text[:40],
                "remaining": remaining_target,
                "match_len": match_len,
                "match": match_len > 0,
            })

            if match_len == 0:
                # No match — user types one character
                keystrokes_typed += 1
                typed_so_far += target[len(typed_so_far)]
            else:
                # Accept prediction: 1 keystroke (Tab) for match_len chars
                keystrokes_typed += 1
                typed_so_far += pred_text[:match_len]
                chars_auto_completed += match_len
                if len(typed_so_far) >= len(target):
                    finished = True

            if len(typed_so_far) >= len(target):
                finished = True

        total_chars = len(target)
        csr = 1.0 - (keystrokes_typed / total_chars) if total_chars > 0 else 0.0
        chars_saved = total_chars - keystrokes_typed

        results.append({
            "prompt": prompt,
            "target": target,
            "category": category,
            "total_chars": total_chars,
            "keystrokes_needed": keystrokes_typed,
            "keystrokes_saved": chars_saved,
            "chars_auto_completed": chars_auto_completed,
            "csr": round(csr, 4),
            "predictions": predictions_made,
        })

    return results


# ──────────────────────────────────────────────────────────────────────
#  Strategy 2: Morpheme Boundary Accuracy (MorphAcc)
# ──────────────────────────────────────────────────────────────────────
#
#  For each test with known valid suffixes, checks if any valid suffix
#  appears in the top-K predictions (exact match or prefix match).
#
#  MorphAcc = proportion of tests where at least one valid suffix is
#             found in top-K.

def evaluate_morphacc(model, sp, tests, device="cuda", k=5):
    """Morpheme Boundary Accuracy evaluation.

    Returns list of result dicts with morphacc_hit, best_rank,
    boundary_prob_mass, and top_predictions.
    """
    model.eval()
    results = []

    for t in tests:
        prompt = t["input"]
        valid_suffixes = t.get("valid_suffixes", t.get("valid_continuations", []))
        category = t.get("category", "unknown")

        prompt_ids = sp.encode(prompt, out_type=int)
        if len(prompt_ids) > 1024:
            prompt_ids = prompt_ids[-1024:]

        top_k = get_top_k_predictions(model, sp, prompt_ids, device, k=k)
        if not top_k:
            results.append({
                "prompt": prompt,
                "category": category,
                "valid_suffixes": valid_suffixes,
                "morphacc_hit": False,
                "best_rank": None,
                "boundary_prob_mass": 0.0,
                "top_predictions": [],
                "error": "no_predictions",
            })
            continue

        best_rank = None
        boundary_prob_mass = 0.0
        top_preds = []

        for rank, pred in enumerate(top_k, start=1):
            decoded = pred["decoded"].lstrip()
            top_preds.append({"rank": rank, "decoded": decoded, "prob": pred["prob"]})

            for suffix in valid_suffixes:
                if decoded == suffix or decoded.startswith(suffix):
                    boundary_prob_mass += pred["prob"]
                    if best_rank is None:
                        best_rank = rank
                    break

        morphacc_hit = best_rank is not None and best_rank <= k

        results.append({
            "prompt": prompt,
            "category": category,
            "valid_suffixes": valid_suffixes,
            "morphacc_hit": morphacc_hit,
            "best_rank": best_rank,
            "boundary_prob_mass": round(boundary_prob_mass, 4),
            "top_predictions": top_preds,
        })

    return results


# ──────────────────────────────────────────────────────────────────────
#  Bootstrap confidence interval
# ──────────────────────────────────────────────────────────────────────
#
#  Resamples per-test CSR values with replacement to estimate the
#  sampling distribution of the mean. With n=30 the CI is wide (explains
#  why 30-sentence eval can't rank checkpoints); with n=300+ it is tight.
#
#  No number should be reported without a CI hereafter.

def bootstrap_mean_ci(values, n_bootstrap=1000, confidence=0.95, seed=42):
    """Bootstrap confidence interval for the mean of a list of values.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1 - alpha) * 100))
    point = float(arr.mean())
    return point, lower, upper


# ──────────────────────────────────────────────────────────────────────
#  Next-Word CSR: Demo-faithful keyboard simulation (PyTorch)
# ──────────────────────────────────────────────────────────────────────
#
#  Ports the demo/server.py _keyboard_candidates algorithm AND the
#  frontend mergeCandidates/acceptChip logic (predictive-keyboard.html)
#  into PyTorch, so training-time validation reflects what users
#  actually experience in the deployed keyboard.
#
#  This is a SECONDARY metric. PPL remains primary for checkpoint ranking.
#  The decomposed metrics (top1_accuracy, top3_accuracy, top5_accuracy,
#  acceptance_rate, avg_prefix) are more informative than raw CSR because
#  they don't suffer from the CSR paradox (§6.14: agglutinative word length
#  penalizes native-language keystroke savings).
#
#  Strategies ported (§5.5):
#    1. Retokenization fallback (shorter-prefix queries)
#    2. Top-k fetch (k=5), top-3 display
#    3. Sticky merge (carry-forward +0.1 boost)
#    4. Next-word candidate extraction
#    5. Word-level matching (not char-level)
#    6. Acceptance semantics (auto-space, punctuation attachment)
#
#  The only model interaction is greedy generation + top-k logprobs,
#  implemented natively in PyTorch (no llama.cpp dependency).

_NW_PUNCT_CHARS = '.!,?;:()[]{}'
_NW_STICKY_BOOST = 0.1
_NW_FETCH_K = 5       # candidates fetched from model
_NW_POOL_SIZE = 5    # sticky pool stores this many
_NW_DISPLAY_K = 3    # only top-3 are "visible" as chips
_NW_MAX_TOKENS = 5   # max tokens per greedy generation call
_NW_EOS_ID = 2       # </s> separator token


@torch.no_grad()
def _nw_greedy_generate(model, sp, context_ids, device, max_tokens=_NW_MAX_TOKENS, n_probs=_NW_FETCH_K):
    """PyTorch equivalent of demo _call_llama (greedy mode).

    Returns dict with 'content' (greedy decoded text) and
    'completion_probabilities' (per-token: id, logprob, top_logprobs).
    This is the only model interaction needed by the keyboard algorithm.
    """
    ids = list(context_ids)
    if not ids:
        return {"content": "", "completion_probabilities": []}

    generated_ids = []
    completion_probs = []

    for _ in range(max_tokens):
        ctx = torch.tensor([ids], device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ctx).logits[0, -1, :].float()

        log_probs = torch.log_softmax(logits, dim=-1)
        next_id = int(torch.argmax(logits).item())
        next_logprob = float(log_probs[next_id].item())

        topk_logprobs, topk_ids = torch.topk(log_probs, n_probs)
        top_logprobs = [
            {"id": int(tid), "logprob": float(lp)}
            for tid, lp in zip(topk_ids.tolist(), topk_logprobs.tolist())
        ]

        completion_probs.append({
            "id": next_id,
            "logprob": next_logprob,
            "top_logprobs": top_logprobs,
        })

        if next_id == 0 or next_id == _NW_EOS_ID:
            break

        ids.append(next_id)
        generated_ids.append(next_id)

    content = sp.decode(generated_ids) if generated_ids else ""
    return {"content": content, "completion_probabilities": completion_probs}


def _nw_decode_token(sp, tid):
    """Decode a token ID to surface text (replaces demo _decode_token)."""
    piece = sp.id_to_piece(tid)
    return piece.replace("\u2581", " ")


def _nw_token_has_digit(text):
    return any(c.isdigit() for c in text)


def _nw_extract_current_word(text):
    """Split text into (text_before_word, current_word) at cursor (end).

    Port of demo/server.py _extract_current_word.
    """
    if not text or text[-1].isspace():
        return text, ""
    i = len(text) - 1
    while i >= 0 and not text[i].isspace():
        i -= 1
    return text[:i + 1], text[i + 1:]


def _nw_extract_first_word(content):
    """Extract first word from generated content. Strips trailing punct.

    Port of demo/server.py _extract_first_word.
    Returns empty string if content starts with whitespace.
    """
    if not content or content[0].isspace():
        return ""
    word = ""
    for c in content:
        if c.isspace():
            break
        word += c
    return word.rstrip(_NW_PUNCT_CHARS)


def _nw_keyboard_candidates(model, sp, text, device, top_k=_NW_FETCH_K, filter_digits=True):
    """Generate word-completion candidates with retokenization fallback.

    Faithful PyTorch port of demo/server.py _keyboard_candidates.
    Returns list of {text, prob, is_next_word?} dicts.
    """
    text_before_word, current_word = _nw_extract_current_word(text)

    # ── Next-word prediction (cursor after space) ──
    if not current_word:
        prompt_ids = sp.encode(text, out_type=int)
        if len(prompt_ids) > 1024:
            prompt_ids = prompt_ids[-1024:]
        result = _nw_greedy_generate(model, sp, prompt_ids, device)
        probs = result.get("completion_probabilities", [])
        content = result.get("content", "")
        candidates = []
        seen = set()

        if content:
            stripped = content.lstrip()
            first_word = ""
            if stripped:
                for c in stripped:
                    if c.isspace():
                        break
                    first_word += c
                first_word = first_word.rstrip(_NW_PUNCT_CHARS)
            if first_word and not (filter_digits and _nw_token_has_digit(first_word)):
                prob = math.exp(probs[0]["logprob"]) if probs else 0.5
                candidates.append({"text": first_word, "prob": round(prob, 4)})
                seen.add(first_word)

        if probs:
            for tok in probs[0].get("top_logprobs", []):
                tok_text = _nw_decode_token(sp, tok["id"])
                if not tok_text.strip():
                    continue
                if filter_digits and _nw_token_has_digit(tok_text):
                    continue
                word = tok_text.strip()
                if word and word not in seen:
                    candidates.append({"text": word, "prob": round(math.exp(tok["logprob"]), 4)})
                    seen.add(word)
                if len(candidates) >= top_k:
                    break
        return candidates[:top_k]

    # ── Word completion (cursor mid-word) ──
    candidates_map = {}

    max_fallback = min(2, len(current_word) - 1)
    fallback_paths = []
    for fallback in range(max_fallback + 1):
        shorter_len = len(current_word) - fallback
        if shorter_len < 1:
            break
        shorter_word = current_word[:shorter_len]
        prefix = text_before_word + shorter_word
        ids = sp.encode(prefix, out_type=int)
        if len(ids) > 1024:
            ids = ids[-1024:]
        fallback_paths.append((shorter_word, ids, False))
    if text_before_word.strip():
        ids = sp.encode(text_before_word, out_type=int)
        if len(ids) > 1024:
            ids = ids[-1024:]
        fallback_paths.append(("", ids, True))

    for shorter_word, ids, is_from_scratch in fallback_paths:
        result = _nw_greedy_generate(model, sp, ids, device)
        probs = result.get("completion_probabilities", [])
        content = result.get("content", "")

        if content:
            raw = content.lstrip() if is_from_scratch else content
            word_completion = _nw_extract_first_word(raw)
            if word_completion:
                full_word = shorter_word + word_completion
                if (full_word.startswith(current_word)
                        and len(full_word) >= len(current_word)
                        and not (filter_digits and _nw_token_has_digit(full_word))):
                    prob = math.exp(probs[0]["logprob"]) if probs else 0.5
                    if full_word not in candidates_map or prob > candidates_map[full_word]["prob"]:
                        candidates_map[full_word] = {"text": full_word, "prob": round(prob, 4)}

            if not is_from_scratch and content and content[0].isspace():
                next_word = _nw_extract_first_word(content.lstrip())
                if next_word and not (filter_digits and _nw_token_has_digit(next_word)):
                    prob = math.exp(probs[0]["logprob"]) if probs else 0.5
                    key = "__next__" + next_word
                    if key not in candidates_map or prob > candidates_map[key]["prob"]:
                        candidates_map[key] = {"text": next_word, "prob": round(prob, 4), "is_next_word": True}

        if probs:
            greedy_first_id = probs[0].get("id")
            for tok in probs[0].get("top_logprobs", []):
                if tok.get("id") == greedy_first_id:
                    continue
                tok_text = _nw_decode_token(sp, tok["id"])
                if not tok_text.strip():
                    continue
                if filter_digits and _nw_token_has_digit(tok_text):
                    continue
                if is_from_scratch:
                    tok_text = tok_text.lstrip()
                    if not tok_text:
                        continue
                else:
                    if tok_text[0].isspace():
                        next_word = tok_text.strip()
                        if next_word and not (filter_digits and _nw_token_has_digit(next_word)):
                            prob = math.exp(tok["logprob"])
                            key = "__next__" + next_word
                            if key not in candidates_map or prob > candidates_map[key]["prob"]:
                                candidates_map[key] = {"text": next_word, "prob": round(prob, 4), "is_next_word": True}
                        continue
                full_word = shorter_word + tok_text
                if (full_word.startswith(current_word)
                        and len(full_word) >= len(current_word)):
                    prob = math.exp(tok["logprob"])
                    if full_word not in candidates_map or prob > candidates_map[full_word]["prob"]:
                        candidates_map[full_word] = {"text": full_word, "prob": round(prob, 4)}

    sorted_cands = sorted(candidates_map.values(), key=lambda x: x["prob"], reverse=True)
    return sorted_cands[:top_k]


def _nw_clean_word(w):
    """Lowercase and strip punctuation for word comparison."""
    return w.lower().strip('.,!?;:()[]{}"\'')


def _nw_is_punct(text):
    t = text.strip()
    return len(t) > 0 and all(c in _NW_PUNCT_CHARS for c in t)


def _nw_get_current_word(text):
    i = len(text) - 1
    while i >= 0 and not text[i].isspace():
        i -= 1
    return text[i + 1:]


def _nw_get_current_word_start(text):
    i = len(text) - 1
    while i >= 0 and not text[i].isspace():
        i -= 1
    return i + 1


def _nw_merge_candidates(sticky_pool, fresh_candidates, current_word):
    """Port of frontend mergeCandidates(): carry-forward + boost + merge."""
    cw = current_word.lower()

    survivors = []
    if sticky_pool and len(cw) > 0:
        survivors = [
            {"text": c["text"], "prob": c["prob"],
             "is_next_word": c.get("is_next_word", False), "_sticky": True}
            for c in sticky_pool
            if c["text"].lower().startswith(cw)
        ]

    survivor_texts = set(c["text"] for c in survivors)
    fresh_only = [
        {"text": c["text"], "prob": c["prob"],
         "is_next_word": c.get("is_next_word", False), "_sticky": False}
        for c in fresh_candidates
        if c["text"] not in survivor_texts
    ]

    merged = survivors + fresh_only
    merged.sort(
        key=lambda c: c["prob"] + (_NW_STICKY_BOOST if c["_sticky"] else 0),
        reverse=True
    )

    new_pool = [
        {"text": c["text"], "prob": c["prob"],
         "is_next_word": c.get("is_next_word", False)}
        for c in merged[:_NW_POOL_SIZE]
    ]

    displayed = [
        {"text": c["text"], "prob": c["prob"],
         "is_next_word": c.get("is_next_word", False),
         "_sticky": c["_sticky"]}
        for c in merged[:_NW_DISPLAY_K]
    ]
    return displayed, new_pool


def _nw_accept_chip(text, is_next_word, editor_value, cursor_pos):
    """Port of frontend acceptChip(): returns (new_value, new_cursor)."""
    new_text = text
    is_p = _nw_is_punct(new_text)

    if is_next_word:
        insert = " " + new_text + " "
        new_val = editor_value[:cursor_pos] + insert + editor_value[cursor_pos:]
        new_pos = cursor_pos + len(insert)
    elif is_p:
        insert_pos = cursor_pos
        before = editor_value[:insert_pos]
        after = editor_value[cursor_pos:]
        if before.endswith(" "):
            before = before[:-1]
            insert_pos -= 1
        insert = new_text.strip() + " "
        new_val = before + insert + after
        new_pos = insert_pos + len(insert)
    else:
        if not new_text.endswith(" ") and not new_text.endswith("\n"):
            new_text += " "
        word_start = _nw_get_current_word_start(editor_value[:cursor_pos])
        new_val = editor_value[:word_start] + new_text + editor_value[cursor_pos:]
        new_pos = word_start + len(new_text)

    return new_val, new_pos


def evaluate_next_word_csr(model, sp, tests, device="cuda"):
    """Next-word CSR: demo-faithful keyboard simulation in PyTorch.

    Types each test's target_completion word-by-word, using the exact same
    algorithm as the deployed predictive keyboard (retokenization fallback,
    sticky merge, top-3 display, acceptance semantics). Accepts a suggestion
    the moment it matches the target word in the top-3 chips.

    Returns per-test result dicts with:
      csr, acceptance_rate, top1_accuracy, top3_accuracy, top5_accuracy,
      avg_prefix, avg_confidence,
      completed_words, events, keystrokes, taps, n_words
    """
    model.eval()
    results = []

    for t in tests:
        prompt = t["input"]
        target = t["target_completion"]
        words = target.split()

        word_idx = 0
        char_idx = 0
        editor = prompt
        cursor = len(prompt)
        sticky_pool = []
        events = []
        keystrokes = 0
        chars_typed = 0
        taps = 0
        completed_words = []
        # Per-word tracking: was the target ever the #1 / #3 / #5 candidate?
        was_top1 = False
        was_top3 = False
        was_top5 = False
        top1_count = 0
        top3_count = 0
        top5_count = 0

        while word_idx < len(words):
            target_word = words[word_idx]
            target_c = _nw_clean_word(target_word)

            prefix = editor[:cursor]
            fresh = _nw_keyboard_candidates(model, sp, prefix, device)
            current_word = _nw_get_current_word(prefix)
            displayed, sticky_pool = _nw_merge_candidates(sticky_pool, fresh, current_word)

            # Record prefix length for acceptance events
            events.append({
                "type": "check",
                "prefix": prefix[-60:],
                "displayed": [{"text": c["text"], "prob": c["prob"]} for c in displayed],
            })

            # ── Track Top-K accuracy at this keystroke ──
            # Top-1: correct word is the #1 displayed chip (post-sticky-merge)
            if displayed and len(displayed) > 0:
                top1_cand = _nw_clean_word(displayed[0]["text"])
                if top1_cand == target_c and len(top1_cand) > 0:
                    was_top1 = True
            # Top-3: correct word in top-3 displayed chips (= acceptance condition)
            for c in displayed[:3]:
                if _nw_clean_word(c["text"]) == target_c and len(_nw_clean_word(c["text"])) > 0:
                    was_top3 = True
                    break
            # Top-5: correct word in the raw fetched pool (pre-sticky-merge ceiling)
            for c in fresh[:5]:
                if _nw_clean_word(c["text"]) == target_c and len(_nw_clean_word(c["text"])) > 0:
                    was_top5 = True
                    break

            accepted = False
            for c in displayed:
                cand_c = _nw_clean_word(c["text"])
                if cand_c == target_c and len(cand_c) > 0:
                    is_nw = c.get("is_next_word", False)
                    # Record prefix length: how many chars of this word were typed
                    typed_word = _nw_get_current_word(prefix)
                    events.append({
                        "type": "accept",
                        "method": "next_word" if is_nw else "completion",
                        "accepted": c["text"],
                        "prob": c["prob"],
                        "sticky": c.get("_sticky", False),
                        "prefix_len": len(typed_word),
                    })

                    editor, cursor = _nw_accept_chip(c["text"], is_nw, editor, cursor)
                    taps += 1
                    keystrokes += 1
                    completed_words.append((c["text"], "next_word" if is_nw else "completion", c["prob"]))
                    sticky_pool = []
                    # Tally per-word Top-K flags before resetting
                    if was_top1:
                        top1_count += 1
                    if was_top3:
                        top3_count += 1
                    if was_top5:
                        top5_count += 1
                    was_top1 = was_top3 = was_top5 = False
                    word_idx += 1
                    char_idx = 0
                    accepted = True
                    break

            if accepted:
                continue

            if char_idx < len(target_word):
                if char_idx == 0 and cursor > 0 and not editor[cursor - 1].isspace():
                    editor = editor[:cursor] + " " + editor[cursor:]
                    cursor += 1
                    chars_typed += 1
                    keystrokes += 1

                ch = target_word[char_idx]
                editor = editor[:cursor] + ch + editor[cursor:]
                cursor += 1
                char_idx += 1
                chars_typed += 1
                keystrokes += 1
            else:
                # Typed the whole word manually — tally per-word Top-K flags
                if was_top1:
                    top1_count += 1
                if was_top3:
                    top3_count += 1
                if was_top5:
                    top5_count += 1
                was_top1 = was_top3 = was_top5 = False
                completed_words.append((target_word, "manual", 0.0))
                word_idx += 1
                char_idx = 0

        total_chars = len(target)
        all_probs = [p for w, m, p in completed_words if p > 0]
        accept_events = [e for e in events if e["type"] == "accept"]
        prefix_lens = [e.get("prefix_len", 0) for e in accept_events]

        results.append({
            "prompt": prompt,
            "target": target,
            "csr": round((total_chars - keystrokes) / total_chars, 4) if total_chars > 0 else 0.0,
            "acceptance_rate": round(taps / len(words), 4) if words else 0.0,
            "top1_accuracy": round(top1_count / len(words), 4) if words else 0.0,
            "top3_accuracy": round(top3_count / len(words), 4) if words else 0.0,
            "top5_accuracy": round(top5_count / len(words), 4) if words else 0.0,
            "avg_prefix": round(sum(prefix_lens) / len(prefix_lens), 2) if prefix_lens else 0.0,
            "avg_confidence": round(sum(all_probs) / len(all_probs), 4) if all_probs else 0.0,
            "completed_words": completed_words,
            "events": events,
            "keystrokes": keystrokes,
            "chars_typed": chars_typed,
            "taps": taps,
            "n_words": len(words),
            "total_chars": total_chars,
        })

    model.train()
    return results


# ──────────────────────────────────────────────────────────────────────
#  Aggregate helper (for train.py W&B logging)
# ──────────────────────────────────────────────────────────────────────

def compute_autocomplete_metrics(model, sp, csr_tests, morphacc_tests, device, k=5):
    """Run CSR and MorphAcc eval, return aggregate metrics for W&B logging.

    Restores model to train mode after evaluation.
    """
    csr_results = evaluate_csr(model, sp, csr_tests, device)
    total_chars = sum(r["total_chars"] for r in csr_results)
    total_saved = sum(r["keystrokes_saved"] for r in csr_results)
    csr = total_saved / total_chars if total_chars > 0 else 0.0

    # Bootstrap CI on per-test macro CSR (so small eval sets show their
    # uncertainty honestly)
    macro_csrs = [r["csr"] for r in csr_results]
    csr_point, csr_lo, csr_hi = bootstrap_mean_ci(macro_csrs)

    morphacc_results = evaluate_morphacc(model, sp, morphacc_tests, device, k=k)
    morphacc_hits = sum(1 for r in morphacc_results if r["morphacc_hit"])
    morphacc_total = len(morphacc_results)
    morphacc = morphacc_hits / morphacc_total if morphacc_total > 0 else 0.0

    model.train()

    # Next-word CSR (demo-faithful keyboard simulation, §6.14)
    nw_results = evaluate_next_word_csr(model, sp, csr_tests, device)
    nw_macro_csrs = [r["csr"] for r in nw_results]
    nw_csr_point, nw_csr_lo, nw_csr_hi = bootstrap_mean_ci(nw_macro_csrs)

    nw_total_chars = sum(r["total_chars"] for r in nw_results)
    nw_total_ks = sum(r["keystrokes"] for r in nw_results)
    nw_total_words = sum(r["n_words"] for r in nw_results)
    nw_total_accepts = sum(r["taps"] for r in nw_results)
    nw_total_top1 = sum(r["top1_accuracy"] * r["n_words"] for r in nw_results)
    nw_total_top3 = sum(r["top3_accuracy"] * r["n_words"] for r in nw_results)
    nw_total_top5 = sum(r["top5_accuracy"] * r["n_words"] for r in nw_results)
    nw_all_probs = [p for r in nw_results for w, m, p in r["completed_words"] if p > 0]
    nw_prefix_lens = []
    for r in nw_results:
        for e in r["events"]:
            if e["type"] == "accept":
                nw_prefix_lens.append(e.get("prefix_len", 0))

    model.train()

    return {
        "valid/csr": round(csr, 4),
        "valid/csr_macro": round(csr_point, 4),
        "valid/csr_macro_ci_lower": round(csr_lo, 4),
        "valid/csr_macro_ci_upper": round(csr_hi, 4),
        "valid/morphacc": round(morphacc, 4),
        "valid/csr_chars_saved": total_saved,
        "valid/csr_total_chars": total_chars,
        # Next-word CSR — demo-faithful keyboard simulation (secondary metric)
        "valid/nw_csr": round((nw_total_chars - nw_total_ks) / nw_total_chars, 4) if nw_total_chars > 0 else 0.0,
        "valid/nw_csr_macro": round(nw_csr_point, 4),
        "valid/nw_csr_ci_lower": round(nw_csr_lo, 4),
        "valid/nw_csr_ci_upper": round(nw_csr_hi, 4),
        "valid/nw_acceptance_rate": round(nw_total_accepts / nw_total_words, 4) if nw_total_words > 0 else 0.0,
        "valid/nw_top1_accuracy": round(nw_total_top1 / nw_total_words, 4) if nw_total_words > 0 else 0.0,
        "valid/nw_top3_accuracy": round(nw_total_top3 / nw_total_words, 4) if nw_total_words > 0 else 0.0,
        "valid/nw_top5_accuracy": round(nw_total_top5 / nw_total_words, 4) if nw_total_words > 0 else 0.0,
        "valid/nw_avg_prefix_before_accept": round(sum(nw_prefix_lens) / len(nw_prefix_lens), 2) if nw_prefix_lens else 0.0,
        "valid/nw_avg_confidence": round(sum(nw_all_probs) / len(nw_all_probs), 4) if nw_all_probs else 0.0,
    }
