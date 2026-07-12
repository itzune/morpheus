#!/usr/bin/env python3
"""
Simulate typing through the predictive keyboard, faithfully replicating the
FRONTEND algorithm from demo/static/predictive-keyboard.html:

  1. WebSocket /ws/greedy with keyboard_mode=true (we use the equivalent REST
     endpoint /api/autocomplete/keyboard — same backend _keyboard_candidates)
  2. Sticky merge: carry forward previous candidates matching the current
     prefix, with +0.1 prob boost (STICKY_BOOST = 0.1)
  3. Top-k=5 fetched, sticky pool stores 5, only top-3 displayed
  4. Acceptance semantics:
     - Next-word: insert ' ' + word + ' ' (leading + trailing space)
     - Punctuation: attach to previous word (remove pre-space), add trailing space
     - Normal word: replace partial word from wordStart, add trailing space
  5. Sticky pool resets on accept and on send

This produces results comparable to what a real user would experience.

Usage:
    python3 scripts/simulate_typing.py
    python3 scripts/simulate_typing.py --host http://localhost:9090 --delay 0 --verbose
"""
import argparse
import json
import time
import sys
from pathlib import Path

import httpx

# ── Constants matching the frontend ──────────────────────────────────────────
STICKY_BOOST = 0.1
FETCH_K = 5       # candidates fetched from server
POOL_SIZE = 5     # sticky pool stores this many
DISPLAY_K = 3     # only top-3 are visible as chips

# ── Test sentences (mixed languages, varied topics) ──────────────────────────
SENTENCES = [
    # Basque (native language)
    ("eu", "Gaur eguraldi ona egiten du eta kalean paseatzera aterako naiz"),
    ("eu", "Euskara ikasten ari naiz baina oraindik zaila iruditzen zait"),
    ("eu", "Bihar goizean lanera joan beharko dut goiz"),
    ("eu", "Lagunekin afaria egitea asteburuan plangarria da"),
    ("eu", "Musika entzutea gustuko dut lanean ari naizenean"),
    # English (cross-lingual, <1% training data)
    ("en", "The weather is nice today and I will go for a walk"),
    ("en", "I am learning a new language but it is still difficult"),
    ("en", "Tomorrow morning I have to go to work early"),
    ("en", "Having dinner with friends this weekend sounds great"),
    ("en", "I like to listen to music while I am working"),
    # Spanish (cross-lingual, <1% training data)
    ("es", "El tiempo está bueno hoy y voy a salir a caminar"),
    ("es", "Estoy aprendiendo un idioma nuevo pero todavía es difícil"),
    ("es", "Mañana por la mañana tengo que ir a trabajar temprano"),
    ("es", "Cenar con amigos este fin de semana suena genial"),
    ("es", "Me gusta escuchar música mientras estoy trabajando"),
]

PUNCT = set(".,!?;:()[]{}")


def clean_word(w):
    """Lowercase and strip punctuation for comparison."""
    return w.lower().strip(".,!?;:()[]{}\"'")


def is_punct(text):
    """Check if text is purely punctuation (matching frontend isPunct logic)."""
    t = text.strip()
    return len(t) > 0 and all(c in PUNCT for c in t)


def query_keyboard(host, text, top_k=FETCH_K, timeout=60, retries=3):
    """Query the keyboard autocomplete API with retry + backoff.

    llama-server can deadlock or slow down under rapid load (especially
    after model reloads). Retries with exponential backoff prevent
    cascading failures.
    """
    for attempt in range(retries):
        try:
            r = httpx.get(
                f"{host}/api/autocomplete/keyboard",
                params={"text": text, "top_k": top_k, "max_tokens": 5},
                timeout=timeout,
            )
            return r.json().get("candidates", [])
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"  [RETRY {attempt+1}/{retries}] {e} — waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [ERROR] API query failed after {retries} retries: {e}", file=sys.stderr)
                return []


def get_current_word(text):
    """Extract the current partial word being typed (text after last whitespace)."""
    i = len(text) - 1
    while i >= 0 and not text[i].isspace():
        i -= 1
    return text[i + 1:]


def get_current_word_start(text):
    """Find where the current word starts in the full text."""
    i = len(text) - 1
    while i >= 0 and not text[i].isspace():
        i -= 1
    return i + 1


def merge_candidates(sticky_pool, fresh_candidates, current_word):
    """
    Faithfully replicate the frontend's mergeCandidates() function.

    1. Filter sticky pool by current prefix (case-insensitive)
    2. Give survivors +STICKY_BOOST
    3. Merge with fresh candidates (dedup by text)
    4. Sort by boosted prob
    5. Update sticky pool (store POOL_SIZE, original prob)
    6. Return top DISPLAY_K
    """
    cw = current_word.lower()

    # Filter previous candidates by the current prefix
    survivors = []
    if sticky_pool and len(cw) > 0:
        survivors = [
            {"text": c["text"], "prob": c["prob"],
             "is_next_word": c.get("is_next_word", False), "_sticky": True}
            for c in sticky_pool
            if c["text"].lower().startswith(cw)
        ]

    # Fresh candidates not already in survivors
    survivor_texts = set(c["text"] for c in survivors)
    fresh_only = [
        {"text": c["text"], "prob": c["prob"],
         "is_next_word": c.get("is_next_word", False), "_sticky": False}
        for c in fresh_candidates
        if c["text"] not in survivor_texts
    ]

    # Merge and sort by boosted prob
    merged = survivors + fresh_only
    merged.sort(
        key=lambda c: c["prob"] + (STICKY_BOOST if c["_sticky"] else 0),
        reverse=True
    )

    # Update sticky state (store original prob, no boost)
    new_pool = [
        {"text": c["text"], "prob": c["prob"],
         "is_next_word": c.get("is_next_word", False)}
        for c in merged[:POOL_SIZE]
    ]

    # Return top DISPLAY_K (strip _sticky flag)
    displayed = [
        {"text": c["text"], "prob": c["prob"],
         "is_next_word": c.get("is_next_word", False)}
        for c in merged[:DISPLAY_K]
    ]

    return displayed, new_pool


def accept_chip(text, is_next_word, editor_value, cursor_pos):
    """
    Faithfully replicate the frontend's acceptChip() function.

    Returns (new_editor_value, new_cursor_pos).
    """
    new_text = text
    is_p = is_punct(new_text)

    if is_next_word:
        # Insert ' ' + word + ' '
        insert = " " + new_text + " "
        new_val = editor_value[:cursor_pos] + insert + editor_value[cursor_pos:]
        new_pos = cursor_pos + len(insert)
    elif is_p:
        # Attach to previous word (remove pre-space), add trailing space
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
        # Normal word: replace partial from wordStart, add trailing space
        if not new_text.endswith(" ") and not new_text.endswith("\n"):
            new_text += " "
        word_start = get_current_word_start(editor_value[:cursor_pos])
        new_val = editor_value[:word_start] + new_text + editor_value[cursor_pos:]
        new_pos = word_start + len(new_text)

    return new_val, new_pos


def simulate_sentence(host, lang, sentence, delay=0.05, initial_text=""):
    """
    Simulate typing a sentence char by char, using the EXACT same algorithm
    as the frontend: sticky merge, top-3 display, acceptance semantics.

    If initial_text is provided (e.g. CSR test context), the editor starts
    with that text pre-filled, and the sentence is typed after it.
    """
    words = sentence.split()
    word_idx = 0       # which target word we're on
    char_idx = 0       # how many chars of current word we've typed
    editor = initial_text  # full text in the editor
    cursor = len(initial_text)  # cursor position

    sticky_pool = []   # sticky merge pool

    events = []
    keystrokes = 0     # total keystrokes (chars + taps)
    chars_typed = 0
    taps = 0
    completed_words = []
    # Per-word tracking: was the target ever the #1 / #3 / #5 candidate?
    was_top1 = False
    was_top3 = False
    was_top5 = False
    top1_count = 0   # aggregate across all words
    top3_count = 0
    top5_count = 0

    while word_idx < len(words):
        target = words[word_idx]
        target_c = clean_word(target)

        # ── Query server ──
        prefix = editor[:cursor]
        fresh = query_keyboard(host, prefix)

        # ── Sticky merge ──
        current_word = get_current_word(prefix)
        displayed, sticky_pool = merge_candidates(sticky_pool, fresh, current_word)

        events.append({
            "type": "check",
            "prefix": prefix,
            "fresh": fresh,
            "displayed": displayed,
            "sticky_pool": sticky_pool[:3],  # log top 3 of pool
        })

        # ── Track Top-K accuracy at this keystroke ──
        # Top-1: correct word is the #1 displayed chip (post-sticky-merge)
        if displayed and len(displayed) > 0:
            top1_cand = clean_word(displayed[0]["text"])
            if top1_cand == target_c and len(top1_cand) > 0:
                was_top1 = True
        # Top-3: correct word in top-3 displayed chips (= acceptance condition)
        for c in displayed[:DISPLAY_K]:
            if clean_word(c["text"]) == target_c and len(clean_word(c["text"])) > 0:
                was_top3 = True
                break
        # Top-5: correct word in the raw fetched pool (pre-sticky-merge ceiling)
        for c in fresh[:FETCH_K]:
            if clean_word(c["text"]) == target_c and len(clean_word(c["text"])) > 0:
                was_top5 = True
                break

        # ── Look for a matching candidate in DISPLAYED (top-3) ──
        accepted = False
        for c in displayed:
            cand_c = clean_word(c["text"])
            if cand_c == target_c and len(cand_c) > 0:
                # Accept this chip!
                is_nw = c.get("is_next_word", False)
                editor, cursor = accept_chip(c["text"], is_nw, editor, cursor)

                taps += 1
                keystrokes += 1
                method = "next_word" if is_nw else "completion"
                events.append({
                    "type": "accept",
                    "method": method,
                    "accepted": c["text"],
                    "prob": c["prob"],
                    "sticky": c.get("_sticky", False),
                    "prefix_len": len(get_current_word(prefix)),
                    "editor_after": editor[:cursor],
                })
                completed_words.append((c["text"], method, c["prob"]))

                # Reset sticky pool on accept
                sticky_pool = []

                word_idx += 1
                char_idx = 0
                accepted = True
                if delay:
                    time.sleep(delay)
                break

        if accepted:
            # Tally per-word Top-K flags before resetting
            if was_top1:
                top1_count += 1
            if was_top3:
                top3_count += 1
            if was_top5:
                top5_count += 1
            was_top1 = was_top3 = was_top5 = False
            continue

        # ── No match; type next character ──
        if char_idx < len(target):
            # Add space if at word boundary and not first word
            if char_idx == 0 and cursor > 0 and not editor[cursor - 1].isspace():
                editor = editor[:cursor] + " " + editor[cursor:]
                cursor += 1
                chars_typed += 1
                keystrokes += 1
                events.append({"type": "keystroke", "char": " ", "editor": editor[:cursor]})

            ch = target[char_idx]
            editor = editor[:cursor] + ch + editor[cursor:]
            cursor += 1
            char_idx += 1
            chars_typed += 1
            keystrokes += 1
            events.append({"type": "keystroke", "char": ch, "editor": editor[:cursor]})

            if delay:
                time.sleep(delay)
        else:
            # Typed the whole word manually — tally per-word Top-K flags
            if was_top1:
                top1_count += 1
            if was_top3:
                top3_count += 1
            if was_top5:
                top5_count += 1
            was_top1 = was_top3 = was_top5 = False
            completed_words.append((target, "manual", 0.0))
            word_idx += 1
            char_idx = 0

    return {
        "lang": lang,
        "sentence": sentence,
        "words_target": words,
        "completed_words": completed_words,
        "events": events,
        "keystrokes": keystrokes,
        "chars_typed": chars_typed,
        "taps": taps,
        "n_words": len(words),
        "top1_accuracy": round(top1_count / len(words), 4) if words else 0.0,
        "top3_accuracy": round(top3_count / len(words), 4) if words else 0.0,
        "top5_accuracy": round(top5_count / len(words), 4) if words else 0.0,
        "total_chars": len(sentence),
        "csr": (len(sentence) - keystrokes) / len(sentence) * 100 if len(sentence) > 0 else 0,
        "final_editor": editor.strip(),
    }


def print_typing_trace(result):
    """Print a detailed trace of one sentence simulation."""
    lang_name = {"eu": "Basque", "en": "English", "es": "Spanish"}[result["lang"]]
    print(f"\n{'='*70}")
    print(f"  {lang_name}: \"{result['sentence']}\"")
    print(f"{'='*70}")

    for event in result["events"]:
        if event["type"] == "keystroke":
            ch = event["char"]
            label = "[space]" if ch == " " else f"'{ch}'"
            print(f"  ⌨  {label:8s}       → \"{event['editor']}\"")
        elif event["type"] == "check":
            disp = event.get("displayed", [])
            if disp:
                strs = []
                for c in disp:
                    tag = "📎" if c.get("_sticky") else "  "
                    strs.append(f"{tag}{c['text']}({c['prob']:.2f})")
                print(f"     🔍 chips: {' | '.join(strs)}")
            else:
                print(f"     🔍 chips: (none)")
        elif event["type"] == "accept":
            method = event["method"]
            acc = event["accepted"]
            prob = event["prob"]
            sticky = "📎 sticky" if event.get("sticky") else ""
            if method == "next_word":
                print(f"  ✅ ACCEPT (next-word → '{acc}')  prob={prob:.3f} {sticky}")
            else:
                print(f"  ✅ ACCEPT (→ '{acc}')  prob={prob:.3f} {sticky}")

    completions = [w for w in result["completed_words"] if w[1] == "completion"]
    next_words = [w for w in result["completed_words"] if w[1] == "next_word"]
    manuals = [w for w in result["completed_words"] if w[1] == "manual"]
    print(f"\n  📊 Result: Top-1={100*result['top1_accuracy']:.0f}%  Top-3={100*result['top3_accuracy']:.0f}%  Top-5={100*result['top5_accuracy']:.0f}%")
    print(f"     Completions: {len(completions)}  Next-words: {len(next_words)}  Manual: {len(manuals)}")
    print(f"     Keystrokes: {result['keystrokes']} (chars: {result['chars_typed']}, taps: {result['taps']})")
    print(f"     Simulated CSR: {result['csr']:.1f}%")
    all_probs = [w[2] for w in completions + next_words if w[2] > 0]
    if all_probs:
        print(f"     Avg prob of accepted: {sum(all_probs)/len(all_probs):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Simulate typing through predictive keyboard (frontend-faithful)")
    parser.add_argument("--host", default="http://localhost:9090")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between API calls (seconds)")
    parser.add_argument("--output", default=None, help="Save full results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed typing trace")
    parser.add_argument("--targets", default=None,
                        help="Load CSR tests from targets JSON (uses input+target_completion)")
    args = parser.parse_args()

    # Build test list: either CSR tests from file or hardcoded SENTENCES
    if args.targets:
        import json as _json
        with open(args.targets) as f:
            tdata = _json.load(f)
        csr_tests = []
        for s in tdata.get("strategies", []):
            if s.get("name") == "csr":
                for t in s.get("tests", []):
                    csr_tests.append(("eu", t["target_completion"], t["input"]))
                break
        test_list = csr_tests
        print(f"Simulating typing on {args.host}")
        print(f"CSR tests: {len(test_list)} (from {args.targets})")
    else:
        test_list = [(lang, sent, "") for lang, sent in SENTENCES]
        print(f"Simulating typing on {args.host}")
        print(f"Sentences: {len(test_list)} (5 Basque, 5 English, 5 Spanish)")

    print(f"Algorithm: frontend-faithful (sticky merge, top-3 display, acceptance semantics)")
    print(f"Delay: {args.delay}s")
    print()

    all_results = []
    for lang, sentence, initial_text in test_list:
        lang_name = {"eu": "Basque", "en": "English", "es": "Spanish"}.get(lang, lang)
        preview = sentence[:50]
        if initial_text:
            print(f"  Typing [{lang_name}]: ctx=\"{initial_text[:30]}...\" + \"{preview}...\"", end="", flush=True)
        else:
            print(f"  Typing [{lang_name}]: \"{preview}...\"", end="", flush=True)
        result = simulate_sentence(args.host, lang, sentence, delay=args.delay, initial_text=initial_text)
        all_results.append(result)
        status = "✅" if result["taps"] == result["n_words"] else f"⚠️  {result['taps']}/{result['n_words']} accepted"
        print(f" → {status}  CSR={result['csr']:.1f}%  Top1={100*result['top1_accuracy']:.0f}%  Top5={100*result['top5_accuracy']:.0f}%")
        print()

    if args.verbose:
        for result in all_results:
            print_typing_trace(result)

    # ── Aggregate analysis ──
    print(f"\n{'='*70}")
    print(f"  AGGREGATE ANALYSIS (frontend-faithful)")
    print(f"{'='*70}")

    for lang_code in ["eu", "en", "es"]:
        lang_name = {"eu": "Basque", "en": "English", "es": "Spanish"}[lang_code]
        lr = [r for r in all_results if r["lang"] == lang_code]
        if not lr:
            continue

        n_words = sum(r["n_words"] for r in lr)
        total_ks = sum(r["keystrokes"] for r in lr)
        total_chars = sum(r["total_chars"] for r in lr)
        total_taps = sum(r["taps"] for r in lr)
        total_top1 = sum(r["top1_accuracy"] * r["n_words"] for r in lr)
        total_top3 = sum(r["top3_accuracy"] * r["n_words"] for r in lr)
        total_top5 = sum(r["top5_accuracy"] * r["n_words"] for r in lr)

        completions = sum(1 for r in lr for w in r["completed_words"] if w[1] == "completion")
        next_words = sum(1 for r in lr for w in r["completed_words"] if w[1] == "next_word")
        manuals = sum(1 for r in lr for w in r["completed_words"] if w[1] == "manual")
        all_probs = [w[2] for r in lr for w in r["completed_words"] if w[2] > 0]
        prefix_lens = [e["prefix_len"] for r in lr for e in r["events"] if e["type"] == "accept"]

        print(f"\n  {lang_name}:")
        print(f"    Top-1 accuracy:    {100*total_top1/n_words:.1f}%")
        print(f"    Top-3 accuracy:    {100*total_top3/n_words:.1f}%  (= acceptance rate)")
        print(f"    Top-5 accuracy:    {100*total_top5/n_words:.1f}%")
        print(f"    Total keystrokes:  {total_ks} (of {total_chars} chars)")
        print(f"    Simulated CSR:     {100*(total_chars - total_ks)/total_chars:.1f}%")
        print(f"    Acceptances:       {total_taps}/{n_words} words ({100*total_taps/n_words:.0f}%)")
        if prefix_lens:
            print(f"    Avg prefix before accept: {sum(prefix_lens)/len(prefix_lens):.1f} chars")
        print(f"      Completions:     {completions}")
        print(f"      Next-words:      {next_words}")
        print(f"      Manual:          {manuals}")
        if all_probs:
            print(f"    Avg prob (accepted): {sum(all_probs)/len(all_probs):.3f}")
            print(f"    Max prob:            {max(all_probs):.3f}")

    n_words = sum(r["n_words"] for r in all_results)
    total_ks = sum(r["keystrokes"] for r in all_results)
    total_chars = sum(r["total_chars"] for r in all_results)
    total_taps = sum(r["taps"] for r in all_results)
    total_top1 = sum(r["top1_accuracy"] * r["n_words"] for r in all_results)
    total_top3 = sum(r["top3_accuracy"] * r["n_words"] for r in all_results)
    total_top5 = sum(r["top5_accuracy"] * r["n_words"] for r in all_results)
    all_probs = [w[2] for r in all_results for w in r["completed_words"] if w[2] > 0]
    prefix_lens = [e["prefix_len"] for r in all_results for e in r["events"] if e["type"] == "accept"]

    print(f"\n  OVERALL:")
    print(f"    Top-1 accuracy:    {100*total_top1/n_words:.1f}%")
    print(f"    Top-3 accuracy:    {100*total_top3/n_words:.1f}%  (= acceptance rate)")
    print(f"    Top-5 accuracy:    {100*total_top5/n_words:.1f}%")
    print(f"    Total keystrokes:  {total_ks} (of {total_chars} chars)")
    print(f"    Simulated CSR:     {100*(total_chars - total_ks)/total_chars:.1f}%")
    print(f"    Acceptances:       {total_taps}/{n_words} words ({100*total_taps/n_words:.0f}%)")
    if prefix_lens:
        print(f"    Avg prefix before accept: {sum(prefix_lens)/len(prefix_lens):.1f} chars")
    if all_probs:
        print(f"    Avg prob (accepted): {sum(all_probs)/len(all_probs):.3f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Full results saved to {out_path}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
