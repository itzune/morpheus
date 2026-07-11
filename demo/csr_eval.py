#!/usr/bin/env python3
"""
CSR (Character Savings Rate) eval over the REAL corpus, via the demo API.

Replicates src/eval_utils.py::evaluate_csr's free-acceptance algorithm exactly,
but drives predictions through the demo HTTP endpoint instead of direct model
logits — so it measures the DEPLOYED product (filters included) and needs no GPU.

Algorithm (Trnka & McCoy 2008, free-acceptance; centralized in src/eval_utils.py):
  For each (prompt, target) pair, simulate keystroke-by-keystroke typing of
  `target`. At each position:
    - ask the model for its top-1 prediction (demo greedy, max_tokens=1)
    - pred = suggestion.lstrip()            # match evaluate_csr's lstrip
    - find longest prefix match vs remaining target
    - if match_len > 0: accept (1 keystroke/Tab), advance match_len chars
    - else: type 1 character (1 keystroke)
  CSR = 1 - keystrokes_needed / total_chars

Prompt text is built as `evaluate_csr` does:
    text = prompt                         if nothing typed yet
    text = prompt + " " + typed_so_far    once typing has started

Why CSR over this corpus:
  - It's the product metric (keystrokes saved), with partial credit (unlike the
    all-or-nothing first-word-match in real_corpus_eval).
  - Methodology-clean: a mechanical string-prefix match against gold text. The
    assistant makes ZERO Basque judgments — gold is correct by construction.
  - Comparable to the GPU evaluate_csr variant (same algorithm, no filters) for
    later, and to eval/gpt2-baseline.

Modes:
  --mode filtered  : deployed defaults (numeric + punctuation filters ON)
  --mode raw       : filters OFF — model's true output (comparable to GPU eval)
  --mode both      : run both over the same sentences; report the gap
                     (gap ≈ 0 would show filters are CSR-neutral, since junk
                     never matches gold anyway — confirmed or refuted here)

Usage:
    python3 demo/csr_eval.py --mode both --save
    python3 demo/csr_eval.py --mode filtered --count 100
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import urllib.parse
import urllib.request

# Reuse the real-corpus sentence loader (Wikipedia + Berria, clean prose)
from extract_real_prompts import load_sentences

OUT_DIR = Path("eval/demo-results")


def query(host, port, text, max_tokens, numeric_filter, punct_filter):
    url = f"http://{host}:{port}/api/autocomplete/greedy"
    params = urllib.parse.urlencode({
        "text": text, "max_tokens": max_tokens,
        "filter_numeric": str(numeric_filter).lower(),
        "filter_punctuation": str(punct_filter).lower(),
    })
    try:
        with urllib.request.urlopen(f"{url}?{params}", timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def longest_prefix_match(pred: str, remaining: str) -> int:
    max_len = min(len(pred), len(remaining))
    match_len = 0
    for j in range(1, max_len + 1):
        if pred[:j] == remaining[:j]:
            match_len = j
        else:
            break
    return match_len


def csr_one(prompt, target, host, port, numeric, punct, max_steps=120):
    """Run CSR on one (prompt, target). Returns dict with csr + trace stats."""
    typed = ""
    keystrokes = 0
    accepted = 0
    chars_auto = 0
    preds = 0
    steps = 0
    while len(typed) < len(target) and steps < max_steps:
        steps += 1
        remaining = target[len(typed):]
        text = prompt if not typed else prompt + " " + typed
        r = query(host, port, text, 1, numeric, punct)
        if "error" in r:
            keystrokes += 1
            typed += target[len(typed)]
            continue
        preds += 1
        pred = r.get("suggestion", "").lstrip()  # match evaluate_csr's lstrip
        if not pred:
            keystrokes += 1
            typed += target[len(typed)]
            continue
        ml = longest_prefix_match(pred, remaining)
        if ml == 0:
            keystrokes += 1
            typed += target[len(typed)]
        else:
            keystrokes += 1          # Tab
            typed += pred[:ml]
            accepted += 1
            chars_auto += ml
    total = len(target)
    csr = 1.0 - (keystrokes / total) if total > 0 else 0.0
    return {
        "prompt": prompt, "target": target, "total_chars": total,
        "keystrokes": keystrokes, "chars_saved": total - keystrokes,
        "csr": round(csr, 4), "predictions": preds, "accepted": accepted,
        "chars_auto": chars_auto,
    }


def build_pairs(count, seed=20260709, min_target_words=4, max_target_chars=70):
    """Build (prompt, target) pairs from real corpus sentences.

    Cut each clean sentence at its midpoint; target = second half (the gold
    continuation the user 'types'). Prefer targets of 4-10 words.
    """
    import random
    random.seed(seed)
    sents = load_sentences()  # list of (source, sentence), deduped
    # prefer longer sentences (more CSR signal), but cap target length
    cands = []
    for src, s in sents:
        words = s.split()
        if len(words) < 8:
            continue
        cut = len(words) // 2
        target = " ".join(words[cut:])
        if (min_target_words <= len(target.split()) <= 12
                and len(target) <= max_target_chars):
            prompt = " ".join(words[:cut])
            cands.append((src, prompt, target, s))
    random.shuffle(cands)
    return cands[:count]


def build_pairs_from_targets(targets_file):
    """Load CSR tests from eval/targets.json (input/target_completion).

    Uses the EXACT same test set as eval.py / eval_gpt2_baseline.py, so the
    demo-API CSR is directly comparable to the GPU f16 CSR (29.17%) and the
    GPT-2 baseline (0.486). This isolates quantization + demo decoding from
    test-set differences.
    """
    with open(targets_file) as f:
        td = json.load(f)
    csr_tests = None
    for strat in td.get("strategies", []):
        if strat["name"] == "csr":
            csr_tests = strat["tests"]
            break
    pairs = []
    for t in csr_tests:
        prompt = t["input"]
        target = t["target_completion"]
        cat = t.get("category", "targets_json")
        pairs.append((cat, prompt, target, prompt + " " + target))
    return pairs


def run_mode(pairs, host, port, numeric, punct, label):
    print(f"\n{'─'*72}\n  CSR mode: {label}\n{'─'*72}")
    results = []
    t0 = time.time()
    for i, (src, prompt, target, full) in enumerate(pairs, 1):
        r = csr_one(prompt, target, host, port, numeric, punct)
        r["source"] = src
        results.append(r)
        if i % 10 == 0 or i == len(pairs):
            macro = sum(x["csr"] for x in results) / len(results)
            elapsed = time.time() - t0
            print(f"  [{i:>3}/{len(pairs)}] macro CSR={macro:.3f}  "
                  f"({elapsed:.0f}s, {r['keystrokes']} keys / {r['total_chars']} ch)")
    # summary
    n = len(results)
    total_chars = sum(r["total_chars"] for r in results)
    total_saved = sum(r["chars_saved"] for r in results)
    macro = sum(r["csr"] for r in results) / n if n else 0
    micro = total_saved / total_chars if total_chars else 0
    accepted = sum(r["accepted"] for r in results)
    preds = sum(r["predictions"] for r in results)
    print(f"\n  {label} SUMMARY")
    print(f"    sentences           : {n}")
    print(f"    total target chars  : {total_chars}")
    print(f"    keystrokes saved    : {total_saved}")
    print(f"    macro CSR (per-sent): {macro:.4f}  ({macro*100:.2f}%)")
    print(f"    micro CSR (char-wt) : {micro:.4f}  ({micro*100:.2f}%)")
    print(f"    predictions made    : {preds}")
    print(f"    accepted (matched)  : {accepted}  ({accepted/preds*100:.1f}% of preds)" if preds else "    accepted: 0")
    # show any sentence with csr > 0
    wins = [r for r in results if r["csr"] > 0]
    if wins:
        print(f"    sentences with CSR>0: {len(wins)}")
        for r in wins[:5]:
            print(f"      csr={r['csr']:.2f}  '{r['prompt']}' → '{r['target']}'")
    else:
        print(f"    sentences with CSR>0: 0  (model saved no keystrokes)")
    return {"label": label, "results": results, "n": n,
            "macro_csr": round(macro, 4), "micro_csr": round(micro, 4),
            "total_chars": total_chars, "total_saved": total_saved,
            "predictions": preds, "accepted": accepted}


def main():
    p = argparse.ArgumentParser(description="Real-corpus CSR eval via demo API")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--count", type=int, default=100, help="number of sentences")
    p.add_argument("--targets-file", default=None, help="load CSR tests from eval/targets.json instead of real corpus")
    p.add_argument("--mode", choices=["filtered", "raw", "both"], default="both")
    p.add_argument("--save", action="store_true")
    p.add_argument("--model-label", default=None)
    args = p.parse_args()

    if args.targets_file:
        pairs = build_pairs_from_targets(args.targets_file)
        print(f"CSR eval — {len(pairs)} tests from {args.targets_file} (SAME set as GPU/GPT-2 eval)")
    else:
        pairs = build_pairs(args.count)
        print(f"CSR eval — {len(pairs)} real sentences (Wikipedia + Berria)")
    print(f"Algorithm: free-acceptance (Trnka & McCoy 2008), 1 token/step, "
          f"matches src/eval_utils.py::evaluate_csr")
    print(f"Mode: {args.mode}")

    modes = []
    if args.mode in ("filtered", "both"):
        # filter_numeric/filter_punctuation = True  → filter ACTIVE (deployed)
        modes.append(("filtered (deployed)", True, True))
    if args.mode in ("raw", "both"):
        # = False → filters OFF (raw model output, comparable to GPU eval)
        modes.append(("raw (no filters)", False, False))

    all_runs = []
    for label, numeric, punct in modes:
        run = run_mode(pairs, args.host, args.port, numeric, punct, label)
        all_runs.append(run)

    if len(all_runs) == 2:
        a, b = all_runs
        gap = abs(a["macro_csr"] - b["macro_csr"])
        print(f"\n{'='*72}")
        print(f"  FILTER IMPACT: {a['label']} macro={a['macro_csr']:.4f}  vs  "
              f"{b['label']} macro={b['macro_csr']:.4f}  |gap|={gap:.4f}")
        print(f"  (gap ≈ 0 ⇒ filters are CSR-neutral: stripped junk wouldn't")
        print(f"   match gold anyway. Filters affect UX cleanliness, not CSR.)")
        print(f"{'='*72}")

    if args.save:
        _save(args, pairs, all_runs)


def _save(args, pairs, all_runs):
    model_label = args.model_label or "unknown"
    model_loaded = False
    try:
        with urllib.request.urlopen(f"http://{args.host}:{args.port}/health", timeout=5) as r:
            h = json.loads(r.read())
            model_label = h.get("model", model_label)
            model_loaded = h.get("model_loaded", False)
    except Exception:
        pass
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.now().strftime("%Y-%m-%d")
    out_dir = OUT_DIR / f"{ts}_{model_label.replace('.gguf','')}_csr"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": ts, "date": date, "model": model_label,
        "model_loaded": model_loaded, "eval_type": "csr_real_corpus",
        "algorithm": "free-acceptance (Trnka & McCoy 2008), 1 token/step, matches src/eval_utils.py",
        "n_sentences": len(pairs),
        "runs": [{"label": r["label"], "macro_csr": r["macro_csr"],
                  "micro_csr": r["micro_csr"], "n": r["n"],
                  "total_chars": r["total_chars"], "total_saved": r["total_saved"],
                  "predictions": r["predictions"], "accepted": r["accepted"],
                  "results": r["results"]} for r in all_runs],
    }
    (out_dir / "results.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    md = [f"# CSR Eval (real corpus) — {model_label}", "",
          f"- **Date:** {date}", f"- **Model:** `{model_label}`",
          f"- **Sentences:** {len(pairs)} (Wikipedia + Berria)",
          f"- **Algorithm:** free-acceptance, 1 token/step (matches `evaluate_csr`)", ""]
    for r in all_runs:
        md += [f"## {r['label']}", "",
               f"- macro CSR: **{r['macro_csr']:.4f}** ({r['macro_csr']*100:.2f}%)",
               f"- micro CSR: {r['micro_csr']:.4f} ({r['micro_csr']*100:.2f}%)",
               f"- predictions: {r['predictions']}, accepted: {r['accepted']}", ""]
    if len(all_runs) == 2:
        a, b = all_runs
        md += ["## Filter impact", "",
               f"| mode | macro CSR | micro CSR | accepted |",
               f"|------|-----------|-----------|----------|",
               f"| {a['label']} | {a['macro_csr']:.4f} | {a['micro_csr']:.4f} | {a['accepted']} |",
               f"| {b['label']} | {b['macro_csr']:.4f} | {b['micro_csr']:.4f} | {b['accepted']} |",
               "", f"|gap| = {abs(a['macro_csr']-b['macro_csr']):.4f}", ""]
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"\n  ✓ Saved: {out_dir}/")


if __name__ == "__main__":
    main()
