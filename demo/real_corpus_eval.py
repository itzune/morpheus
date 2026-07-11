#!/usr/bin/env python3
"""
Real-corpus autocomplete eval.

Uses prompts cut from REAL Basque sentences (Wikipedia + Berria) — see
demo/extract_real_prompts.py — and compares the model's suggestion against
the GOLD continuation (the actual rest of the sentence).

This is the strongest qualitative signal we have: the expert can see, side
by side, what the model predicted vs. what a real speaker wrote.

Quantitative stats (rough signals, NOT correctness verdicts):
  - first_word_match : model's 1st word == gold's 1st word
  - prefix_on_track   : gold starts with the model's suggestion (or the
                        suggestion is a prefix of the gold) — model headed
                        the right way
  - any_overlap       : model shares >=1 of gold's first 3 words, in order
  - nonempty          : model produced something (not filtered to empty)

Judgment of Basque correctness/acceptability is LEFT TO THE EXPERT. The
script only runs queries and reports what the model produced.

Usage:
    python3 demo/real_corpus_eval.py
    python3 demo/real_corpus_eval.py --tokens 8 --save
"""
import argparse
import json
import re as _re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
import urllib.parse
import urllib.request

PROMPTS_FILE = Path("demo/real_prompts.json")


def norm(w: str) -> str:
    """lowercase, strip punctuation/whitespace for loose word comparison."""
    w = w.strip().lower()
    w = "".join(c for c in w if c.isalnum() or c in "-'")
    return w


def query(host, port, text, max_tokens, numeric_filter=True, punct_filter=True):
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


def match_stats(suggestion: str, gold: str):
    s_words = [norm(w) for w in suggestion.split() if norm(w)]
    g_words = [norm(w) for w in gold.split() if norm(w)]
    if not s_words or not g_words:
        return {"first_word_match": False, "prefix_on_track": False,
                "any_overlap": False}
    first_word_match = s_words[0] == g_words[0]
    # suggestion is a prefix of gold (model on track, just shorter/longer)
    s_joined = " ".join(s_words)
    g_joined = " ".join(g_words)
    prefix_on_track = g_joined.startswith(s_joined) or s_joined.startswith(g_joined)
    # any of model's words appear in gold's first 3, preserving order
    g_head = g_words[:3]
    any_overlap = any(sw in g_head for sw in s_words[:3])
    return {"first_word_match": first_word_match,
            "prefix_on_track": prefix_on_track,
            "any_overlap": any_overlap}


def main():
    p = argparse.ArgumentParser(description="Real-corpus autocomplete eval")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--tokens", type=int, default=8, help="max tokens to generate")
    p.add_argument("--trailing-space", action="store_true",
                   help="append a space to each prompt (next-WORD prediction mode)")
    p.add_argument("--no-numeric-filter", action="store_true",
                   help="show raw model output incl. digit artifacts (diagnostic)")
    p.add_argument("--no-punct-filter", action="store_true",
                   help="disable punctuation cleanup (raw)")
    p.add_argument("--save", action="store_true")
    p.add_argument("--model-label", default=None)
    args = p.parse_args()

    prompts = json.loads(PROMPTS_FILE.read_text())
    mode = "next-WORD (trailing space)" if args.trailing_space else "word-COMPLETION (no trailing space)"
    filt = "RAW (no filters)" if (args.no_numeric_filter or args.no_punct_filter) else "filtered (deployed defaults)"
    print(f"Real-corpus eval — {len(prompts)} prompts from {PROMPTS_FILE}")
    print(f"Sources: Wikipedia + Berria (clean prose). Tokens={args.tokens}")
    print(f"Mode: {mode}   |   Output: {filt}")
    print(f"Criterion: does the model predict the real next word(s)?\n")

    results = []
    for i, pr in enumerate(prompts):
        ptxt = pr["prompt"] + (" " if args.trailing_space else "")
        r = query(args.host, args.port, ptxt, args.tokens,
                  numeric_filter=not args.no_numeric_filter,
                  punct_filter=not args.no_punct_filter)
        if "error" in r:
            print(f"  [ERROR] {pr['prompt']!r}: {r['error']}")
            results.append({**pr, "suggestion": "", "confidence": 0,
                            "latency_ms": 0, "error": r["error"], **match_stats("", pr["gold"])})
            continue
        sugg = r.get("suggestion", "")
        conf = r.get("confidence", 0)
        lat = r.get("latency_ms", 0)
        ms = match_stats(sugg, pr["gold"])
        has_digit = bool(_re.search(r"\d", sugg)) if sugg else False
        rec = {**pr, "prompt_sent": ptxt, "suggestion": sugg, "confidence": conf,
               "latency_ms": lat, "digit_artifact": has_digit, **ms}
        results.append(rec)
        # display: prompt | model(green) | gold(dim)
        dig = " \033[31m[DIGIT]\033[0m" if has_digit else ""
        full = f"  {pr['prompt']}\033[32m{sugg}\033[0m  \033[2m[gold: {pr['gold']}]\033[0m{dig}"
        tag = "✓" if ms["first_word_match"] else ("~" if ms["any_overlap"] else " ")
        print(f"{tag} {full}")
        print(f"     conf={conf:.2f} {lat:.0f}ms  ({pr['source']})")

    # summary
    n = len(results)
    nonempty = sum(1 for r in results if r["suggestion"].strip())
    fwm = sum(1 for r in results if r["first_word_match"])
    pot = sum(1 for r in results if r["prefix_on_track"])
    ov = sum(1 for r in results if r["any_overlap"])
    digits = sum(1 for r in results if r["digit_artifact"])
    avg_conf = sum(r["confidence"] for r in results) / n if n else 0
    print(f"\n{'='*72}")
    print("SUMMARY (rough signals — expert judges acceptability)")
    print(f"{'='*72}")
    print(f"  mode                  : {mode}")
    print(f"  output                : {filt}")
    print(f"  prompts               : {n}")
    print(f"  nonempty suggestions  : {nonempty} ({nonempty/n*100:.0f}%)")
    print(f"  digit artifacts       : {digits} ({digits/n*100:.0f}%)  ← contamination signal")
    print(f"  first-word exact match: {fwm} ({fwm/n*100:.0f}%)  ← next-word prediction")
    print(f"  any overlap (1 of 3)  : {ov} ({ov/n*100:.0f}%)")
    print(f"  prefix on track       : {pot} ({pot/n*100:.0f}%)")
    print(f"  avg confidence        : {avg_conf:.3f}")
    # per-bucket
    print(f"\n  Per bucket:")
    for b in sorted(set(r["bucket"] for r in results)):
        br = [r for r in results if r["bucket"] == b]
        bf = sum(1 for r in br if r["first_word_match"])
        bo = sum(1 for r in br if r["any_overlap"])
        print(f"    {b:<14} n={len(br):>2}  1st-match={bf:>2}  overlap={bo:>2}")
    print(f"\n  Expert: which suggestions are acceptable Basque continuations?")

    if args.save:
        _save(args, results, n, nonempty, fwm, pot, ov, avg_conf, digits, mode, filt)


def _save(args, results, n, nonempty, fwm, pot, ov, avg_conf, digits=0, mode="", filt=""):
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
    out_dir = Path("eval/demo-results") / f"{ts}_{model_label.replace('.gguf','')}_realcorpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": ts, "date": date, "model": model_label,
        "model_loaded": model_loaded, "eval_type": "real_corpus",
        "max_tokens": args.tokens, "prompts_file": str(PROMPTS_FILE),
        "total_prompts": n,
        "stats": {
            "mode": mode, "output": filt,
            "nonempty": nonempty, "nonempty_pct": round(nonempty/n*100, 1),
            "digit_artifacts": digits, "digit_artifact_pct": round(digits/n*100, 1),
            "first_word_match": fwm, "first_word_match_pct": round(fwm/n*100, 1),
            "any_overlap": ov, "any_overlap_pct": round(ov/n*100, 1),
            "prefix_on_track": pot, "prefix_on_track_pct": round(pot/n*100, 1),
            "avg_confidence": round(avg_conf, 4),
        },
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    md = [f"# Real-Corpus Autocomplete Eval — {model_label}", "",
          f"- **Date:** {date}", f"- **Model:** `{model_label}`",
          f"- **Mode:** {mode}", f"- **Output:** {filt}",
          f"- **Tokens:** {args.tokens}", f"- **Prompts:** {n} (Wikipedia + Berria)",
          f"- **digit artifacts:** {digits}/{n} ({digits/n*100:.0f}%)",
          f"- **first-word exact match:** {fwm}/{n} ({fwm/n*100:.0f}%)",
          f"- **any overlap:** {ov}/{n} ({ov/n*100:.0f}%)",
          f"- **prefix on track:** {pot}/{n} ({pot/n*100:.0f}%)",
          f"- **avg confidence:** {avg_conf:.3f}", "",
          "## Results (prompt + MODEL + [gold])", "",
          "| ✓ | Prompt + suggestion | Gold | Conf |", "|---|---|---|---|"]
    for r in results:
        tag = "✓" if r["first_word_match"] else ("~" if r["any_overlap"] else " ")
        md.append(f"| {tag} | `{r['prompt']}`**{r['suggestion']}** | {r['gold']} | {r['confidence']:.2f} |")
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"\n  ✓ Saved: {out_dir}/")


if __name__ == "__main__":
    main()
