#!/usr/bin/env python3
"""Replay logged keyboard completions against different model checkpoints.

Reads the completion log (logs/completions.jsonl) produced by the demo's
/api/log endpoint, then for each checkpoint reloads the model and checks
whether the user-accepted word would have appeared in the top-k candidates.

This lets us scientifically compare checkpoints on *real user behavior* rather
than synthetic prompts — e.g. "did step_0054000 lose 'Kaixo' that step_0032000
had?"

Usage:
    # Compare all checkpoints against the logged acceptances:
    python scripts/replay_completions.py

    # Compare specific checkpoints:
    python scripts/replay_completions.py --models step_0032000.Q4_K_M step_0054000.Q4_K_M

    # Use a specific log file:
    python scripts/replay_completions.py --log logs/completions.jsonl

    # Custom demo server URL:
    python scripts/replay_completions.py --host http://localhost:9090

Output:
    Per-checkpoint table: top-1 hit, top-3 hit, avg prob of accepted word,
    and a per-case breakdown showing where checkpoints disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / "logs" / "completions.jsonl"
EXPORTS_DIR = REPO_ROOT / "exports"


def load_acceptances(log_file: Path) -> list[dict]:
    """Load 'accept' events from the completion log."""
    if not log_file.exists():
        print(f"Error: log file not found: {log_file}")
        sys.exit(1)

    entries = []
    for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") != "accept":
            continue
        accepted = entry.get("accepted", {})
        if not accepted.get("text"):
            continue
        # Skip pure punctuation accepts (not meaningful for word prediction)
        if accepted.get("is_punct"):
            continue
        entries.append(entry)

    # Deduplicate by (context, accepted_text) — user may accept the same
    # completion multiple times in a session.
    seen = set()
    unique = []
    for e in entries:
        key = (e.get("context", ""), e["accepted"]["text"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


def discover_models() -> list[str]:
    """Find all Q4_K_M GGUF checkpoints in exports/.

    Returns names without .gguf extension (e.g. 'step_0032000.Q4_K_M').
    """
    models = sorted(EXPORTS_DIR.glob("step_*.Q4_K_M.gguf"))
    return [m.name.replace(".gguf", "") for m in models]


def reload_model(host: str, model_name: str, timeout: float = 60.0) -> bool:
    """Hot-reload the demo server to a different checkpoint."""
    # Normalize: accept "step_0032000", "step_0032000.Q4_K_M",
    # or "step_0032000.Q4_K_M.gguf" — all map to the .gguf filename.
    if not model_name.endswith(".gguf"):
        if model_name.endswith(".Q4_K_M"):
            model_name += ".gguf"
        else:
            model_name += ".Q4_K_M.gguf"
    try:
        r = httpx.post(
            f"{host}/api/model/reload",
            params={"model": model_name},
            timeout=timeout,
        )
        data = r.json()
        if data.get("status") == "ok":
            print(f"  ✓ Reloaded to {data.get('current')}")
            # Wait for llama-server to be ready
            for _ in range(30):
                try:
                    h = httpx.get(f"{host}/health", timeout=5.0)
                    if h.json().get("model_loaded"):
                        return True
                except Exception:
                    pass
                time.sleep(1)
            print("  ⚠ Model reloaded but health check failed")
            return False
        else:
            print(f"  ✗ Reload failed: {data}")
            return False
    except Exception as e:
        print(f"  ✗ Reload error: {e}")
        return False


def query_candidates(host: str, context: str, timeout: float = 30.0) -> list[dict]:
    """Query the keyboard autocomplete API for a given context."""
    try:
        r = httpx.get(
            f"{host}/api/autocomplete/keyboard",
            params={"text": context, "top_k": 5},
            timeout=timeout,
        )
        data = r.json()
        return data.get("candidates", [])
    except Exception as e:
        print(f"    Query error for '{context[:40]}...': {e}")
        return []


def evaluate_checkpoint(
    host: str, model_name: str, acceptances: list[dict], top_k: int = 5
) -> dict:
    """Evaluate a single checkpoint against all acceptances."""
    results = []
    for acc in acceptances:
        context = acc.get("smart_context") or acc.get("context", "")
        accepted_text = acc["accepted"]["text"]
        accepted_prob_original = acc["accepted"].get("prob")

        candidates = query_candidates(host, context)
        candidate_texts = [c["text"] for c in candidates[:top_k]]

        # Check if accepted word appears in top-k
        rank = None
        for i, ct in enumerate(candidate_texts):
            if ct == accepted_text:
                rank = i
                break

        # Also check partial match (accepted word might be a prefix extension)
        # e.g. accepted "Kaixo" but candidate is "Kaixaoko" — still a hit
        # because the model was on the right track. We report exact match
        # separately from fuzzy match.
        fuzzy_rank = None
        if rank is None:
            for i, ct in enumerate(candidate_texts):
                if accepted_text in ct or ct in accepted_text:
                    fuzzy_rank = i
                    break

        # Find the prob of the accepted word in this checkpoint's candidates
        accepted_prob_new = None
        for c in candidates:
            if c["text"] == accepted_text:
                accepted_prob_new = c.get("prob")
                break

        results.append({
            "context": context,
            "accepted_text": accepted_text,
            "original_prob": accepted_prob_original,
            "new_rank": rank,                    # 0-indexed, None = not found
            "fuzzy_rank": fuzzy_rank,
            "new_prob": accepted_prob_new,
            "candidates": candidate_texts,
        })

    n = len(results)
    top1_exact = sum(1 for r in results if r["new_rank"] == 0)
    top3_exact = sum(1 for r in results if r["new_rank"] is not None and r["new_rank"] < 3)
    topk_exact = sum(1 for r in results if r["new_rank"] is not None)
    topk_fuzzy = sum(1 for r in results if r["new_rank"] is not None or r["fuzzy_rank"] is not None)
    probs = [r["new_prob"] for r in results if r["new_prob"] is not None]

    return {
        "model": model_name,
        "n": n,
        "top1_exact": top1_exact,
        "top3_exact": top3_exact,
        "topk_exact": topk_exact,
        "topk_fuzzy": topk_fuzzy,
        "top1_pct": 100 * top1_exact / n if n else 0,
        "top3_pct": 100 * top3_exact / n if n else 0,
        "topk_pct": 100 * topk_exact / n if n else 0,
        "fuzzy_pct": 100 * topk_fuzzy / n if n else 0,
        "avg_prob": sum(probs) / len(probs) if probs else 0,
        "results": results,
    }


def print_comparison(all_results: list[dict]):
    """Print a comparison table across checkpoints."""
    print("\n" + "=" * 80)
    print("COMPLETION REPLAY COMPARISON")
    print("=" * 80)
    print(f"{'Model':<35} {'N':>4} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'Fuzzy':>8} {'AvgP':>8}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['model']:<35} {r['n']:>4} "
            f"{r['top1_pct']:>6.1f}% {r['top3_pct']:>6.1f}% "
            f"{r['topk_pct']:>6.1f}% {r['fuzzy_pct']:>6.1f}% "
            f"{r['avg_prob']:>6.3f}"
        )

    # Show disagreements: cases where checkpoints differ
    if len(all_results) >= 2:
        print("\n" + "=" * 80)
        print("DISAGREEMENTS (cases where checkpoints differ)")
        print("=" * 80)
        n_cases = len(all_results[0]["results"])
        for i in range(n_cases):
            ranks = [r["results"][i]["new_rank"] for r in all_results]
            # Only show if there's disagreement
            if len(set(ranks)) > 1:
                ctx = all_results[0]["results"][i]["context"]
                accepted = all_results[0]["results"][i]["accepted_text"]
                print(f"\n  Context: {repr(ctx[:60])}")
                print(f"  Accepted: {repr(accepted)}")
                for r in all_results:
                    res = r["results"][i]
                    rank_str = f"#{res['new_rank']}" if res["new_rank"] is not None else "MISS"
                    prob_str = f"p={res['new_prob']:.3f}" if res["new_prob"] else "p=-"
                    cands = ", ".join(res["candidates"][:3])
                    print(f"    {r['model']:<35} {rank_str:>5} {prob_str:>8}  [{cands}]")


def main():
    parser = argparse.ArgumentParser(
        description="Replay logged completions against different checkpoints"
    )
    parser.add_argument(
        "--log", type=Path, default=DEFAULT_LOG,
        help=f"Path to completions log (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--host", default="http://localhost:9090",
        help="Demo server URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Checkpoints to evaluate (default: all step_*.Q4_K_M.gguf in exports/)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top-k cutoff for hit rate (default: 5)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save detailed results as JSON to this path",
    )
    args = parser.parse_args()

    # Load acceptances
    acceptances = load_acceptances(args.log)
    print(f"Loaded {len(acceptances)} unique acceptances from {args.log}")
    if not acceptances:
        print("No acceptances found. Use the keyboard demo and accept some completions first.")
        sys.exit(0)

    print("\nAcceptances:")
    for a in acceptances:
        ctx = a.get("context", "")[:50]
        print(f"  {repr(ctx):52s} → {repr(a['accepted']['text'])}")

    # Discover models
    if args.models:
        models = args.models
    else:
        models = discover_models()
    if not models:
        print("No models found in exports/")
        sys.exit(1)

    print(f"\nEvaluating {len(models)} checkpoints: {', '.join(models)}")

    # Evaluate each checkpoint
    all_results = []
    for model_name in models:
        print(f"\n{'─' * 60}")
        print(f"Loading {model_name}...")
        if not reload_model(args.host, model_name):
            print(f"  Skipping {model_name} (failed to load)")
            continue
        time.sleep(2)  # extra warmup
        result = evaluate_checkpoint(args.host, model_name, acceptances, args.top_k)
        all_results.append(result)
        print(f"  Top-1: {result['top1_pct']:.1f}%  Top-3: {result['top3_pct']:.1f}%  "
              f"Top-{args.top_k}: {result['topk_pct']:.1f}%  Avg prob: {result['avg_prob']:.3f}")

    if len(all_results) < 1:
        print("\nNo checkpoints evaluated successfully.")
        sys.exit(1)

    # Print comparison
    print_comparison(all_results)

    # Save detailed results
    if args.output:
        args.output.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
