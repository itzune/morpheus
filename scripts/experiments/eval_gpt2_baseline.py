#!/usr/bin/env python3
"""
eval_gpt2_baseline.py — Run eval tests against HiTZ/gpt2-eus-euscrawl (124M GPT-2).

Loads the HuggingFace model via transformers, then runs the same three-strategy
evaluation that eval.py runs for Morpheus Mamba-2.

Usage:
  pip install transformers torch sentencepiece
  python scripts/experiments/eval_gpt2_baseline.py \
    --targets eval/targets.json \
    --output-dir eval/gpt2-baseline

Strategy differences vs Mamba-2 eval:
  - GPT-2 uses AutoTokenizer, not SentencePiece directly
  - GPT-2 uses model.generate() or forward pass, not our custom get_top_k_predictions
  - CSR simulation uses greedy next-token generation instead of top-1 from logits
  - MorphAcc/Paradigm use full logits distribution (GPT-2 output logits have no bias toward
    morphology-respecting boundaries beyond what the training data captures)
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ──────────────────────────────────────────────────────────────────────
#  Model Loading
# ──────────────────────────────────────────────────────────────────────

def load_gpt2(device="cuda"):
    """Load HiTZ/gpt2-eus-euscrawl from HuggingFace."""
    model_name = "HiTZ/gpt2-eus-euscrawl"

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    model.eval()

    print(f"  Vocab size: {tokenizer.vocab_size}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params / 1e6:.0f}M")

    return model, tokenizer


def get_top_k_predictions_gpt2(model, tokenizer, text, device="cuda", k=10):
    """
    Get top-K token predictions from GPT-2 for next token after text.
    Returns same format as eval's get_top_k_predictions for compatibility.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # shape: [vocab_size]

    probs = torch.softmax(logits.float(), dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k)

    results = []
    for tid, prob in zip(topk_ids.tolist(), topk_probs.tolist()):
        piece = tokenizer.decode([tid])
        # Normalize: strip leading spaces like our SentencePiece ▁ handling
        decoded = piece
        import math as _math
        results.append({
            "id": tid,
            "piece": piece,
            "decoded": decoded,
            "logprob": _math.log(prob) if prob > 0 else float("-inf"),
            "prob": prob,
        })

    return results


# ──────────────────────────────────────────────────────────────────────
#  Strategy 1: Character Savings Rate (CSR)
# ──────────────────────────────────────────────────────────────────────

def evaluate_csr_gpt2(model, tokenizer, tests, device="cuda"):
    """
    CSR evaluation adapted for GPT-2.

    Key difference from Mamba: GPT-2's tokenizer separates tokens with
    spaces in decode(), so we need to handle that differently.
    """
    model.eval()
    results = []

    for t in tests:
        prompt = t["input"]
        target = t["target_completion"]
        category = t.get("category", "unknown")

        # Simulate typing
        typed_so_far = ""
        keystrokes_typed = 0
        chars_auto_completed = 0
        predictions_made = []
        finished = False

        while len(typed_so_far) < len(target) and not finished:
            remaining_target = target[len(typed_so_far):]

            # Current state: what user has typed so far after the prompt
            if typed_so_far:
                current_full_text = prompt + " " + typed_so_far
            else:
                current_full_text = prompt

            # Get model's greedy next-token prediction
            # GPT-2 generate with max_new_tokens=1 is greedy by default
            inputs = tokenizer(current_full_text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(logits, dim=-1).item()

            pred_text = tokenizer.decode([next_token_id])

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

            # Find longest prefix match
            # Normalize: strip leading space from GPT-2 decoded tokens
            pred_normalized = pred_text.lstrip()
            max_len = min(len(pred_normalized), len(remaining_target))
            match_len = 0
            for j in range(1, max_len + 1):
                if pred_normalized[:j] == remaining_target[:j]:
                    match_len = j
                else:
                    break

            predictions_made.append({
                "pos": len(typed_so_far),
                "typed": typed_so_far,
                "pred": pred_normalized[:40],
                "remaining": remaining_target,
                "match_len": match_len,
                "match": match_len > 0,
            })

            if match_len == 0:
                keystrokes_typed += 1
                typed_so_far += target[len(typed_so_far)]
            else:
                # Accept prediction: 1 keystroke (Tab) for match_len chars
                keystrokes_typed += 1
                typed_so_far += pred_normalized[:match_len]
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
#  Strategy 2: MorphAcc
# ──────────────────────────────────────────────────────────────────────

def evaluate_morphacc_gpt2(model, tokenizer, tests, device="cuda", k=5):
    """MorphAcc evaluation adapted for GPT-2."""
    model.eval()
    results = []

    for t in tests:
        prompt = t["input"]
        valid_suffixes = t.get("valid_suffixes", t.get("valid_continuations", []))
        category = t.get("category", "unknown")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

        probs = torch.softmax(logits.float(), dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k)

        best_rank = None
        boundary_prob_mass = 0.0
        top_preds = []

        for rank, (tid, prob) in enumerate(zip(topk_ids.tolist(), topk_probs.tolist()), start=1):
            decoded = tokenizer.decode([tid]).lstrip()
            top_preds.append({"rank": rank, "decoded": decoded, "prob": prob})

            for suffix in valid_suffixes:
                if decoded == suffix or decoded.startswith(suffix):
                    boundary_prob_mass += prob
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
#  Strategy 3: Paradigm
# ──────────────────────────────────────────────────────────────────────

def evaluate_paradigm_gpt2(model, tokenizer, roots, cases, device="cuda", k_values=(1, 3, 5)):
    """Paradigm evaluation adapted for GPT-2."""
    model.eval()
    results = []
    all_hits = {k: 0 for k in k_values}
    all_total = 0

    for root in roots:
        root_results = []

        for case_info in cases:
            case_name = case_info["case"]
            suffix = case_info["suffix"]
            description = case_info.get("description", "")

            # Get top-10 from GPT-2
            top_k_preds = get_top_k_predictions_gpt2(model, tokenizer, root, device, k=10)

            best_rank = None
            best_prob = 0.0
            for rank, pred in enumerate(top_k_preds, start=1):
                decoded = pred["decoded"].lstrip()
                if decoded == suffix or decoded.startswith(suffix):
                    best_rank = rank
                    best_prob = pred["prob"]
                    break

            # Full-distribution search if not in top-10
            if best_rank is None:
                inputs = tokenizer(root, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)

                suffix_ids = []
                for tid in range(min(len(probs), tokenizer.vocab_size)):
                    piece = tokenizer.decode([tid]).lstrip()
                    if piece == suffix or piece.startswith(suffix):
                        suffix_ids.append(tid)

                if suffix_ids:
                    suffix_probs = {tid: probs[tid].item() for tid in suffix_ids}
                    best_tid = max(suffix_probs, key=suffix_probs.get)
                    best_prob = suffix_probs[best_tid]
                    higher_count = (probs > best_prob).sum().item()
                    best_rank = higher_count + 1

            result = {
                "root": root,
                "case": case_name,
                "suffix": suffix,
                "description": description,
                "rank": best_rank,
                "prob": round(best_prob, 6) if best_prob else 0.0,
            }

            for k in k_values:
                result[f"hit@{k}"] = bool(best_rank and best_rank <= k)
                if result[f"hit@{k}"]:
                    all_hits[k] += 1

            all_total += 1
            root_results.append(result)

        results.append({"root": root, "cases": root_results})

    return results, all_hits, all_total


# ──────────────────────────────────────────────────────────────────────
#  Printing (reuse eval formatting)
# ──────────────────────────────────────────────────────────────────────

def summarize_csr(results, model_name="GPT-2"):
    print(f"\n{'='*80}")
    print(f"  Strategy 1: Character Savings Rate (CSR) — {model_name}")
    print(f"{'='*80}")
    print(f"{'Prompt':<35s} {'Target':<18s} {'Chars':>5s} {'Typed':>6s} {'Saved':>6s} {'CSR':>7s}")
    print("-" * 80)

    for r in results:
        print(f"{r['prompt'][:34]:<35s} {r['target']:<18s} {r['total_chars']:>5d} "
              f"{r['keystrokes_needed']:>6d} {r['keystrokes_saved']:>6d} {r['csr']:>7.2%}")

    macro_csr = sum(r["csr"] for r in results) / len(results) if results else 0
    total_saved = sum(r["keystrokes_saved"] for r in results)
    total_chars = sum(r["total_chars"] for r in results)
    micro_csr = 1.0 - ((total_chars - total_saved) / total_chars) if total_chars > 0 else 0

    print("-" * 80)
    print(f"{'AVERAGE':<35s} {'':<18s} {total_chars:>5d} {total_chars - total_saved:>6d} "
          f"{total_saved:>6d} {micro_csr:>7.2%} (micro)")
    print(f"  Per-test avg CSR (macro): {macro_csr:.2%}")

    cats = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"total": 0, "saved": 0, "csrs": []}
        cats[cat]["total"] += r["total_chars"]
        cats[cat]["saved"] += r["keystrokes_saved"]
        cats[cat]["csrs"].append(r["csr"])

    print(f"\n  By category:")
    for cat in sorted(cats):
        c = cats[cat]
        avg = sum(c["csrs"]) / len(c["csrs"])
        micro = 1.0 - ((c["total"] - c["saved"]) / c["total"]) if c["total"] > 0 else 0
        print(f"    {cat:<20s}: {len(c['csrs']):>2d} tests, avg CSR={avg:.2%}, micro CSR={micro:.2%}")

    return {
        "macro_csr": macro_csr,
        "micro_csr": micro_csr,
    }


def summarize_morphacc(results, k=5, model_name="GPT-2"):
    print(f"\n{'='*80}")
    print(f"  Strategy 2: MorphAcc@{k} — {model_name}")
    print(f"{'='*80}")

    for r in results:
        rank_str = f"#{r['best_rank']}" if r["best_rank"] else "N/A"
        status = "✓" if r["morphacc_hit"] else "✗"
        suffixes_str = ", ".join(r['valid_suffixes'][:4])
        if len(r['valid_suffixes']) > 4:
            suffixes_str += f" +{len(r['valid_suffixes']) - 4} more"
        print(f"{status} {r['prompt'][:39]:<40s} {suffixes_str:<30s} {rank_str:>9s}  {r['boundary_prob_mass']:>6.2%}")

    hits = sum(1 for r in results if r["morphacc_hit"])
    total = len(results)
    morphacc = hits / total if total > 0 else 0

    print(f"  MorphAcc@{k}: {hits}/{total} = {morphacc:.2%}")

    return {"morphacc": morphacc, "hits": hits, "total": total}


def summarize_paradigm(results, all_hits, all_total, k_values, model_name="GPT-2"):
    print(f"\n{'='*80}")
    print(f"  Strategy 3: Case Paradigm Completion — {model_name}")
    print(f"{'='*80}")

    for k in k_values:
        ratio = all_hits[k] / all_total if all_total > 0 else 0
        print(f"    Hit@{k}: {all_hits[k]}/{all_total} = {ratio:.2%}")

    # Per-case breakdown
    case_stats = {}
    for root_data in results:
        for cr in root_data["cases"]:
            case = cr["case"]
            if case not in case_stats:
                case_stats[case] = {"total": 0, "hits": {k: 0 for k in k_values}, "ranks": []}
            case_stats[case]["total"] += 1
            for k in k_values:
                if cr.get(f"hit@{k}"):
                    case_stats[case]["hits"][k] += 1
            if cr["rank"]:
                case_stats[case]["ranks"].append(cr["rank"])

    min_k = min(k_values) if k_values else 1
    for case in sorted(case_stats):
        cs = case_stats[case]
        hit_smallest = cs["hits"][min_k] / cs["total"] if cs["total"] > 0 else 0
        avg_rank = sum(cs["ranks"]) / len(cs["ranks"]) if cs["ranks"] else float("inf")
        print(f"    {case:<22s}: Hit@{min_k}={hit_smallest:.0%} ({cs['hits'][min_k]}/{cs['total']}), "
              f"avg rank={avg_rank:.0f}")

    return {"hit_at_k": {k: all_hits[k] / all_total for k in k_values}}


# ──────────────────────────────────────────────────────────────────────
#  Manual test: Basque prompts
# ──────────────────────────────────────────────────────────────────────

def run_manual_tests(model, tokenizer, device="cuda"):
    """Run some qualitative manual prompts for inspection."""
    print(f"\n{'='*80}")
    print(f"  Manual Inspection Tests")
    print(f"{'='*80}")

    tests = [
        # Morphology: noun + case
        "etxe",           # should continue with ra/tik/an/ko
        "mendi",          # should continue with ra/tik/an/ko
        "etxea",          # the house
        "etxeko",         # of the house
        "etxetik",        # from the house

        # Verb conjugation
        "Bihar goizean, nire etxera etorriko",    # I will come
        "Gaur arratsaldean, mendira joango",       # I will go
        "Euskal Herrian bizi",                     # I live in Basque Country
        "Atzo ikusi nuen pelikula",                # The movie I saw yesterday

        # Possessive
        "Nire izebaren",     # my aunt's
        "Gure herriko",      # our town's
        "Haurren",           # children's

        # General Basque
        "Kaixo, zer",         # Hello, what
        "Eskerrik asko",     # Thank you very much
        "Egun on",            # Good morning
        "Gaur eguraldi",     # Today the weather
    ]

    for prompt in tests:
        top5 = get_top_k_predictions_gpt2(model, tokenizer, prompt, device, k=5)
        preds_str = " | ".join(f"{p['decoded']} ({p['prob']:.3f})" for p in top5[:5])
        print(f"  {prompt:<30s} → {preds_str}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="eval_gpt2_baseline.py — Run eval against HiTZ/gpt2-eus-euscrawl"
    )
    parser.add_argument("--targets", default="eval/targets.json",
                        help="JSON file with v3 evaluation targets")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for JSON results")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--strategy", default="all",
                        help="Strategy: csr, morphacc, paradigm, manual, all")
    parser.add_argument("--k", default="1,3,5", help="K values for MorphAcc and Paradigm")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    k_values = tuple(int(k) for k in args.k.split(","))

    # Output dir
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"eval/gpt2-baseline/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_gpt2(device)

    all_summaries = {}

    # ── Manual tests ──
    if args.strategy in ("manual", "all"):
        run_manual_tests(model, tokenizer, device)

    # ── Load targets ──
    with open(args.targets) as f:
        targets_data = json.load(f)

    # ── Strategy 1: CSR ──
    if args.strategy in ("csr", "all"):
        csr_tests = None
        for s in targets_data.get("strategies", []):
            if s["name"] == "csr":
                csr_tests = s["tests"]
                break

        if csr_tests:
            print(f"\nRunning CSR evaluation ({len(csr_tests)} tests)...")
            t0 = time.time()
            csr_results = evaluate_csr_gpt2(model, tokenizer, csr_tests, device)
            csr_summary = summarize_csr(csr_results, model_name="GPT-2 eus-euscrawl")
            print(f"  Completed in {time.time() - t0:.1f}s")

            with open(os.path.join(out_dir, "csr_results.json"), "w") as f:
                json.dump({"summary": csr_summary, "details": csr_results}, f,
                          indent=2, ensure_ascii=False)
            all_summaries["csr"] = csr_summary

    # ── Strategy 2: MorphAcc ──
    if args.strategy in ("morphacc", "all"):
        morphacc_tests = None
        for s in targets_data.get("strategies", []):
            if s["name"] == "morphacc":
                morphacc_tests = s["tests"]
                break

        if morphacc_tests:
            print(f"\nRunning MorphAcc evaluation ({len(morphacc_tests)} tests, k={max(k_values)})...")
            t0 = time.time()
            morphacc_results = evaluate_morphacc_gpt2(model, tokenizer, morphacc_tests, device, k=max(k_values))
            morphacc_summary = summarize_morphacc(morphacc_results, k=max(k_values), model_name="GPT-2 eus-euscrawl")
            print(f"  Completed in {time.time() - t0:.1f}s")

            with open(os.path.join(out_dir, "morphacc_results.json"), "w") as f:
                json.dump({"summary": morphacc_summary, "details": morphacc_results}, f,
                          indent=2, ensure_ascii=False)
            all_summaries["morphacc"] = morphacc_summary

    # ── Strategy 3: Paradigm ──
    if args.strategy in ("paradigm", "all"):
        paradigm_config = None
        for s in targets_data.get("strategies", []):
            if s["name"] == "paradigm":
                paradigm_config = s
                break

        if paradigm_config:
            roots = paradigm_config["roots"]
            cases = paradigm_config["cases"]
            total_tests = len(roots) * len(cases)
            print(f"\nRunning Paradigm evaluation ({len(roots)} roots × {len(cases)} cases = {total_tests} tests)...")
            t0 = time.time()
            paradigm_results, all_hits, all_total = evaluate_paradigm_gpt2(
                model, tokenizer, roots, cases, device, k_values
            )
            paradigm_summary = summarize_paradigm(paradigm_results, all_hits, all_total, k_values,
                                                   model_name="GPT-2 eus-euscrawl")
            print(f"  Completed in {time.time() - t0:.1f}s")

            with open(os.path.join(out_dir, "paradigm_results.json"), "w") as f:
                json.dump({"summary": paradigm_summary, "details": paradigm_results}, f,
                          indent=2, ensure_ascii=False)
            all_summaries["paradigm"] = paradigm_summary

    # ── Final summary ──
    print(f"\n{'='*80}")
    print(f"  GPT-2 Baseline Evaluation Complete")
    print(f"{'='*80}")
    print(f"  Model: HiTZ/gpt2-eus-euscrawl (124M params)")
    print(f"  Results saved to: {out_dir}/")

    if "csr" in all_summaries:
        print(f"\n  CSR:  macro={all_summaries['csr']['macro_csr']:.2%}, "
              f"micro={all_summaries['csr']['micro_csr']:.2%}")
    if "morphacc" in all_summaries:
        print(f"  MorphAcc@{max(k_values)}: {all_summaries['morphacc']['morphacc']:.2%}")
    if "paradigm" in all_summaries:
        for k in k_values:
            print(f"  Paradigm Hit@{k}: {all_summaries['paradigm']['hit_at_k'][k]:.2%}")

    # Side-by-side comparison
    print(f"\n{'='*80}")
    print(f"  COMPARISON: Morpheus Mamba-2 (step 30K) vs GPT-2 eus-euscrawl")
    print(f"{'='*80}")
    print(f"  {'Metric':<30s} {'Morpheus 30K':>15s} {'GPT-2 eus':>15s}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Architecture':<30s} {'Mamba-2 91M':>15s} {'GPT-2 124M':>15s}")
    print(f"  {'Training tokens':<30s} {'4.77B':>15s} {'423M':>15s}")
    print(f"  {'Tokenizer':<30s} {'Unigram 4K':>15s} {'BPE 50K':>15s}")

    if "csr" in all_summaries:
        # NOTE: morph_csr=0.836 is from eval v1 (pre-fix, free-acceptance bug → inflated).
        # Do NOT compare directly. Re-run eval.py on step30k for a valid v2 number.
        morph_csr = 0.836
        print(f"  {'CSR (macro, v1!)':<30s} {morph_csr:>15.1%} {all_summaries['csr']['macro_csr']:>15.1%}")
    if "morphacc" in all_summaries:
        morph_morphacc = 0.20
        print(f"  {'MorphAcc@5':<30s} {morph_morphacc:>15.1%} {all_summaries['morphacc']['morphacc']:>15.1%}")
    if "paradigm" in all_summaries:
        morph_paradigm = 0.036
        for k in k_values:
            print(f"  {'Paradigm Hit@{k}':<30s} {morph_paradigm:>15.1%} {all_summaries['paradigm']['hit_at_k'][k]:>15.1%}")
            break  # just show Hit@1

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
