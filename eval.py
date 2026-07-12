#!/usr/bin/env python3
"""
eval.py — Three-Strategy Evaluation for Agglutinative Autocomplete (Basque)

Strategies (see docs/eval-strategies.md for research):
  1. CSR  — Character Savings Rate: simulate keystroke-by-keystroke typing
  2. MorphAcc — Morpheme Boundary Accuracy: does the model respect morpheme boundaries?
  3. Paradigm — Case Paradigm Completion: systematic coverage of Basque 14-case system

References:
  - Trnka & McCoy (2008). Evaluating Word Prediction: Framing Keystroke Savings. ACL.
  - Contreras (2026). QuechuaTok: Morphological Boundary Accuracy... arXiv:2606.23943.
  - Lane et al. (2022). Interactive Word Completion for Plains Cree. ACL.
"""

import argparse
import json
import os
import time
from datetime import datetime

import sentencepiece as spm
import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from src.eval_utils import evaluate_csr, evaluate_morphacc, evaluate_next_word_csr, get_next_token_logits, get_top_k_predictions, bootstrap_mean_ci


# ──────────────────────────────────────────────────────────────────────
#  Model Loading
# ──────────────────────────────────────────────────────────────────────

def _pad_vocab(vocab_size, multiple):
    """Pad vocab size to multiple for hardware alignment."""
    return ((vocab_size + multiple - 1) // multiple) * multiple


def build_model_from_checkpoint(checkpoint_path, device):
    """Load Mamba model from training checkpoint.

    Uses strict=True so any key/shape mismatch raises an error instead
    of silently dropping weights.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]

    # Pad vocab the same way train.py does
    pad_multiple = raw_cfg.get("pad_vocab_size_multiple", 16)
    vocab_size = raw_cfg.get("padded_vocab_size",
                             _pad_vocab(raw_cfg["vocab_size"], pad_multiple))

    config = MambaConfig(
        d_model=raw_cfg["d_model"],
        n_layer=raw_cfg["n_layer"],
        vocab_size=vocab_size,
        ssm_cfg={
            "layer": raw_cfg.get("ssm_layer", "Mamba2"),
            "d_state": raw_cfg["d_state"],
            "d_conv": raw_cfg["d_conv"],
            "expand": raw_cfg["expand"],
            "headdim": raw_cfg["headdim"],
            "chunk_size": raw_cfg.get("chunk_size", 256),
        },
        residual_in_fp32=raw_cfg.get("residual_in_fp32", True),
        fused_add_norm=raw_cfg.get("fused_add_norm", True),
        rms_norm=True,
    )

    model = MambaLMHeadModel(config, device=device, dtype=torch.bfloat16)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    del ckpt

    return model, raw_cfg


# ──────────────────────────────────────────────────────────────────────
#  Strategy 1: CSR — Summary
# ──────────────────────────────────────────────────────────────────────

def summarize_csr(results):
    """Print CSR summary table and return aggregate stats.

    macro_csr = per-test average (each test weighted equally)
    micro_csr = char-weighted (longer tests have more weight)
    """
    print(f"\n{'='*80}")
    print(f"  Strategy 1: Character Savings Rate (CSR)")
    print(f"  Higher = better. 1.0 = zero keystrokes needed, 0.0 = type everything.")
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

    # Bootstrap 95% CI on macro CSR
    macro_csrs = [r["csr"] for r in results]
    _, ci_lo, ci_hi = bootstrap_mean_ci(macro_csrs)

    print("-" * 80)
    print(f"{'AVERAGE':<35s} {'':<18s} {total_chars:>5d} {total_chars - total_saved:>6d} "
          f"{total_saved:>6d} {micro_csr:>7.2%} (micro)")
    print(f"  Per-test avg CSR (macro): {macro_csr:.2%}")
    print(f"  Bootstrap 95% CI:        [{ci_lo:.2%}, {ci_hi:.2%}]  (n={len(results)})")

    # By category
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
        macro = 1.0 - ((c["total"] - c["saved"]) / c["total"]) if c["total"] > 0 else 0
        print(f"    {cat:<20s}: {len(c['csrs']):>2d} tests, avg CSR={avg:.2%}, micro CSR={macro:.2%}")

    return {
        "macro_csr": macro_csr,
        "micro_csr": micro_csr,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_tests": len(results),
        "total_chars": total_chars,
        "total_saved": total_saved,
        "by_category": {cat: {"macro_csr": sum(c["csrs"]) / len(c["csrs"])} for cat, c in cats.items()},
    }


# ──────────────────────────────────────────────────────────────────────
#  Strategy 2: MorphAcc — Summary
# ──────────────────────────────────────────────────────────────────────

def summarize_morphacc(results, k=5):
    """Print MorphAcc summary."""
    print(f"\n{'='*80}")
    print(f"  Strategy 2: Morpheme Boundary Accuracy (MorphAcc)")
    print(f"  {len(results)} tests. Proportion where a valid suffix appears in top-{k}.")
    print(f"{'='*80}")
    print(f"{'Prompt':<40s} {'Valid suffixes':<30s} {'Best rank':>9s}  {'Mass':>6s}")
    print("-" * 90)

    for r in results:
        rank_str = f"#{r['best_rank']}" if r["best_rank"] else "N/A"
        status = "✓" if r["morphacc_hit"] else "✗"
        suffixes_str = ", ".join(r['valid_suffixes'][:4])
        if len(r['valid_suffixes']) > 4:
            suffixes_str += f" +{len(r['valid_suffixes']) - 4} more"
        print(f"{status} {r['prompt'][:39]:<40s} {suffixes_str:<30s} {rank_str:>9s}  {r['boundary_prob_mass']:>6.2%}")

    hits = sum(1 for r in results if r["morphacc_hit"])
    total = len(results)
    avg_mass = sum(r["boundary_prob_mass"] for r in results) / total if total > 0 else 0
    morphacc = hits / total if total > 0 else 0

    print("-" * 90)
    print(f"  MorphAcc@{k}: {hits}/{total} = {morphacc:.2%}  |  "
          f"Avg boundary prob mass: {avg_mass:.2%}")

    # Detail: show top predictions for each test
    print(f"\n  Top-{k} predictions detail:")
    for r in results:
        print(f"\n  {r['prompt']}")
        for p in r["top_predictions"][:k]:
            marker = " ← boundary" if any(
                p["decoded"] == s or p["decoded"].startswith(s)
                for s in r["valid_suffixes"]
            ) else ""
            print(f"    #{p['rank']}: {p['decoded']:<20s} prob={p['prob']:.4f}{marker}")

    return {
        "morphacc": morphacc,
        "hits": hits,
        "total": total,
        "avg_boundary_prob_mass": avg_mass,
    }


# ──────────────────────────────────────────────────────────────────────
#  Strategy 3: Case Paradigm Completion
# ──────────────────────────────────────────────────────────────────────
#
#  Systematic evaluation across Basque's case suffixes.
#  For each noun root + case pair, construct the prompt as the bare root
#  and check if the correct case suffix appears in the top-K predictions.

def evaluate_paradigm(model, sp, roots, cases, device="cuda", k_values=(1, 3, 5)):
    """Case Paradigm Completion evaluation.

    For each (root, case) pair, checks if the correct case suffix
    ranks in top-K predictions.

    Optimization: logits are computed once per root (not per case), and
    a piece→id map is precomputed once per call for full-vocab suffix search.
    """
    model.eval()
    results = []
    all_hits = {k: 0 for k in k_values}
    all_total = 0

    # Precompute piece → id map (once, not per missed case)
    piece_to_ids = {}
    for tid in range(sp.vocab_size()):
        piece = sp.id_to_piece(tid).replace("\u2581", " ").strip()
        if piece:
            piece_to_ids.setdefault(piece, []).append(tid)

    for root in roots:
        root_ids = sp.encode(root, out_type=int)
        if len(root_ids) > 1024:
            root_ids = root_ids[-1024:]
        root_results = []

        # Compute logits once per root, reuse for all cases
        logits = get_next_token_logits(model, root_ids, device)
        probs = torch.softmax(logits, dim=-1)

        for case_info in cases:
            case_name = case_info["case"]
            suffix = case_info["suffix"]
            description = case_info.get("description", "")

            # Get top-10 predictions (reuse pre-computed logits)
            top_k_preds = get_top_k_predictions(model, sp, root_ids, device, k=10, logits=logits)

            # Find the rank of the correct suffix in top-10
            best_rank = None
            best_prob = 0.0
            for rank, pred in enumerate(top_k_preds, start=1):
                decoded = pred["decoded"].lstrip()
                if decoded == suffix or decoded.startswith(suffix):
                    best_rank = rank
                    best_prob = pred["prob"]
                    break

            # If not in top-10, search full vocab using precomputed map
            if best_rank is None:
                matching_ids = []
                for piece, ids in piece_to_ids.items():
                    if piece == suffix or piece.startswith(suffix):
                        matching_ids.extend(ids)

                if matching_ids:
                    suffix_probs = {tid: probs[tid].item() for tid in matching_ids}
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
                if best_rank and best_rank <= k:
                    all_hits[k] += 1
                    result[f"hit@{k}"] = True
                else:
                    result[f"hit@{k}"] = False

            all_total += 1
            root_results.append(result)

        results.append({
            "root": root,
            "cases": root_results,
        })

    return results, all_hits, all_total


def summarize_paradigm(results, all_hits, all_total, k_values):
    """Print paradigm completion summary."""
    print(f"\n{'='*80}")
    print(f"  Strategy 3: Case Paradigm Completion")
    print(f"  Systematic evaluation across Basque's case system.")
    print(f"{'='*80}")

    # Per-root table
    print(f"\n{'Root':<10s}", end="")
    for root_data in results:
        for case_result in root_data["cases"]:
            case = case_result["case"]
            print(f" {case:<16s}", end="")
        break
    print()

    print(f"{'':<10s}", end="")
    for root_data in results:
        for case_result in root_data["cases"]:
            print(f" {'rank':>4s} {'prob':>6s}  ", end="")
        break
    print()

    print("-" * (10 + 26 * len(results[0]["cases"])))

    for root_data in results:
        root = root_data["root"]
        print(f"{root:<10s}", end="")
        for cr in root_data["cases"]:
            if cr["rank"]:
                print(f" #{cr['rank']:<4d} {cr['prob']:>6.4f}", end=" ")
            else:
                print(f" {'N/A':>4s} {'-':>6s}", end=" ")
        print()

    # Global stats
    print(f"\n  Global Hit@K:")
    for k in k_values:
        ratio = all_hits[k] / all_total if all_total > 0 else 0
        print(f"    Hit@{k}: {all_hits[k]}/{all_total} = {ratio:.2%}")

    # Per-case stats
    print(f"\n  Per-case accuracy:")
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

    for case in sorted(case_stats):
        cs = case_stats[case]
        min_k = min(k_values) if k_values else 1
        hit_smallest = cs["hits"][min_k] / cs["total"] if cs["total"] > 0 else 0
        avg_rank = sum(cs["ranks"]) / len(cs["ranks"]) if cs["ranks"] else float("inf")
        print(f"    {case:<22s}: Hit@{min_k}={hit_smallest:.0%} ({cs['hits'][min_k]}/{cs['total']}), "
              f"avg rank={avg_rank:.0f}")

    return {
        "hit_at_k": {k: all_hits[k] / all_total for k in k_values},
        "total": all_total,
        "by_case": {case: case_stats[case]["hits"][min(k_values)] / case_stats[case]["total"]
                     for case in case_stats},
    }


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="eval.py — Three-strategy evaluation for agglutinative autocomplete"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to training checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to SentencePiece model")
    parser.add_argument("--targets", default="eval/targets.json",
                        help="JSON file with evaluation targets")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for JSON results (default: eval/results/<timestamp>)")
    parser.add_argument("--strategy", default="all",
                        help="Strategy to run: csr, morphacc, paradigm, all")
    parser.add_argument("--k", default="1,3,5", help="K values for MorphAcc and Paradigm")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    k_values = tuple(int(k) for k in args.k.split(","))

    # Setup output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"eval/results/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, raw_cfg = build_model_from_checkpoint(args.checkpoint, device)
    print(f"  d_model={raw_cfg['d_model']}, n_layer={raw_cfg['n_layer']}, "
          f"vocab_size={raw_cfg.get('vocab_size', raw_cfg.get('padded_vocab_size'))}")

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"  Tokenizer vocab: {sp.vocab_size()}")

    # Load targets
    with open(args.targets) as f:
        targets_data = json.load(f)

    all_summaries = {}
    csr_tests = None  # initialized at function scope for Strategy 1b

    # ── Strategy 1: CSR ──
    if args.strategy in ("csr", "all"):
        for s in targets_data.get("strategies", []):
            if s["name"] == "csr":
                csr_tests = s["tests"]
                break

        if csr_tests:
            print(f"\nRunning CSR evaluation ({len(csr_tests)} tests)...")
            t0 = time.time()
            csr_results = evaluate_csr(model, sp, csr_tests, device)
            csr_summary = summarize_csr(csr_results)
            print(f"  Completed in {time.time() - t0:.1f}s")

            with open(os.path.join(out_dir, "csr_results.json"), "w") as f:
                json.dump({"summary": csr_summary, "details": csr_results}, f,
                          indent=2, ensure_ascii=False)
            all_summaries["csr"] = csr_summary

    # ── Strategy 1b: Next-Word CSR (demo-faithful keyboard simulation) ──
    if args.strategy in ("csr", "all"):
        if csr_tests:
            print(f"\nRunning Next-Word CSR evaluation ({len(csr_tests)} tests)...")
            t0 = time.time()
            nw_results = evaluate_next_word_csr(model, sp, csr_tests, device)
            nw_total_chars = sum(r["total_chars"] for r in nw_results)
            nw_total_ks = sum(r["keystrokes"] for r in nw_results)
            nw_total_words = sum(r["n_words"] for r in nw_results)
            nw_total_correct = sum(r["correct_words"] for r in nw_results)
            nw_total_accepts = sum(r["taps"] for r in nw_results)
            nw_all_probs = [p for r in nw_results for w, m, p in r["completed_words"] if p > 0]
            nw_prefix_lens = [e.get("prefix_len", 0) for r in nw_results for e in r["events"] if e["type"] == "accept"]
            nw_macro_csrs = [r["csr"] for r in nw_results]
            nw_point, nw_lo, nw_hi = bootstrap_mean_ci(nw_macro_csrs)
            nw_summary = {
                "nw_csr": round((nw_total_chars - nw_total_ks) / nw_total_chars, 4) if nw_total_chars > 0 else 0.0,
                "nw_csr_macro": round(nw_point, 4),
                "nw_csr_ci_lower": round(nw_lo, 4),
                "nw_csr_ci_upper": round(nw_hi, 4),
                "word_accuracy": round(nw_total_correct / nw_total_words, 4) if nw_total_words > 0 else 0.0,
                "acceptance_rate": round(nw_total_accepts / nw_total_words, 4) if nw_total_words > 0 else 0.0,
                "avg_prefix_before_accept": round(sum(nw_prefix_lens) / len(nw_prefix_lens), 2) if nw_prefix_lens else 0.0,
                "avg_confidence": round(sum(nw_all_probs) / len(nw_all_probs), 4) if nw_all_probs else 0.0,
                "n_tests": len(nw_results),
                "n_words": nw_total_words,
            }
            print(f"  Completed in {time.time() - t0:.1f}s")
            print(f"  NW-CSR: {nw_summary['nw_csr']:.3f} (macro {nw_summary['nw_csr_macro']:.3f}, CI [{nw_summary['nw_csr_ci_lower']:.3f}, {nw_summary['nw_csr_ci_upper']:.3f}])")
            print(f"  Word accuracy: {nw_summary['word_accuracy']:.3f}  Acceptance: {nw_summary['acceptance_rate']:.3f}")
            print(f"  Avg prefix before accept: {nw_summary['avg_prefix_before_accept']:.1f}  Avg confidence: {nw_summary['avg_confidence']:.3f}")

            with open(os.path.join(out_dir, "nw_csr_results.json"), "w") as f:
                json.dump({"summary": nw_summary, "details": nw_results}, f,
                          indent=2, ensure_ascii=False)
            all_summaries["nw_csr"] = nw_summary

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
            morphacc_results = evaluate_morphacc(model, sp, morphacc_tests, device, k=max(k_values))
            morphacc_summary = summarize_morphacc(morphacc_results, k=max(k_values))
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
            paradigm_results, all_hits, all_total = evaluate_paradigm(
                model, sp, roots, cases, device, k_values
            )
            paradigm_summary = summarize_paradigm(paradigm_results, all_hits, all_total, k_values)
            print(f"  Completed in {time.time() - t0:.1f}s")

            with open(os.path.join(out_dir, "paradigm_results.json"), "w") as f:
                json.dump({"summary": paradigm_summary, "details": paradigm_results}, f,
                          indent=2, ensure_ascii=False)
            all_summaries["paradigm"] = paradigm_summary

    # ── Final summary ──
    print(f"\n{'='*80}")
    print(f"  Evaluation Complete")
    print(f"{'='*80}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Results saved to: {out_dir}/")

    if "csr" in all_summaries:
        csr_s = all_summaries['csr']
        print(f"\n  CSR:  macro={csr_s['macro_csr']:.2%}, "
              f"micro={csr_s['micro_csr']:.2%}, "
              f"95% CI=[{csr_s.get('ci_lower',0):.2%}, {csr_s.get('ci_upper',0):.2%}] "
              f"(n={csr_s.get('n_tests',0)})")
    if "morphacc" in all_summaries:
        print(f"  MorphAcc@{max(k_values)}: {all_summaries['morphacc']['morphacc']:.2%}")
    if "paradigm" in all_summaries:
        for k in k_values:
            print(f"  Paradigm Hit@{k}: {all_summaries['paradigm']['hit_at_k'][k]:.2%}")

    # Save unified summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    return all_summaries


if __name__ == "__main__":
    main()
