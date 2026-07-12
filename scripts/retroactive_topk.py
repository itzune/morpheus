#!/usr/bin/env python3
"""
Retroactively compute Top-1/Top-3/Top-5 accuracy from stored simulation events.

The old GGUF eval (eval/gguf_nw_csr/) was run with the pre-Top-K version of
simulate_typing.py. However, the stored events contain the full `fresh` (5
candidates) and `displayed` (3 candidates) lists at every check point.

This script replays those events, reconstructs which target word was active
at each check, and computes Top-K metrics — without re-running the eval.

Validation: the reconstructed completed_words list should match the stored one.
"""
import json
import sys
from pathlib import Path

PUNCT = set(".,!?;:()[]{}")


def clean_word(w):
    """Lowercase and strip punctuation for comparison."""
    return w.lower().strip(".,!?;:()[]{}\"'")


def replay_sentence(result):
    """
    Replay events from one sentence simulation to compute Top-K metrics.

    The original simulate_typing.py flow (which generated these events):
      1. CHECK (query server, track candidates)
      2. If target in displayed[:3] → ACCEPT (next event)
      3. If char_idx >= len(target) and no match → manual complete
      4. Else → KEYSTROKE (type next char, char_idx++)

    Key insight: manual completion happens at CHECK time, not KEYSTROKE time.
    A word can be accepted even after being fully typed (the CHECK after the
    last keystroke may find the target in displayed candidates).

    Returns (top1_count, top3_count, top5_count, n_words, reconstructed_completed).
    """
    words_target = result["words_target"]
    events = result["events"]

    word_idx = 0
    char_idx = 0  # non-space chars typed for current word
    was_top1 = False
    was_top3 = False
    was_top5 = False

    top1_count = 0
    top3_count = 0
    top5_count = 0
    reconstructed = []

    def tally_and_advance(method):
        nonlocal word_idx, char_idx, was_top1, was_top3, was_top5
        nonlocal top1_count, top3_count, top5_count
        if was_top1:
            top1_count += 1
        if was_top3:
            top3_count += 1
        if was_top5:
            top5_count += 1
        was_top1 = was_top3 = was_top5 = False
        target = words_target[word_idx] if word_idx < len(words_target) else "?"
        reconstructed.append([target, method, 0.0])
        word_idx += 1
        char_idx = 0

    for event in events:
        if event["type"] == "check":
            displayed = event.get("displayed", [])
            fresh = event.get("fresh", [])

            # Process this check for the current word (and possibly next words
            # if manual completions chain)
            while word_idx < len(words_target):
                target = words_target[word_idx]
                target_c = clean_word(target)
                if not target_c:
                    break

                # Track Top-K at this check
                if displayed:
                    top1_cand = clean_word(displayed[0]["text"])
                    if top1_cand == target_c and len(top1_cand) > 0:
                        was_top1 = True
                for c in displayed[:3]:
                    if clean_word(c["text"]) == target_c and len(clean_word(c["text"])) > 0:
                        was_top3 = True
                        break
                for c in fresh[:5]:
                    if clean_word(c["text"]) == target_c and len(clean_word(c["text"])) > 0:
                        was_top5 = True
                        break

                # Is target in displayed? → will be accepted (wait for ACCEPT)
                in_displayed = any(
                    clean_word(c["text"]) == target_c and len(clean_word(c["text"])) > 0
                    for c in displayed[:3]
                )
                if in_displayed:
                    break  # wait for ACCEPT event

                # All chars typed and no match? → manual completion
                if char_idx >= len(target):
                    tally_and_advance("manual")
                    continue  # re-process this check for the next word

                break  # normal: wait for keystrokes

        elif event["type"] == "accept":
            tally_and_advance("completion")

        elif event["type"] == "keystroke":
            if event["char"] != " ":
                char_idx += 1

    n_words = len(words_target)
    return top1_count, top3_count, top5_count, n_words, reconstructed


def main():
    files = sorted(Path("eval/gguf_nw_csr").glob("step_*_gguf_results.json"))

    if not files:
        print("No GGUF result files found in eval/gguf_nw_csr/")
        sys.exit(1)

    print("=" * 80)
    print("  Retroactive Top-K Computation from GGUF Simulation Events")
    print("=" * 80)
    print()

    all_results = []

    for fpath in files:
        step = fpath.stem.replace("_gguf_results", "")
        results = json.load(open(fpath))

        total_top1 = 0
        total_top3 = 0
        total_top5 = 0
        total_words = 0
        total_taps = 0
        total_ks = 0
        total_chars = 0
        all_probs = []
        prefix_lens = []
        validation_ok = True

        for r in results:
            top1, top3, top5, n_words, reconstructed = replay_sentence(r)

            # Validate reconstruction against stored completed_words
            # (use clean_word for comparison; treat next_word==completion)
            stored = r["completed_words"]
            if len(reconstructed) != len(stored):
                validation_ok = False
            else:
                for rec, sto in zip(reconstructed, stored):
                    if clean_word(rec[0]) != clean_word(sto[0]):
                        validation_ok = False
                        break
                    # Both 'completion' and 'next_word' are acceptances
                    if rec[1] == "manual" and sto[1] != "manual":
                        validation_ok = False
                        break
                    if rec[1] != "manual" and sto[1] == "manual":
                        validation_ok = False
                        break

            total_top1 += top1
            total_top3 += top3
            total_top5 += top5
            total_words += n_words
            total_taps += r["taps"]
            total_ks += r["keystrokes"]
            total_chars += r["total_chars"]

            for w in r["completed_words"]:
                if w[2] > 0:
                    all_probs.append(w[2])

            for e in r["events"]:
                if e["type"] == "accept" and "prefix_len" in e:
                    prefix_lens.append(e["prefix_len"])

        csr = (total_chars - total_ks) / total_chars if total_chars > 0 else 0
        acceptance = total_taps / total_words if total_words > 0 else 0
        conf = sum(all_probs) / len(all_probs) if all_probs else 0
        avg_prefix = sum(prefix_lens) / len(prefix_lens) if prefix_lens else 0

        result = {
            "checkpoint": step,
            "n_words": total_words,
            "nw_csr": round(csr, 4),
            "top1_accuracy": round(total_top1 / total_words, 4) if total_words else 0,
            "top3_accuracy": round(total_top3 / total_words, 4) if total_words else 0,
            "top5_accuracy": round(total_top5 / total_words, 4) if total_words else 0,
            "acceptance_rate": round(acceptance, 4),
            "avg_confidence": round(conf, 4),
            "avg_prefix": round(avg_prefix, 1),
            "validation_ok": validation_ok,
        }
        all_results.append(result)

        val_str = "✓" if validation_ok else "✗ MISMATCH"
        print(f"  {step}:")
        print(f"    NW-CSR:      {csr:.3f}")
        print(f"    Top-1 Acc:   {result['top1_accuracy']:.3f}")
        print(f"    Top-3 Acc:   {result['top3_accuracy']:.3f}  (= acceptance)")
        print(f"    Top-5 Acc:   {result['top5_accuracy']:.3f}")
        print(f"    Acceptance:  {acceptance:.3f} ({total_taps}/{total_words})")
        print(f"    Avg Prefix:  {avg_prefix:.1f} chars")
        print(f"    Confidence:  {conf:.3f}")
        print(f"    Validation:  {val_str}")
        print()

    # Summary table
    print("=" * 80)
    print("  SUMMARY: GGUF Top-K Across Checkpoints")
    print("=" * 80)
    print()
    header = f"  {'Checkpoint':<20s} {'NW-CSR':>7s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s} {'Conf':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        print(f"  {r['checkpoint']:<20s} {r['nw_csr']:>7.3f} {r['top1_accuracy']:>7.3f} {r['top3_accuracy']:>7.3f} {r['top5_accuracy']:>7.3f} {r['avg_confidence']:>7.3f}")

    print()
    print("  Trajectory (32K → 74K):")
    if len(all_results) >= 3:
        r0, r1, r2 = all_results[0], all_results[1], all_results[2]
        for metric in ["nw_csr", "top1_accuracy", "top3_accuracy", "top5_accuracy", "avg_confidence"]:
            v0, v1, v2 = r0[metric], r1[metric], r2[metric]
            trend = "↑" if v2 > v0 else "↓" if v2 < v0 else "→"
            print(f"    {metric:<20s}: {v0:.3f} → {v1:.3f} → {v2:.3f}  {trend}")

    # Save
    out_path = Path("eval/gguf_nw_csr/retroactive_topk.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
