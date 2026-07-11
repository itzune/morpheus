#!/usr/bin/env python3
"""
Reveal A/B evaluation results after the expert has judged.

Reads the key (which checkpoint is A/B per prompt) and the expert's judgments,
then reports which checkpoint won overall.

Usage:
    python3 scripts/ab_eval_reveal.py --judgments "1:A 2:B 3:T 4:A ..."
    python3 scripts/ab_eval_reveal.py --judgments-file eval/ab_eval/judgments.txt
"""
import argparse
import json
import re
from pathlib import Path

KEY_FILE = Path("eval/ab_eval/key.json")


def parse_judgments(text):
    """Parse '1:A 2:B 3:T 4:A' or '1=A 2=B' or newline-separated into {n: verdict}."""
    judgments = {}
    # Match patterns like "1:A", "1=A", "1: A", "1. A"
    for m in re.finditer(r'(\d+)\s*[:=.]?\s*([ABTab])', text, re.IGNORECASE):
        n = int(m.group(1))
        verdict = m.group(2).upper()
        if verdict == 'T':
            verdict = 'T'
        judgments[n] = verdict
    return judgments


def main():
    parser = argparse.ArgumentParser(description="Reveal A/B eval results")
    parser.add_argument("--judgments", default=None, help="Judgments as string: '1:A 2:B 3:T ...'")
    parser.add_argument("--judgments-file", default=None, help="File containing judgments")
    parser.add_argument("--key-file", default=str(KEY_FILE))
    args = parser.parse_args()

    if args.judgments:
        judgments_text = args.judgments
    elif args.judgments_file:
        judgments_text = Path(args.judgments_file).read_text()
    else:
        parser.error("Provide --judgments or --judgments-file")

    judgments = parse_judgments(judgments_text)
    print(f"Parsed {len(judgments)} judgments: {judgments}")
    print()

    key = json.loads(Path(args.key_file).read_text())
    step_a = key["ckpt_a_step"]
    step_b = key["ckpt_b_step"]
    assignments = key["assignments"]

    # Tally
    wins_a_step = 0  # judgments where the earlier checkpoint won
    wins_b_step = 0  # judgments where the later checkpoint won
    ties = 0
    detail = []

    for entry in assignments:
        n = entry["n"]
        if n not in judgments:
            detail.append(f"  {n}: (no judgment)")
            continue
        verdict = judgments[n]
        # A_is_step tells us which checkpoint is shown as "A"
        a_step = entry["A_is_step"]
        b_step = entry["B_is_step"]
        gold = entry.get("gold", "")

        if verdict == "A":
            winner_step = a_step
        elif verdict == "B":
            winner_step = b_step
        else:  # T
            winner_step = None

        if winner_step is None:
            ties += 1
            label = "TIE"
        elif winner_step == step_a:
            wins_a_step += 1
            label = f"step {step_a} wins"
        else:
            wins_b_step += 1
            label = f"step {step_b} wins"

        detail.append(f"  {n}: {verdict} → {label}  [gold: {gold}]")

    total_judged = wins_a_step + wins_b_step + ties
    total_decisive = wins_a_step + wins_b_step

    print(f"{'='*60}")
    print(f"  A/B EVALUATION RESULTS (REVEALED)")
    print(f"{'='*60}")
    print(f"  Checkpoint A (file): {key['ckpt_a_file']} (step {step_a})")
    print(f"  Checkpoint B (file): {key['ckpt_b_file']} (step {step_b})")
    print(f"  Judgments: {total_judged} total ({total_decisive} decisive, {ties} ties)")
    print()
    print(f"  Step {step_a} wins: {wins_a_step}")
    print(f"  Step {step_b} wins: {wins_b_step}")
    print(f"  Ties:               {ties}")
    print()

    if total_decisive > 0:
        pct_a = wins_a_step / total_decisive * 100
        pct_b = wins_b_step / total_decisive * 100
        print(f"  Among decisive judgments:")
        print(f"    Step {step_a}: {pct_a:.1f}%")
        print(f"    Step {step_b}: {pct_b:.1f}%")
        print()
        if wins_b_step > wins_a_step:
            print(f"  → Step {step_b} (the LATER checkpoint) is preferred by the expert.")
        elif wins_a_step > wins_b_step:
            print(f"  → Step {step_a} (the EARLIER checkpoint) is preferred by the expert.")
        else:
            print(f"  → Tied among decisive judgments.")
    print()
    print(f"{'='*60}")
    print(f"  Per-prompt detail:")
    for line in detail:
        print(line)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
