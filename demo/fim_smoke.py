#!/usr/bin/env python3
"""Quick FIM smoke test: 5 examples, print actual generated text.

Used to check if Phase 6 v2 (token-level FIM + packed) fixed the v1 problems:
  - Does the model generate <EOT>?
  - Is the output coherent (not repetitive garbage)?
"""
import sys, random, torch
from pathlib import Path
import sentencepiece as spm

# build_fim is in scripts/pipeline/, fim_eval is in scripts/
_scripts = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, _scripts)
sys.path.insert(0, str(Path(_scripts) / "pipeline"))

from build_fim import make_fim_split_tokens
from fim_eval import load_checkpoint, FIM_TOKENS, EOS_ID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hardcoded test sentences (Basque) with known middles
# Fill-in-the-blank style: strong contextual constraint, single correct answer
TESTS = [
    ("Ni atzo amonaren etxera joan nintzen bazkaltzera.", "nintzen"),
    ("Zein da zure izena? Ni Xabi naiz.", "izena"),
    ("Urrutiko intxaurrak hamalau.", "intxaurrak"),
    ("Zuek animatuko zarete etxera ostiralean bazkaltzera?", "zarete"),
]

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/phase6/best.pt"
    sp_path = sys.argv[2] if len(sys.argv) > 2 else "tokenizer/basque_unigram_fim.model"

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)

    print(f"Loading checkpoint: {ckpt}")
    model, vocab_size = load_checkpoint(ckpt, DEVICE)
    print(f"Model loaded (vocab={vocab_size})\n")

    random.seed(42)
    eot_count = 0

    for full_text, target_middle in TESTS:
        # Tokenize full line
        full_ids = sp.encode(full_text, out_type=int)
        if EOS_ID not in full_ids:
            full_ids = full_ids + [EOS_ID]

        # Make FIM split targeting the known middle
        # Find the target middle tokens
        mid_tokens = sp.encode(target_middle, out_type=int)

        # Find target position in full_ids
        mid_len = len(mid_tokens)
        mid_start = -1
        for i in range(len(full_ids) - mid_len + 1):
            if full_ids[i:i+mid_len] == mid_tokens:
                mid_start = i
                break

        if mid_start == -1:
            # Fallback: random split
            result = make_fim_split_tokens(full_ids, sp, seed=random.randint(0, 999999))
        else:
            # Build split manually at the target
            prefix_ids = full_ids[:mid_start]
            suffix_ids = full_ids[mid_start + mid_len:]
            middle_ids = full_ids[mid_start:mid_start + mid_len]
            result = ("psm", prefix_ids, suffix_ids, middle_ids)

        fmt, prefix_ids, suffix_ids, middle_ids = result

        # Build FIM prompt (PSM format)
        prompt = ([FIM_TOKENS["<PRE>"]] + prefix_ids +
                  [FIM_TOKENS["<SUF>"]] + suffix_ids +
                  [FIM_TOKENS["<MID>"]])

        # Generate
        ids = list(prompt)
        generated = []
        stopped_on_eot = False
        with torch.no_grad():
            for _ in range(64):
                x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                logits = model(x).logits[:, -1, :]
                next_id = int(logits.argmax(dim=-1).item())
                if next_id == FIM_TOKENS["<EOT>"] or next_id == EOS_ID:
                    if next_id == FIM_TOKENS["<EOT>"]:
                        stopped_on_eot = True
                    break
                generated.append(next_id)
                ids.append(next_id)

        # Decode
        gen_text = sp.decode(generated) if generated else "(empty)"
        ref_text = sp.decode(middle_ids)

        # Check for repetition
        is_repetitive = False
        if len(generated) > 4:
            unique = len(set(generated))
            if unique < len(generated) * 0.4:
                is_repetitive = True

        print(f"─" * 60)
        print(f"Text:    {full_text}")
        print(f"Target:  {ref_text}")
        print(f"Got:     {gen_text}")
        print(f"Tokens:  {len(generated)} gen, stopped={'<EOT>' if stopped_on_eot else 'max/other'}")
        print(f"Repetitive: {'⚠️ YES' if is_repetitive else 'no'}")

        if stopped_on_eot:
            eot_count += 1

    print(f"\n{'='*60}")
    print(f"<EOT> generated: {eot_count}/{len(TESTS)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
