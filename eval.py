"""
Evaluate Morpheus v2 Mamba on next-word prediction accuracy.

Computes Hit@K metrics on a test set: what fraction of the time does
the actual next word appear in the model's top-K predictions?

Usage:
    python eval.py \\
        --checkpoint checkpoints/best.pt \\
        --tokenizer tokenizer/basque_unigram_32k.model \\
        --test-file data/splits/test/test.txt \\
        --k 1,3,5

Full implementation: Morpheus_v2_Mamba.md §6.2
"""

import argparse
import torch
import sentencepiece as spm
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def evaluate_hit_at_k(model, sp, test_sentences, device="cuda", k_values=(1, 3, 5)):
    """Compute Hit@K for next-word prediction on test sentences.

    For each position in each sentence (after at least 2 context tokens),
    predict the next token and check if the actual token is in top-K.

    Args:
        model: MambaLMHeadModel instance
        sp: SentencePieceProcessor
        test_sentences: list of raw text strings
        device: torch device
        k_values: tuple of K values for Hit@K

    Returns:
        dict with Hit@K percentages and total predictions
    """
    model.eval()
    hits = {k: 0 for k in k_values}
    total = 0

    for sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=int)
        if len(tokens) < 3:
            continue

        for i in range(2, len(tokens)):
            context = torch.tensor([tokens[:i]], device=device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(context)
                logits = output.logits[0, -1, :]  # Last position

            target = tokens[i]
            top_k_preds = torch.topk(logits, max(k_values)).indices.tolist()

            for k in k_values:
                if target in top_k_preds[:k]:
                    hits[k] += 1
            total += 1

    results = {}
    for k in k_values:
        results[f"Hit@{k}"] = hits[k] / total if total > 0 else 0.0
    results["total_predictions"] = total
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Morpheus v2 Mamba")
    parser.add_argument("--checkpoint", required=True, help="Path to training checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to SentencePiece model")
    parser.add_argument("--test-file", required=True, help="Path to test file (one sentence per line)")
    parser.add_argument("--k", default="1,3,5", help="Comma-separated K values")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--max-sentences", type=int, default=10000,
                        help="Maximum sentences to evaluate")
    args = parser.parse_args()

    k_values = tuple(int(k) for k in args.k.split(","))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    from mamba_ssm.models.config_mamba import MambaConfig

    config = MambaConfig(**{k: v for k, v in ckpt["config"].items()
                            if k in MambaConfig.__dataclass_fields__})
    model = MambaLMHeadModel(config, device=device, dtype=torch.bfloat16)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    # Load test sentences
    with open(args.test_file) as f:
        test_sentences = [line.strip() for line in f if line.strip()]

    if len(test_sentences) > args.max_sentences:
        test_sentences = test_sentences[:args.max_sentences]

    print(f"Evaluating on {len(test_sentences):,} sentences...")

    # Evaluate
    results = evaluate_hit_at_k(model, sp, test_sentences, device, k_values)

    print(f"\nResults:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f} ({value*100:.1f}%)")
        else:
            print(f"  {metric}: {value:,}")

    return results


if __name__ == "__main__":
    main()
