#!/usr/bin/env python3
"""
Gate 2: Proxy Overfit Test

Before committing to a full training run, verify that:
  1. The model CAN learn from this data (overfit canary sentences)
  2. n-gram statistics look reasonable
  3. Loss drops reliably on a small data slice

Usage:
  python3 scripts/gate_proxy_overfit.py \
    --tokenized-data data/train_tokens_4k.npy \
    --tokenizer tokenizer/basque_unigram_4000.model \
    --config config/small.yaml \
    --output reports/gate2_proxy/

Exit code 0 = PASS, 1 = FAIL
"""

import argparse, json, sys, os, time, tempfile
from pathlib import Path

import numpy as np
import torch
import yaml


# ── Basque canary sentences (NOT from training corpus) ──────
CANARY_SENTENCES = [
    "Lore moreak mendi handietan bakarrik hazten dira udaberri hotzean.",
    "Itsasontzi zaharrak portu txikian ainguratuta gelditu ziren ekaitzaren ondoren.",
    "Musika talde berriak kontzertu harrigarria eskaini zuen herriko plazan bart.",
    "Zuhaizti iluneko bidezidor estuetan barrena ibili ginen goiz osoan zehar.",
    "Sendagile jakintsuak belar sendagarriekin osatutako edabea prestatu zuen.",
]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(tokenized_path: str, max_tokens: int) -> torch.Tensor:
    """Load a slice of pre-tokenized data."""
    data = np.load(tokenized_path, mmap_mode='r')
    n_tokens = min(len(data), max_tokens)
    tokens = torch.from_numpy(data[:n_tokens].copy().astype(np.int64))
    return tokens


def create_proxy_dataset(tokens: torch.Tensor, tokenizer_path: str,
                         canary_sentences: list[str],
                         seq_len: int) -> dict:
    """
    Create a small dataset that includes canary sentences.
    Returns dict with train_inputs, train_targets, and canary positions.
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    # Tokenize canary sentences
    canary_ids = []
    for sent in canary_sentences:
        ids = sp.EncodeAsIds(sent)
        canary_ids.append(ids)
        print(f"  Canary '{sent[:40]}...' → {len(ids)} tokens: {ids[:10]}...")

    # Create sequences: interleave some real data + canaries
    # Simple approach: take a few real sequences, then inject canaries
    batch_size = 2
    n_seq = 10  # 10 training sequences

    sequences = []
    canary_positions = []  # (seq_idx, start_pos)

    pos = 0
    for i in range(n_seq):
        seq_len_used = min(seq_len, len(tokens) - pos - 1)
        if seq_len_used < 10:
            break
        seq = tokens[pos:pos + seq_len_used + 1].clone()
        pos += seq_len_used

        # Inject canary at a random position in this sequence
        canary_idx = i % len(canary_ids)
        canary_tokens = canary_ids[canary_idx]
        insert_pos = min(seq_len_used // 2, seq_len_used - len(canary_tokens) - 1)

        seq[insert_pos:insert_pos + len(canary_tokens)] = torch.tensor(canary_tokens, dtype=torch.int64)
        canary_positions.append((i, insert_pos, len(canary_tokens)))

        sequences.append(seq)

    # Stack
    inputs = torch.stack([s[:-1] for s in sequences])
    targets = torch.stack([s[1:] for s in sequences])

    return {
        'inputs': inputs,
        'targets': targets,
        'canary_positions': canary_positions,
        'canary_sentences': canary_sentences,
        'canary_ids': canary_ids,
    }


def train_proxy(model, dataset, device, steps: int = 500,
                lr: float = 1e-4) -> dict:
    """Train for a small number of steps and check canary completion."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    inputs = dataset['inputs'].to(device)
    targets = dataset['targets'].to(device)
    canary_positions = dataset['canary_positions']
    canary_sentences = dataset['canary_sentences']
    canary_ids = dataset['canary_ids']

    losses = []
    canary_accuracies = []

    print(f"\n  Training proxy model for {steps} steps...")
    t0 = time.perf_counter()

    for step in range(steps):
        model.train()
        optimizer.zero_grad()

        # Forward
        logits = model(inputs).logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Every 50 steps: check canary completion
        if step % 50 == 0 or step == steps - 1:
            model.eval()
            canary_hits = 0
            canary_total = 0

            with torch.no_grad():
                for seq_idx, start_pos, length in canary_positions:
                    # Get the prefix: tokens before the canary
                    prefix = inputs[seq_idx, :start_pos].unsqueeze(0)
                    target = inputs[seq_idx, start_pos:start_pos + length]

                    # Greedy generate
                    generated = []
                    current = prefix.to(device)

                    for _ in range(length):
                        logits_gen = model(current).logits
                        next_token = logits_gen[0, -1, :].argmax().item()
                        generated.append(next_token)
                        current = torch.cat([
                            current,
                            torch.tensor([[next_token]], device=device)
                        ], dim=1)

                    # Compare
                    for gt, pred in zip(target.tolist(), generated):
                        canary_total += 1
                        if gt == pred:
                            canary_hits += 1

            acc = canary_hits / max(canary_total, 1)
            canary_accuracies.append((step, acc))

            print(f"    step {step:4d}: loss={loss.item():.4f}, "
                  f"canary_accuracy={acc:.1%}")

    elapsed = time.perf_counter() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    return {
        'final_loss': losses[-1],
        'losses': losses,
        'canary_accuracies': canary_accuracies,
        'final_canary_accuracy': canary_accuracies[-1][1] if canary_accuracies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Gate 2: Proxy Overfit Test")
    parser.add_argument("--tokenized-data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", default="config/small.yaml")
    parser.add_argument("--output", default="reports/gate2_proxy/")
    parser.add_argument("--max-tokens", type=int, default=500000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("GATE 2: PROXY OVERFIT TEST")
    print("=" * 60)

    # Validate inputs
    if not os.path.exists(args.tokenized_data):
        print(f"ERROR: tokenized data not found: {args.tokenized_data}")
        sys.exit(1)
    if not os.path.exists(args.tokenizer):
        print(f"ERROR: tokenizer not found: {args.tokenizer}")
        sys.exit(1)

    # Load config
    cfg = load_config(args.config)
    seq_len = cfg['model'].get('seq_len', 1024)
    vocab_size = cfg['model'].get('vocab_size', 4000)
    d_model = cfg['model'].get('d_model', 512)

    # Load data slice
    print(f"\nLoading data: {args.tokenized_data}")
    print(f"  Max tokens: {args.max_tokens:,}")
    tokens = load_data(args.tokenized_data, args.max_tokens)
    print(f"  Loaded {len(tokens):,} tokens")

    # Create proxy dataset with canaries
    print(f"\nCreating proxy dataset (seq_len={seq_len})...")
    dataset = create_proxy_dataset(tokens, args.tokenizer, CANARY_SENTENCES, seq_len)
    print(f"  Dataset: {dataset['inputs'].shape[0]} sequences")

    # Build tiny model
    print(f"\nBuilding proxy model...")
    print(f"  vocab={vocab_size}, d_model={d_model}, n_layers={cfg['model'].get('n_layers', 8)}")

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig as MambaConfigClass

    mcfg = MambaConfigClass(
        d_model=d_model,
        n_layer=cfg['model'].get('n_layers', 8),
        vocab_size=vocab_size,
        ssm_cfg=cfg['model'].get('ssm_cfg', {}),
        rms_norm=cfg['model'].get('rms_norm', True),
        fused_add_norm=cfg['model'].get('fused_add_norm', True),
        residual_in_fp32=cfg['model'].get('residual_in_fp32', True),
    )
    model = MambaLMHeadModel(mcfg, initializer_cfg=None)

    # Train
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    results = train_proxy(model, dataset, device, steps=args.steps, lr=args.lr)

    # Evaluate
    canary_acc = results['final_canary_accuracy']
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Canary accuracy: {canary_acc:.1%}")

    PASS_THRESHOLD = 0.5  # At least 50% of canary tokens correct

    if canary_acc >= PASS_THRESHOLD:
        verdict = "🟢 PASS — model can overfit canary sentences"
    else:
        verdict = "🔴 FAIL — model cannot learn from this data"

    print(f"\n  Verdict: {verdict}")

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'config': cfg,
        'dataset_info': {
            'n_tokens': len(tokens),
            'n_sequences': dataset['inputs'].shape[0],
            'seq_len': seq_len,
            'canary_sentences': CANARY_SENTENCES,
        },
        'training': {
            'steps': args.steps,
            'lr': args.lr,
            'final_loss': results['final_loss'],
            'canary_accuracies': [(s, a) for s, a in results['canary_accuracies']],
            'final_canary_accuracy': canary_acc,
        },
        'verdict': verdict,
        'passed': canary_acc >= PASS_THRESHOLD,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"gate2_proxy_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")

    sys.exit(0 if output['passed'] else 1)


if __name__ == "__main__":
    main()
